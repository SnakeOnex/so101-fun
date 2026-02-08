"""
SO-101 Training Script
======================
Trains RL (PPO) or BC policies for SO-101 arm tasks.

Tasks (--task):
  reach   : Move EE to a target position (simplest, validates pipeline)
  hover   : Position EE above an object at hover height
  grasp   : Full approach-from-above with orientation + height control

Usage:
  # Train reach task (simplest, start here)
  python train.py --task=reach --action_mode=joint_delta

  # Train reach with random targets
  python train.py --task=reach --reach_target=random

  # Train hover task
  python train.py --task=hover

  # Train grasp task (full complexity)
  python train.py --task=grasp

  # Disable wandb (use tensorboard only)
  python train.py --task=reach --no_wandb

  # BC stage (requires RL trained first)
  python train.py --task=grasp --stage=bc
"""

import argparse
import os
import pickle
import re
import tempfile
from importlib import metadata
from pathlib import Path

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e

from rsl_rl.runners import OnPolicyRunner
from so101_behavior_cloning import BehaviorCloning

import genesis as gs

from envs import TASK_REGISTRY, make_env


# ---------------------------------------------------------------------------
# wandb compatibility: patch store_config to handle plain dicts
# (rsl-rl's WandbSummaryWriter calls asdict(env_cfg) which crashes on dicts)
# ---------------------------------------------------------------------------
def _patch_wandb_store_config():
    """Monkey-patch WandbSummaryWriter.store_config to handle dict configs."""
    try:
        from rsl_rl.utils.wandb_utils import WandbSummaryWriter
        import wandb

        def _store_config_dict_safe(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
            # env_cfg might be a dict or a dataclass. Handle both.
            if isinstance(env_cfg, dict):
                wandb.config.update({"env_cfg": env_cfg})
            else:
                from dataclasses import asdict
                wandb.config.update({"env_cfg": asdict(env_cfg)})
            wandb.config.update({"runner_cfg": runner_cfg})
            wandb.config.update({"policy_cfg": policy_cfg})
            wandb.config.update({"alg_cfg": alg_cfg})

        WandbSummaryWriter.store_config = _store_config_dict_safe
    except ImportError:
        pass  # wandb not installed, no patching needed


# ---------------------------------------------------------------------------
# Training configs
# ---------------------------------------------------------------------------

def get_train_cfg(exp_name: str, max_iterations: int, use_wandb: bool, wandb_project: str):
    """PPO and BC training configurations."""
    rl_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "relu",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    # wandb logger config
    if use_wandb:
        rl_cfg_dict["logger"] = "wandb"
        rl_cfg_dict["wandb_project"] = wandb_project
    else:
        rl_cfg_dict["logger"] = "tensorboard"

    bc_cfg_dict = {
        "num_steps_per_env": 24,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "num_mini_batches": 10,
        "max_grad_norm": 1.0,
        "policy": {
            "vision_encoder": {
                "conv_layers": [
                    {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
                    {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
                    {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1},
                ],
                "pooling": "adaptive_avg",
            },
            "action_head": {
                "state_obs_dim": 7,
                "hidden_dims": [128, 128, 64],
            },
            "pose_head": {
                "hidden_dims": [64, 64],
            },
        },
        "buffer_size": 1000,
        "log_freq": 10,
        "save_freq": 50,
        "eval_freq": 50,
    }

    return rl_cfg_dict, bc_cfg_dict


def load_teacher_policy(env, rl_train_cfg, log_dir):
    """Load trained RL policy to use as teacher for BC stage."""
    assert log_dir.exists(), f"Log directory {log_dir} does not exist. Train RL first!"
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    try:
        *_, last_ckpt = sorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    runner = OnPolicyRunner(env, rl_train_cfg, str(log_dir), device=gs.device)
    runner.load(last_ckpt)
    print(f"Loaded teacher policy from {last_ckpt}")
    return runner.get_inference_policy(device=gs.device)


# ---------------------------------------------------------------------------
# Post-training evaluation + video upload
# ---------------------------------------------------------------------------

def run_eval_and_log_video(
    task: str,
    action_mode: str,
    log_dir: Path,
    rl_train_cfg: dict,
    use_wandb: bool,
    num_eval_envs: int = 4,
    num_eval_episodes: int = 1,
):
    """
    Run evaluation after training, record video, upload to wandb.
    Creates a separate small env for eval to avoid interfering with training.
    """
    import torch

    print("\n" + "=" * 60)
    print("Post-training evaluation")
    print("=" * 60)

    # Find latest checkpoint
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        print("No checkpoints found, skipping eval.")
        return
    *_, last_ckpt = sorted(checkpoint_files)
    print(f"Evaluating checkpoint: {last_ckpt.name}")

    # Create eval env with vis camera enabled
    eval_env = make_env(
        task=task,
        num_envs=num_eval_envs,
        action_mode=action_mode,
        show_viewer=False,
        env_overrides={"visualize_camera": True},
    )

    # Load policy
    runner = OnPolicyRunner(eval_env, rl_train_cfg, str(log_dir), device=gs.device)
    runner.load(last_ckpt)
    policy = runner.get_inference_policy(device=gs.device)

    # Run eval episodes
    obs, _ = eval_env.reset()
    total_reward = torch.zeros(num_eval_envs, device=gs.device)
    max_steps = eval_env.max_episode_length

    eval_env.vis_cam.start_recording()
    with torch.no_grad():
        for step in range(max_steps):
            actions = policy(obs)
            eval_env.vis_cam.render()
            obs, rewards, dones, infos = eval_env.step(actions)
            total_reward += rewards

    mean_reward = total_reward.mean().item()
    print(f"Eval mean reward: {mean_reward:.4f}")

    # Save video
    video_dir = Path("eval_videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{task}_{action_mode}_eval.mp4"
    eval_env.vis_cam.stop_recording(
        save_to_filename=str(video_path),
        fps=30,
    )
    print(f"Eval video saved to {video_path}")

    # Upload to wandb
    if use_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/video": wandb.Video(str(video_path), fps=30, format="mp4"),
                })
                print("Eval video and metrics uploaded to wandb.")
            else:
                print("wandb run not active, skipping upload.")
        except ImportError:
            print("wandb not installed, skipping upload.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SO-101 Training")
    parser.add_argument("-e", "--exp_name", type=str, default="so101")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--stage", type=str, default="rl", choices=["rl", "bc"])
    parser.add_argument(
        "--task", type=str, default="reach",
        choices=list(TASK_REGISTRY.keys()),
        help="Task to train on: reach, hover, grasp",
    )
    parser.add_argument(
        "--action_mode", type=str, default="joint_delta",
        choices=["joint_delta", "ee_delta_pos", "ee_delta"],
        help="Action space: joint_delta (5D), ee_delta_pos (3D), ee_delta (6D)",
    )
    parser.add_argument(
        "--reach_target", type=str, default="fixed",
        choices=["fixed", "random"],
        help="Reach task target mode (only used with --task=reach)",
    )
    # wandb
    parser.add_argument("--wandb_project", type=str, default="so101-rl")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    # Reward scale overrides (for sweeps)
    parser.add_argument("--reward_overrides", type=str, default=None,
                        help="JSON string of reward scale overrides, e.g. '{\"xy_align\": 2.0}'")
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    # Patch wandb before anything else
    if use_wandb:
        _patch_wandb_store_config()

    # Check for wandb sweep overrides
    reward_overrides = None
    if args.reward_overrides:
        import json
        reward_overrides = json.loads(args.reward_overrides)

    # Check wandb sweep config
    if use_wandb:
        try:
            import wandb
            # If running as a wandb sweep agent, pull reward overrides from wandb.config
            if wandb.run is not None and hasattr(wandb.config, "items"):
                sweep_overrides = {}
                for key, val in wandb.config.items():
                    if key.startswith("reward_"):
                        reward_key = key[len("reward_"):]
                        sweep_overrides[reward_key] = val
                if sweep_overrides:
                    reward_overrides = reward_overrides or {}
                    reward_overrides.update(sweep_overrides)
                    print(f"Applied wandb sweep reward overrides: {sweep_overrides}")
        except ImportError:
            pass

    # === init ===
    gs.init(
        backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True
    )

    # === env overrides ===
    env_overrides = {}
    if args.task == "reach":
        env_overrides["reach_target_mode"] = args.reach_target

    # === create env ===
    num_envs = args.num_envs if args.stage == "rl" else 10
    env = make_env(
        task=args.task,
        num_envs=num_envs,
        action_mode=args.action_mode,
        show_viewer=args.vis,
        env_overrides=env_overrides,
        reward_overrides=reward_overrides,
    )

    # === training configs ===
    exp_name = f"{args.exp_name}_{args.task}"
    rl_train_cfg, bc_train_cfg = get_train_cfg(
        exp_name, args.max_iterations, use_wandb, args.wandb_project
    )

    # === log dir ===
    log_dir = Path("logs") / f"{exp_name}_{args.action_mode}_{args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save configs for eval script
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump({
            "env_cfg": env.env_cfg,
            "reward_scales": {k: v / env.ctrl_dt for k, v in env.reward_scales.items()},
            "robot_cfg": env.robot._args,
            "rl_train_cfg": rl_train_cfg,
            "bc_train_cfg": bc_train_cfg,
            "task": args.task,
            "action_mode": args.action_mode,
        }, f)

    # === train ===
    if args.stage == "bc":
        rl_log_dir = Path("logs") / f"{exp_name}_{args.action_mode}_rl"
        teacher_policy = load_teacher_policy(env, rl_train_cfg, rl_log_dir)
        runner = BehaviorCloning(env, bc_train_cfg, teacher_policy, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=str(log_dir))
    else:
        runner = OnPolicyRunner(env, rl_train_cfg, str(log_dir), device=gs.device)
        runner.learn(
            num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
        )

    # === post-training eval + video ===
    if args.stage == "rl":
        run_eval_and_log_video(
            task=args.task,
            action_mode=args.action_mode,
            log_dir=log_dir,
            rl_train_cfg=rl_train_cfg,
            use_wandb=use_wandb,
        )

    # Finish wandb
    if use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass

    print("\nDone!")


if __name__ == "__main__":
    main()

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
import copy
import os
import pickle
import re
import tempfile
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

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
        *_, last_ckpt = sorted(
            checkpoint_files,
            key=lambda f: int(re.search(r"model_(\d+)", f.name).group(1)),
        )
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    runner = OnPolicyRunner(env, rl_train_cfg, str(log_dir), device=gs.device)
    runner.load(last_ckpt)
    print(f"Loaded teacher policy from {last_ckpt}")
    return runner.get_inference_policy(device=gs.device)


# ---------------------------------------------------------------------------
# Post-training evaluation + video upload
# ---------------------------------------------------------------------------

def _make_camera_grid(frames: list[torch.Tensor], grid_rows: int = 3, grid_cols: int = 3) -> np.ndarray:
    """
    Compose per-env camera frames into a grid image.

    Args:
        frames: list of [H, W, 3] uint8 tensors, one per env
        grid_rows, grid_cols: grid layout

    Returns:
        [grid_H, grid_W, 3] uint8 numpy array
    """
    n = min(len(frames), grid_rows * grid_cols)
    h, w = frames[0].shape[:2]
    grid = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, grid_cols)
        img = frames[i].cpu().numpy() if isinstance(frames[i], torch.Tensor) else frames[i]
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return grid


def _stereo_grid_frames(rgb_obs: torch.Tensor, num_envs: int) -> dict[str, np.ndarray]:
    """
    Extract left and right camera views from stereo tensor and compose grid images.

    Args:
        rgb_obs: [B, 6, H, W] normalized stereo tensor (channels 0-2: left, 3-5: right)
        num_envs: number of envs to include in grid (max 9 for 3x3)

    Returns:
        dict with "left" and "right" keys mapping to [grid_H, grid_W, 3] uint8 arrays
    """
    grids = {}
    for name, ch_start in [("left", 0), ("right", 3)]:
        rgb = rgb_obs[:, ch_start:ch_start + 3]  # [B, 3, H, W]
        frames_hwc = (rgb.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8)
        frame_list = [frames_hwc[i] for i in range(min(num_envs, 9))]
        grids[name] = _make_camera_grid(frame_list)
    return grids


def run_bc_eval_and_log_video(
    task: str,
    action_mode: str,
    log_dir: Path,
    bc_train_cfg: dict,
    use_wandb: bool,
    env_overrides: dict | None = None,
    num_eval_envs: int = 9,
):
    """
    Evaluate a trained BC policy, record debug camera + stereo camera grid videos, upload to wandb.
    """
    import imageio

    print("\n" + "=" * 60)
    print("BC Post-training evaluation")
    print("=" * 60)

    # Find latest BC checkpoint (checkpoint_NNNN.pt)
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        print("No BC checkpoints found, skipping eval.")
        return
    *_, last_ckpt = sorted(
        checkpoint_files,
        key=lambda f: int(re.search(r"checkpoint_(\d+)", f.name).group(1)),
    )
    print(f"Evaluating BC checkpoint: {last_ckpt.name}")

    # Create eval env: needs BOTH debug camera (overhead) and vision cameras (stereo)
    eval_overrides = dict(env_overrides or {})
    eval_overrides["visualize_camera"] = True
    eval_overrides["use_vision_cameras"] = True
    eval_env = make_env(
        task=task,
        num_envs=num_eval_envs,
        action_mode=action_mode,
        show_viewer=False,
        env_overrides=eval_overrides,
    )

    # Load BC policy from checkpoint
    from so101_behavior_cloning import Policy
    checkpoint = torch.load(last_ckpt, map_location=gs.device, weights_only=False)
    policy = Policy(bc_train_cfg["policy"], eval_env.num_actions).to(gs.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    # Run eval episode
    obs, _ = eval_env.reset()
    total_reward = torch.zeros(num_eval_envs, device=gs.device)
    max_steps = eval_env.max_episode_length

    video_fps = 30
    render_interval = max(1, round(1.0 / (eval_env.ctrl_dt * video_fps)))

    # Collect grid frames for both stereo cameras
    cam_grid_frames: dict[str, list[np.ndarray]] = {"left": [], "right": []}

    eval_env.vis_cam.start_recording()
    with torch.no_grad():
        for step in range(max_steps):
            # Get stereo images and ee_pose for BC policy
            rgb_obs = eval_env.get_stereo_rgb_images(normalize=True)
            ee_pose = eval_env.robot.ee_pose

            # BC forward pass
            actions = policy(rgb_obs.float(), ee_pose.float())

            # Render debug camera for overhead video
            if step % render_interval == 0:
                eval_env.vis_cam.render()

                # Capture stereo camera grid frames
                grids = _stereo_grid_frames(rgb_obs, num_eval_envs)
                for name in cam_grid_frames:
                    cam_grid_frames[name].append(grids[name])

            obs, rewards, dones, infos = eval_env.step(actions)
            total_reward += rewards

    mean_reward = total_reward.mean().item()
    print(f"BC Eval mean reward: {mean_reward:.4f}")

    # Save overhead debug camera video
    video_dir = Path("eval_videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    overhead_video_path = video_dir / f"{task}_{action_mode}_bc_eval.mp4"
    eval_env.vis_cam.stop_recording(
        save_to_filename=str(overhead_video_path),
        fps=video_fps,
    )
    if overhead_video_path.exists():
        print(f"BC overhead eval video saved to {overhead_video_path}")

    # Save stereo camera grid videos
    cam_video_paths: dict[str, Path] = {}
    for name, frames in cam_grid_frames.items():
        if frames:
            path = video_dir / f"{task}_{action_mode}_bc_{name}_cam.mp4"
            imageio.mimwrite(str(path), frames, fps=video_fps)
            cam_video_paths[name] = path
            print(f"BC {name} camera grid video saved to {path}")

    # Upload to wandb
    if use_wandb:
        try:
            import wandb

            if wandb.run is not None:
                log_data = {"bc_eval/mean_reward": mean_reward}
                if overhead_video_path.exists():
                    log_data["bc_eval/overhead_video"] = wandb.Video(str(overhead_video_path), format="mp4")
                for name, path in cam_video_paths.items():
                    if path.exists():
                        log_data[f"bc_eval/{name}_cam_video"] = wandb.Video(str(path), format="mp4")
                wandb.log(log_data)
                print("BC eval videos and metrics uploaded to wandb.")
            else:
                print("wandb run not active, skipping upload.")
        except ImportError:
            print("wandb not installed, skipping upload.")


def run_eval_and_log_video(
    task: str,
    action_mode: str,
    log_dir: Path,
    rl_train_cfg: dict,
    use_wandb: bool,
    env_overrides: dict | None = None,
    num_eval_envs: int = 9,
    num_eval_episodes: int = 1,
):
    """
    Run evaluation after training, record video, upload to wandb.
    Creates a separate small env for eval to avoid interfering with training.
    """
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
    *_, last_ckpt = sorted(
        checkpoint_files,
        key=lambda f: int(re.search(r"model_(\d+)", f.name).group(1)),
    )
    print(f"Evaluating checkpoint: {last_ckpt.name}")

    # Create eval env with vis camera enabled, preserving training env settings
    eval_overrides = dict(env_overrides or {})
    eval_overrides["visualize_camera"] = True
    eval_env = make_env(
        task=task,
        num_envs=num_eval_envs,
        action_mode=action_mode,
        show_viewer=False,
        env_overrides=eval_overrides,
    )

    # Load policy (deep copy config because OnPolicyRunner.pop() mutates it)
    runner = OnPolicyRunner(eval_env, copy.deepcopy(rl_train_cfg), str(log_dir), device=gs.device)
    runner.load(last_ckpt)
    policy = runner.get_inference_policy(device=gs.device)

    # Run eval episodes
    obs, _ = eval_env.reset()
    total_reward = torch.zeros(num_eval_envs, device=gs.device)
    max_steps = eval_env.max_episode_length

    # Render every Nth step for real-time playback.
    # Sim at 100Hz, video at 30fps → render every ~3 steps.
    video_fps = 30
    render_interval = max(1, round(1.0 / (eval_env.ctrl_dt * video_fps)))

    eval_env.vis_cam.start_recording()
    with torch.no_grad():
        for step in range(max_steps):
            actions = policy(obs)
            if step % render_interval == 0:
                eval_env.vis_cam.render()
            obs, rewards, dones, infos = eval_env.step(actions)
            total_reward += rewards

    mean_reward = total_reward.mean().item()
    print(f"Eval mean reward: {mean_reward:.4f}")

    # Save video
    # debug=True camera uses Rasterizer and produces a single video file
    # at the exact path we specify (no per-env _0 suffix).
    video_dir = Path("eval_videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{task}_{action_mode}_eval.mp4"
    eval_env.vis_cam.stop_recording(
        save_to_filename=str(video_path),
        fps=30,
    )
    if video_path.exists():
        print(f"Eval video saved to {video_path}")
    else:
        print(f"Warning: expected video at {video_path} but file not found.")

    # Upload to wandb
    if use_wandb:
        try:
            import wandb

            if wandb.run is not None and video_path.exists():
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/video": wandb.Video(str(video_path), format="mp4"),
                })
                print("Eval video and metrics uploaded to wandb.")
            elif wandb.run is not None:
                wandb.log({"eval/mean_reward": mean_reward})
                print("Eval metrics uploaded (video file not found).")
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
    if args.stage == "bc":
        env_overrides["use_vision_cameras"] = True

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
        runner = BehaviorCloning(env, bc_train_cfg, teacher_policy, device=gs.device, use_wandb=use_wandb, wandb_project="so101-bc")
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=str(log_dir))
    else:
        runner = OnPolicyRunner(env, copy.deepcopy(rl_train_cfg), str(log_dir), device=gs.device)
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
            env_overrides=env_overrides,
        )
    elif args.stage == "bc":
        run_bc_eval_and_log_video(
            task=args.task,
            action_mode=args.action_mode,
            log_dir=log_dir,
            bc_train_cfg=bc_train_cfg,
            use_wandb=use_wandb,
            env_overrides=env_overrides,
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

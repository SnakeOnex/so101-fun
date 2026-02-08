"""
SO-101 Evaluation
=================
Evaluate trained RL or BC policies for any SO-101 task.
Loads configs saved during training to reconstruct the correct environment.

Usage:
  # Evaluate RL policy
  python eval.py --log_dir=logs/so101_reach_joint_delta_rl

  # Evaluate with video recording
  python eval.py --log_dir=logs/so101_reach_joint_delta_rl --record

  # Upload video to wandb
  python eval.py --log_dir=logs/so101_grasp_joint_delta_rl --record --wandb
"""

import argparse
import pickle
import re
from importlib import metadata
from pathlib import Path

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
import genesis as gs

from envs import TASK_REGISTRY, make_env
from so101_behavior_cloning import BehaviorCloning


def load_rl_policy(env, train_cfg, log_dir):
    """Load RL policy from checkpoint."""
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    *_, last_ckpt = sorted(
        checkpoint_files,
        key=lambda f: int(re.search(r"model_(\d+)", f.name).group(1)),
    )
    runner.load(last_ckpt)
    print(f"Loaded RL checkpoint: {last_ckpt.name}")
    return runner.get_inference_policy(device=gs.device)


def load_bc_policy(env, bc_cfg, log_dir):
    """Load BC policy from checkpoint."""
    bc_runner = BehaviorCloning(env, bc_cfg, None, device=gs.device)
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    *_, last_ckpt = sorted(
        checkpoint_files,
        key=lambda f: int(re.search(r"checkpoint_(\d+)", f.name).group(1)),
    )
    bc_runner.load(last_ckpt)
    print(f"Loaded BC checkpoint: {last_ckpt.name}")
    return bc_runner._policy


def main():
    parser = argparse.ArgumentParser(description="SO-101 Evaluation")
    parser.add_argument(
        "--log_dir", type=str, required=True,
        help="Path to the training log directory (e.g. logs/so101_reach_joint_delta_rl)",
    )
    parser.add_argument("--record", action="store_true", help="Record evaluation video")
    parser.add_argument("--wandb", action="store_true", help="Upload video to wandb")
    parser.add_argument("--num_envs", type=int, default=9, help="Number of eval envs (default 9 for 3x3 grid)")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    gs.init()

    log_dir = Path(args.log_dir)
    print(f"Loading from: {log_dir}")

    # Load saved configs
    cfgs = pickle.load(open(log_dir / "cfgs.pkl", "rb"))

    # Handle both old format (tuple) and new format (dict)
    if isinstance(cfgs, dict):
        env_cfg = cfgs["env_cfg"]
        reward_scales = cfgs["reward_scales"]
        robot_cfg = cfgs["robot_cfg"]
        rl_train_cfg = cfgs["rl_train_cfg"]
        bc_train_cfg = cfgs["bc_train_cfg"]
        task = cfgs["task"]
        action_mode = cfgs["action_mode"]
    else:
        # Old format: (env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg)
        env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg = cfgs
        task = "grasp"  # old format was always grasp
        action_mode = robot_cfg.get("action_mode", "joint_delta")

    # Determine stage from log dir name
    stage = "bc" if log_dir.name.endswith("_bc") else "rl"

    # Eval overrides
    env_cfg["num_envs"] = args.num_envs
    env_cfg["visualize_camera"] = args.record

    # BC stage needs stereo vision cameras for inference
    if stage == "bc":
        env_cfg["use_vision_cameras"] = True

    # Create env using the task registry
    env_cls = TASK_REGISTRY[task]
    env = env_cls(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=False,
    )

    # Load policy
    if stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()

    # Run eval (cap at 5s for concise videos)
    obs, _ = env.reset()
    eval_duration_s = 5.0
    max_steps = min(
        env.max_episode_length,
        int(eval_duration_s / env.ctrl_dt),
    )
    total_reward = torch.zeros(args.num_envs, device=gs.device)

    video_dir = Path("eval_videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{task}_{action_mode}_{stage}_eval.mp4"

    with torch.no_grad():
        if args.record:
            env.vis_cam.start_recording()

        for step in range(max_steps):
            if stage == "rl":
                actions = policy(obs)
            else:
                rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()
                actions = policy(rgb_obs, ee_pose)

            if args.record:
                env.vis_cam.render()

            obs, rewards, dones, infos = env.step(actions)
            total_reward += rewards

        if args.record:
            env.vis_cam.stop_recording(
                save_to_filename=str(video_path),
                fps=30,
            )
            print(f"Video saved to {video_path}")

    mean_reward = total_reward.mean().item()
    print(f"Eval mean total reward: {mean_reward:.4f}")

    # Upload to wandb
    if args.wandb and args.record:
        try:
            import wandb

            wandb.init(project="so101-rl", job_type="eval")
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/video": wandb.Video(str(video_path), format="mp4"),
                "eval/task": task,
                "eval/action_mode": action_mode,
                "eval/stage": stage,
            })
            wandb.finish()
            print("Results uploaded to wandb.")
        except ImportError:
            print("wandb not installed, skipping upload.")


if __name__ == "__main__":
    main()

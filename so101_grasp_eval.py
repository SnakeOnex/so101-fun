"""
SO-101 Grasp Evaluation
=======================
Evaluate trained RL or BC policies for the SO-101 arm.

Usage:
  # Evaluate RL policy (joint_delta mode)
  python so101_grasp_eval.py --stage=rl --action_mode=joint_delta

  # Evaluate with video recording
  python so101_grasp_eval.py --stage=rl --action_mode=joint_delta --record
"""

import argparse
import re
import pickle
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

from so101_grasp_env import GraspEnv
from so101_behavior_cloning import BehaviorCloning


def load_rl_policy(env, train_cfg, log_dir):
    """Load reinforcement learning policy."""
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    try:
        *_, last_ckpt = sorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    runner.load(last_ckpt)
    print(f"Loaded RL checkpoint from {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


def load_bc_policy(env, bc_cfg, log_dir):
    """Load behavior cloning policy."""
    bc_runner = BehaviorCloning(env, bc_cfg, None, device=gs.device)

    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    try:
        *_, last_ckpt = sorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    print(f"Loaded BC checkpoint from {last_ckpt}")
    bc_runner.load(last_ckpt)

    return bc_runner._policy


def main():
    parser = argparse.ArgumentParser(description="SO-101 Grasp Evaluation")
    parser.add_argument("-e", "--exp_name", type=str, default="so101_grasp")
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "bc"],
    )
    parser.add_argument(
        "--action_mode",
        type=str,
        default="joint_delta",
        choices=["joint_delta", "ee_delta_pos", "ee_delta"],
        help="Must match the action_mode used during training",
    )
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    gs.init()

    # Build log dir with same naming as training
    log_dir = Path("logs") / f"{args.exp_name}_{args.action_mode}_{args.stage}"
    print(f"Loading from: {log_dir}")

    # Load configurations saved during training
    env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(
        open(log_dir / "cfgs.pkl", "rb")
    )

    # Eval overrides
    env_cfg["max_visualize_FPS"] = 60
    env_cfg["box_collision"] = True
    env_cfg["box_fixed"] = False
    env_cfg["num_envs"] = 10
    env_cfg["visualize_camera"] = args.record

    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=False,
    )

    # Load policy
    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()

    obs, _ = env.reset()
    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    # Video filenames include action_mode for clarity, saved into eval_videos/
    video_dir = Path("eval_videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_prefix = video_dir / f"so101_{args.action_mode}_{args.stage}"

    with torch.no_grad():
        if args.record:
            print("Recording video...")
            env.vis_cam.start_recording()
            env.left_cam.start_recording()
            env.right_cam.start_recording()
        for step in range(max_sim_step):
            if args.stage == "rl":
                actions = policy(obs)
            else:
                rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()
                actions = policy(rgb_obs, ee_pose)

            if args.record:
                env.vis_cam.render()

            obs, rews, dones, infos = env.step(actions)
        env.grasp_and_lift_demo()
        if args.record:
            print("Stopping video recording...")
            env.vis_cam.stop_recording(
                save_to_filename=str(video_prefix) + "_eval.mp4",
                fps=env_cfg["max_visualize_FPS"],
            )
            env.left_cam.stop_recording(
                save_to_filename=str(video_prefix) + "_left.mp4",
                fps=env_cfg["max_visualize_FPS"],
            )
            env.right_cam.stop_recording(
                save_to_filename=str(video_prefix) + "_right.mp4",
                fps=env_cfg["max_visualize_FPS"],
            )
            print(f"Videos saved to {video_dir}/")


if __name__ == "__main__":
    main()

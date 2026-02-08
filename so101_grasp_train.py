"""
SO-101 Grasp Training
=====================
Two-stage training pipeline adapted for the SO-101 5-DOF arm:
  Stage 1 (RL):  Privileged PPO with joint-state observations
  Stage 2 (BC):  Vision-based behavior cloning with stereo cameras

Usage:
  # Stage 1: Train RL with joint-space actions (recommended)
  python so101_grasp_train.py --stage=rl --action_mode=joint_delta

  # Stage 1: Train RL with position-only EE actions (comparison)
  python so101_grasp_train.py --stage=rl --action_mode=ee_delta_pos

  # Stage 2: Train BC policy (requires RL to be trained first)
  python so101_grasp_train.py --stage=bc

  # With viewer / custom envs
  python so101_grasp_train.py --stage=rl -v -B 1024 --max_iterations 500
"""

import argparse
import re
import pickle
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

from so101_grasp_env import GraspEnv


def get_train_cfg(exp_name, max_iterations):
    # Stage 1: Privileged reinforcement learning (PPO)
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

    # Stage 2: Vision-based behavior cloning
    bc_cfg_dict = {
        "num_steps_per_env": 24,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "num_mini_batches": 10,
        "max_grad_norm": 1.0,
        "policy": {
            "vision_encoder": {
                "conv_layers": [
                    {
                        "in_channels": 3,
                        "out_channels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                    {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ],
                "pooling": "adaptive_avg",
            },
            "action_head": {
                "state_obs_dim": 7,  # EE pose (pos + quat) as additional state obs
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


def get_task_cfgs(action_mode: str = "joint_delta"):
    """
    SO-101 specific task configuration.

    action_mode options:
      - "joint_delta":  5D delta joint angles (recommended for 5-DOF arm)
      - "ee_delta_pos": 3D delta EE position only (position-only IK)
      - "ee_delta":     6D delta EE pose (original, problematic for 5-DOF)
    """
    # Configure action space based on mode
    if action_mode == "joint_delta":
        num_actions = 5  # delta for each arm joint
        action_scales = [0.1, 0.1, 0.1, 0.1, 0.1]  # radians per step
    elif action_mode == "ee_delta_pos":
        num_actions = 3  # delta x, y, z only
        action_scales = [0.03, 0.03, 0.03]
    elif action_mode == "ee_delta":
        num_actions = 6  # delta pose (6D)
        action_scales = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    else:
        raise ValueError(f"Unknown action_mode: {action_mode}")

    env_cfg = {
        "num_envs": 10,
        # obs: ee_pos-obj_pos(3) + ee_quat(4) + obj_pos(3) + obj_quat(4) + arm_qpos(5) = 19
        "num_obs": 19,
        "num_actions": num_actions,
        "action_scales": action_scales,
        "episode_length_s": 4.0,
        "ctrl_dt": 0.01,
        "box_size": [0.04, 0.02, 0.03],
        "box_collision": False,
        "box_fixed": True,
        "image_resolution": (64, 64),
        "use_rasterizer": True,
        "visualize_camera": False,
    }
    reward_scales = {
        "xy_align": 1.0,  # get directly above the object (XY)
        "hover_height": 1.0,  # stay at ~12cm, hard penalty below 6cm
        "gripper_pointing_down": 0.5,  # orient gripper barrel downward
        "reach_from_above": 0.5,  # only reward closeness when above the object
        "action_penalty": 0.01,  # smooth movements
    }
    # SO-101 robot config
    robot_cfg = {
        "urdf_path": "SO101/so101_new_calib.urdf",
        "ee_link_name": "gripper_link",
        "jaw_link_name": "moving_jaw_so101_v1_link",
        "default_arm_dof": [0.0, 0.0, 0.0, 0.0, 0.0],
        "default_gripper_dof": [0.0],
        "gripper_open_dof": 1.4,
        "gripper_close_dof": 0.0,
        "action_mode": action_mode,
    }
    return env_cfg, reward_scales, robot_cfg


def load_teacher_policy(env, rl_train_cfg, exp_name, action_mode):
    """Load trained RL policy to use as teacher for BC stage."""
    log_dir = Path("logs") / f"{exp_name}_{action_mode}_rl"
    assert log_dir.exists(), f"Log directory {log_dir} does not exist. Train RL first!"
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    try:
        *_, last_ckpt = sorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    assert last_ckpt is not None, f"No checkpoint found in {log_dir}"
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
    runner.load(last_ckpt)
    print(f"Loaded teacher policy from checkpoint {last_ckpt} from {log_dir}")
    teacher_policy = runner.get_inference_policy(device=gs.device)
    return teacher_policy


def main():
    parser = argparse.ArgumentParser(description="SO-101 Grasp Training")
    parser.add_argument("-e", "--exp_name", type=str, default="so101_grasp")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--stage", type=str, default="rl", choices=["rl", "bc"])
    parser.add_argument(
        "--action_mode",
        type=str,
        default="joint_delta",
        choices=["joint_delta", "ee_delta_pos", "ee_delta"],
        help="Action space: joint_delta (5D joints), ee_delta_pos (3D position), ee_delta (6D pose)",
    )
    args = parser.parse_args()

    # === init ===
    gs.init(
        backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True
    )

    # === task cfgs and training algo cfgs ===
    env_cfg, reward_scales, robot_cfg = get_task_cfgs(action_mode=args.action_mode)
    rl_train_cfg, bc_train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # === log dir ===
    # Include action_mode in dir name so runs don't overwrite each other
    if args.stage == "rl":
        log_dir = Path("logs") / f"{args.exp_name}_{args.action_mode}_{args.stage}"
    else:
        log_dir = Path("logs") / f"{args.exp_name}_{args.action_mode}_{args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg), f)

    # === env ===
    # BC only needs a small number of envs
    env_cfg["num_envs"] = args.num_envs if args.stage == "rl" else 10
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # === runner ===
    if args.stage == "bc":
        teacher_policy = load_teacher_policy(
            env, rl_train_cfg, args.exp_name, args.action_mode
        )
        bc_train_cfg["teacher_policy"] = teacher_policy
        runner = BehaviorCloning(env, bc_train_cfg, teacher_policy, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=log_dir)
    else:
        runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
        runner.learn(
            num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
        )


if __name__ == "__main__":
    main()

# SO-101 Sim-to-Real Grasping

Sim-to-real reinforcement learning for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) 5-DOF robotic arm using the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics simulator.

The project trains grasping policies via a two-stage pipeline:

1. **Stage 1 — RL (PPO):** Train a state-based policy with privileged observations using [rsl-rl](https://github.com/leggedrobotics/rsl_rl).
2. **Stage 2 — BC (DAgger):** Distill the RL policy into a vision-based student using stereo camera images.

## Project Structure

```
so101-fun/
├── train.py                     # Unified training (RL + BC, all tasks)
├── eval.py                      # Standalone evaluation
├── so101_behavior_cloning.py    # BC framework (DAgger trainer + vision policy)
│
├── envs/                        # Modular environment package
│   ├── __init__.py              # Task registry + make_env() factory
│   ├── base_env.py              # Base env + SO101Manipulator robot wrapper
│   ├── reach_env.py             # Reach task (simplest)
│   ├── hover_env.py             # Hover task (intermediate)
│   └── grasp_env.py             # Grasp task (full complexity)
│
├── SO101/                       # Robot assets
│   ├── so101_new_calib.urdf     # Calibrated URDF
│   └── assets/                  # STL meshes
│
├── sweep.yaml                   # W&B hyperparameter sweep config
├── requirements.txt
└── setup.sh                     # System-level setup (Vulkan, Mesa, etc.)
```

## Setup

Requires a GPU with Vulkan support. Tested on an NVIDIA 4090.

```bash
# System dependencies (Ubuntu)
bash setup.sh

# Or manually:
uv venv -p 3.13 --seed --clear
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Tasks

Three tasks of increasing difficulty, all sharing the same base environment and robot:

| Task | Obs Dim | Description |
|------|---------|-------------|
| `reach` | 8 | Move EE to a target position (fixed or random) |
| `hover` | 19 | Position EE above a randomly placed object |
| `grasp` | 19 | Full approach-from-above with orientation control |

## Training

### Stage 1: RL (PPO)

Trains a state-based policy using privileged observations (joint positions, target/object positions directly).

```bash
# Reach task with random targets (start here)
python train.py --task=reach --reach_target=random --max_iterations=300

# Hover task
python train.py --task=hover --max_iterations=500

# Grasp task
python train.py --task=grasp --max_iterations=500

# Disable wandb (tensorboard only)
python train.py --task=reach --no_wandb
```

Key flags:
- `--task` — `reach`, `hover`, or `grasp`
- `--action_mode` — `joint_delta` (default, 5D), `ee_delta_pos` (3D), `ee_delta` (6D)
- `--num_envs` — parallel envs (default 2048)
- `--reach_target` — `fixed` or `random` (reach task only)
- `--reward_overrides` — JSON string for reward scale tuning

After training, a 9-env evaluation video is automatically recorded and uploaded to W&B.

Checkpoints are saved to `logs/<exp>_<action_mode>_rl/`.

### Stage 2: BC (DAgger)

Distills the trained RL policy into a vision-based student that uses stereo camera images. Requires a trained RL checkpoint from Stage 1.

```bash
python train.py --task=reach --reach_target=random --stage=bc --max_iterations=300
```

This will:
1. Load the RL teacher from `logs/so101_reach_joint_delta_rl/`
2. Train a CNN student policy via DAgger on stereo RGB + EE pose
3. Record post-training eval with overhead video + left/right camera grid videos
4. Upload everything to W&B (project: `so101-bc`)

BC uses 10 envs with the BatchRenderer for stereo camera rendering.

### How DAgger Works

During BC data collection, both the teacher and student propose actions each step:

- The **teacher's action is always stored as the training label**
- The **student gets to act** when its action is close to the teacher's (`||student - teacher|| < 1.0`)
- Otherwise the **teacher takes over** to prevent catastrophic drift

The `student_control_fraction` metric tracks what percentage of steps the student is actually controlling. Early in training this is low (teacher drives); as the student improves, it takes over.

## Evaluation

```bash
# Evaluate a trained RL policy
python eval.py --log_dir=logs/so101_reach_joint_delta_rl

# Evaluate a BC policy
python eval.py --log_dir=logs/so101_reach_joint_delta_bc --stage=bc
```

## Architecture

### Robot

The SO-101 is a 5-DOF arm with a single revolute jaw gripper (6 actuated joints total). The URDF is loaded via Genesis with `fixed=True` (bolted to table).

**Fingertip position:** Genesis merges fixed-joint links, so the `gripper_frame_link` (fingertip) doesn't exist at runtime. The EE position is computed manually by rotating a local offset (`[-0.008, -0.0002, -0.098]`) from `gripper_link` by its orientation quaternion.

### Environment Hierarchy

```
SO101BaseEnv (base_env.py)
├── Scene setup, robot loading, cameras, step/reset loop, reward aggregation
├── SO101Manipulator — robot wrapper (URDF, PD control, action modes, IK)
│
├── ReachEnv (reach_env.py)  — 8D obs, reach_target + overshoot_penalty rewards
├── HoverEnv (hover_env.py)  — 19D obs, xy_align + hover_height rewards
└── GraspEnv (grasp_env.py)  — 19D obs, 5 reward terms
```

### Camera System

- **Stereo cameras** (left + right) — BatchRenderer (Madrona), used for BC training. Enabled with `use_vision_cameras=True`. Returns `[B, 6, H, W]` tensor (3 RGB channels per camera).
- **Debug camera** — Rasterizer pipeline (`debug=True`), used for eval video recording. Independent of BatchRenderer. Shows a 3x3 grid of 9 envs from overhead.

During RL training, no cameras are active (pure state-based, fast). BC training enables stereo cameras. Eval enables both.

### BC Vision Policy

```
Stereo RGB [B, 6, H, W]
    ├── Left  [B, 3, H, W] ──┐
    └── Right [B, 3, H, W] ──┤
                              ├── Shared CNN Encoder
                              │   (3 conv layers + BN + ReLU + AdaptiveAvgPool)
                              ├── Feature Fusion (Linear + ReLU + Dropout)
                              │
EE Pose [B, 7] ──────────────┤
                              └── MLP Action Head → [B, 5] joint deltas
```

### PPO Config

| Parameter | Value |
|-----------|-------|
| Actor/Critic | `[256, 256, 128]` |
| Learning rate | 3e-4 (adaptive KL schedule) |
| Clip param | 0.2 |
| Gamma / Lambda | 0.99 / 0.95 |
| Steps per env | 24 |
| Mini-batches | 4 |

### Reach Workspace (Random Mode)

| Axis | Range (m) | Notes |
|------|-----------|-------|
| X | 0.10 — 0.35 | Forward from robot base |
| Y | -0.20 — 0.20 | Lateral |
| Z | 0.05 — 0.25 | Vertical |

## W&B Integration

RL training logs to project `so101-rl`, BC to `so101-bc`. Metrics logged:

**RL:** standard rsl-rl metrics (reward, loss, KL, etc.) + `eval/video` + `eval/mean_reward`

**BC:** `bc/action_loss`, `bc/lr`, `bc/student_control_fraction`, `bc/reward_mean`, `bc/fps`, `bc/buffer_size` + eval videos (`bc_eval/overhead_video`, `bc_eval/left_cam_video`, `bc_eval/right_cam_video`)

## Hyperparameter Sweeps

```bash
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

The sweep config searches over reward scales for the grasp task.

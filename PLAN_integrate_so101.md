# SO101 Genesis integration

## Context
I have an SO101 arm at home and recently found genesis and became interested in the project. I would like to make my hand do some stuff IRL eventually, but first want to integrate it into the genesis sim.

The genesis sim in the basic examples has a robotic arm manipulation task, where it uses a Franka Panda robotic arm, which is a 7-dof arm. Seems quite a bit.

It works using a two stage training, first RL on a privileged stage setup, where the model gets robot state in the form of joint angles. It collects the successful episodes including camera information, which are used in the second stage for BC of X=Camera, Y=action.

## Links / Files
The official example overview tutorial page:
[Official example page:](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/manipulation.html)

I have put the URDF and STL files in the `SO101/` folder.
The Genesis repo and the examples is inside of `Genesis/examples/manipulation/`

## General Plan
The goal is to integrate this setup with the SO101 arm. There might be quite some challenges as the robot is different, Inverse Kinematics might not work as well due to it being 5DOF / robot being different etc.

I also want to learn, so goal is to also document & understand how this pipeline works, what are the different components and how they work.

## Step 0: Diagnostic Results (completed)

Ran `so101_diagnostic.py` to verify the robot loads correctly in Genesis and inspect all properties. Key findings:

### Robot Structure
- **6 DOFs total**: 5 arm joints + 1 gripper joint (sequential indices `[0..5]`)
- **Arm joints**: `shoulder_pan` (0), `shoulder_lift` (1), `elbow_flex` (2), `wrist_flex` (3), `wrist_roll` (4)
- **Gripper joint**: `gripper` (5) — single revolute jaw, range `[-10°, +100°]`
- **7 links**: `base_link`, `shoulder_link`, `upper_arm_link`, `lower_arm_link`, `wrist_link`, `gripper_link`, `moving_jaw_so101_v1_link`

### Important Discoveries
1. **Must use `fixed=True`** when loading the URDF, otherwise Genesis adds a FREE root joint (6 extra DOFs) and the robot floats.
2. **`gripper_frame_link` doesn't exist in Genesis** — Genesis merges fixed joints into their parent link. So `gripper_link` is our EE link.
3. **Genesis IK works** for small deltas and nearby targets (position error < 0.5mm on trivial case).
4. **DLS IK works well** — a 2cm x-delta moved the EE exactly +2cm in x with only 3mm parasitic z-drift.
5. **5-DOF orientation limitation confirmed**: When asking IK to reach a new position with a fully specified 6D orientation, position error was ~5cm. The solver over-prioritizes orientation it can't fully achieve. This is expected — the RL policy will learn to only request feasible movements, and DLS with small deltas handles this naturally.

### Robot Config for Integration
```python
robot_cfg = {
    "ee_link_name": "gripper_link",
    "gripper_link_names": ["moving_jaw_so101_v1_link"],
    "default_arm_dof": [0.0, 0.0, 0.0, 0.0, 0.0],
    "default_gripper_dof": [0.0],
    "ik_method": "dls_ik",
    "urdf_path": "SO101/so101_new_calib.urdf",
}
```

### Panda vs SO101 Comparison
| Aspect | Franka Panda | SO101 |
|---|---|---|
| Arm DOFs | 7 | 5 |
| Gripper | 2 parallel fingers | 1 revolute jaw |
| Total DOFs | 9 | 6 |
| Format | MJCF (.xml) | URDF (.urdf, needs `fixed=True`) |
| EE link | `hand` | `gripper_link` |
| EE controllability | Full 6D | 5D (no independent yaw) |
| Servo torques | Industrial (87 Nm) | Hobby STS3215 (~10 Nm) |
| Jacobian shape | [6, 9] | [6, 6] |
| Workspace | Large industrial | Small desktop (~30cm reach) |
| PD gains (kp) | [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100] | ~[300, 300, 300, 300, 300, 50] |

### EE position at home (all joints zero)
- `gripper_link`: pos=`[0.293, -0.0002, 0.234]` — about 29cm forward, 23cm up from base.

## Step 1: RL Training Pipeline (in progress)

Adapted the Genesis manipulation example pipeline for the SO-101. Created 4 files:

### Files Created
| File | Based On | Description |
|---|---|---|
| `so101_grasp_env.py` | `grasp_env.py` | `GraspEnv` + `SO101Manipulator` class adapted for SO-101 |
| `so101_grasp_train.py` | `grasp_train.py` | Training script with SO-101 configs |
| `so101_grasp_eval.py` | `grasp_eval.py` | Evaluation script |
| `so101_behavior_cloning.py` | `behavior_cloning.py` | BC policy (robot-agnostic, no changes needed) |

### Key Adaptations (Panda → SO-101)

#### SO101Manipulator class (`so101_grasp_env.py`)
- **URDF loading** with `fixed=True` instead of MJCF
- **5 arm DOFs + 1 gripper DOF** (vs 7+2)
- **Revolute jaw gripper**: open=1.4 rad (~80°), close=0.0 rad (vs prismatic 0.04m)
- **PD gains**: kp=[300,300,300,300,300,50], kv=[30,30,30,30,30,5]
- **Force limits**: ±10 Nm arm, ±5 Nm gripper (STS3215 servos)
- **No center_finger_pose**: uses `ee_pose` (gripper_link) directly instead of averaging two fingers

#### GraspEnv class (`so101_grasp_env.py`)
- **Object spawn range**: x=[0.15, 0.30], y=[-0.10, 0.10] (within 30cm reach)
- **Smaller box**: [4cm, 2cm, 3cm] (fits SO-101's compact gripper)
- **Camera positions**: closer to workspace (0.5m away vs 1.25m)
- **Keypoint scale**: 0.3 (vs 0.5) for smaller workspace
- **EE tip offset**: -3cm z (vs -6cm for Panda's longer fingers)
- **Observations**: ee_pos - obj_pos (3) + ee_quat (4) + obj_pos (3) + obj_quat (4) = 14D

#### Training configs (`so101_grasp_train.py`)
- **Action scales**: 0.03 (vs 0.05) — smaller deltas for smaller workspace
- **Episode length**: 4.0s (vs 3.0s) — more time for slower servos
- **Default experiment name**: `so101_grasp`

### How to Run

```bash
# Stage 1: Train RL policy (privileged state observations)
python so101_grasp_train.py --stage=rl

# Stage 2: Train BC policy (stereo vision, requires RL first)
python so101_grasp_train.py --stage=bc

# Evaluate
python so101_grasp_eval.py --stage=rl
python so101_grasp_eval.py --stage=bc --record
```

### Known Concerns / Things to Watch
1. **5-DOF orientation limitation**: The DLS IK naturally handles this for small deltas, but the RL policy needs to learn not to request infeasible orientations. If training stalls, may need to reduce action_scales for orientation dims or mask them.
2. **Gripper interaction**: Stage 1 keeps gripper open (approach task only). Grasping with the revolute jaw will need tuning — the single-jaw design grips differently from parallel fingers.
3. **PD gains may need tuning**: The diagnostic values (kp=300, kv=30) are a starting point. If the robot oscillates or is too sluggish during RL, these should be adjusted.
4. **Object spawn range**: Currently conservative. If the policy converges quickly, can expand the range to make it harder.

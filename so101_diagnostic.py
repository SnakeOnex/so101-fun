"""
SO101 Diagnostic Script
=======================
Loads the SO101 arm in Genesis and inspects:
  1. All links (names, indices)
  2. All joints (names, types, DOF indices, limits)
  3. Total DOFs
  4. EE link candidate poses
  5. Per-joint movement test (moves each joint one at a time and records frames)
  6. Gripper open/close test
  7. IK feasibility test (can we reach a target pose?)

Outputs:
  - Console report of all robot properties
  - so101_diagnostic.mp4  — video of each joint being exercised
"""

import genesis as gs
import numpy as np
import torch
import math

# ─── config ───────────────────────────────────────────────────────────────────
VIDEO_OUT = "so101_diagnostic.mp4"
FPS = 30
SETTLE_STEPS = 60          # steps to let the robot settle after a joint move
STEPS_PER_JOINT = 120      # steps per joint sweep (one direction)
JOINT_SWEEP_FRAC = 0.6     # fraction of joint range to sweep (avoid limits)

# Camera
CAM_RES = (640, 480)
CAM_POS = (0.5, -0.4, 0.4)
CAM_LOOKAT = (0.0, 0.0, 0.15)
CAM_FOV = 50

# ─── helpers ──────────────────────────────────────────────────────────────────

def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_row(label: str, value):
    print(f"  {label:<30s}  {value}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=FPS,
            camera_pos=CAM_POS,
            camera_lookat=CAM_LOOKAT,
            camera_fov=CAM_FOV,
        ),
        show_viewer=False,
    )

    # Ground plane
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    # SO101 arm
    so101 = scene.add_entity(
        gs.morphs.URDF(
            file="SO101/so101_new_calib.urdf",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            fixed=True,  # Pin the base to the ground — without this Genesis adds a FREE root_joint (6 DOF)
        ),
    )

    # Camera for recording
    cam = scene.add_camera(
        res=CAM_RES,
        pos=CAM_POS,
        lookat=CAM_LOOKAT,
        fov=CAM_FOV,
        GUI=False,
    )

    scene.build()

    # ── 1. Inspect links ──────────────────────────────────────────────────────
    print_separator("LINKS")
    links = so101.links
    for i, link in enumerate(links):
        print_row(f"[{i}] {link.name}", f"idx={link.idx}")

    # ── 2. Inspect joints ─────────────────────────────────────────────────────
    print_separator("JOINTS")
    joints = so101.joints
    joint_info = []
    for j in joints:
        info = {
            "name": j.name,
            "type": j.type,
            "dofs_idx_local": j.dofs_idx_local,
        }
        joint_info.append(info)
        type_str = str(j.type)
        dof_str = f"dofs_idx_local={j.dofs_idx_local}"
        print_row(f"{j.name}", f"type={type_str}  {dof_str}")

    # ── 3. Total DOFs ─────────────────────────────────────────────────────────
    print_separator("DOFs SUMMARY")
    n_dofs = so101.n_dofs
    print_row("Total DOFs", n_dofs)

    # Identify actuated joints (those that have DOFs, i.e., not fixed)
    # Also skip any FREE root_joint that Genesis might add (shouldn't appear now with fixed=True)
    actuated_joints = [j for j in joints if len(j.dofs_idx_local) > 0 and "root" not in j.name.lower()]
    print_row("Actuated joints", len(actuated_joints))
    for j in actuated_joints:
        print_row(f"  {j.name}", f"dof_idx={j.dofs_idx_local}")

    # Figure out arm vs gripper DOFs
    arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    gripper_joint_names = ["gripper"]

    arm_joints = [j for j in actuated_joints if j.name in arm_joint_names]
    gripper_joints = [j for j in actuated_joints if j.name in gripper_joint_names]
    other_joints = [j for j in actuated_joints if j.name not in arm_joint_names and j.name not in gripper_joint_names]

    print()
    print_row("Arm joints found", f"{len(arm_joints)}/{len(arm_joint_names)}")
    print_row("Gripper joints found", f"{len(gripper_joints)}/{len(gripper_joint_names)}")
    if other_joints:
        print_row("Unexpected joints", [j.name for j in other_joints])

    arm_dof_indices = []
    for j in arm_joints:
        arm_dof_indices.extend(j.dofs_idx_local)
    gripper_dof_indices = []
    for j in gripper_joints:
        gripper_dof_indices.extend(j.dofs_idx_local)

    print_row("Arm DOF indices", arm_dof_indices)
    print_row("Gripper DOF indices", gripper_dof_indices)

    # ── 4. Query joint limits ─────────────────────────────────────────────────
    print_separator("JOINT LIMITS (from URDF)")
    # Known from the URDF — used for sweep ranges and gripper open/close
    urdf_limits = {
        "shoulder_pan":   (-1.91986, 1.91986),
        "shoulder_lift":  (-1.74533, 1.74533),
        "elbow_flex":     (-1.69, 1.69),
        "wrist_flex":     (-1.65806, 1.65806),
        "wrist_roll":     (-2.74385, 2.84121),
        "gripper":        (-0.174533, 1.74533),
    }
    for name, (lo, hi) in urdf_limits.items():
        print_row(f"{name}", f"[{math.degrees(lo):+7.1f}°, {math.degrees(hi):+7.1f}°]  ({lo:+.3f}, {hi:+.3f}) rad")

    # ── 5. Check EE link candidates ──────────────────────────────────────────
    print_separator("END-EFFECTOR LINK CANDIDATES")

    ee_candidates = ["gripper_link", "gripper_frame_link", "wrist_link", "moving_jaw_so101_v1_link"]
    for link_name in ee_candidates:
        try:
            link = so101.get_link(link_name)
            pos = link.get_pos().cpu().numpy().flatten()
            quat = link.get_quat().cpu().numpy().flatten()
            print_row(f"{link_name}", f"pos={np.round(pos, 4)}  quat={np.round(quat, 4)}")
        except Exception as e:
            print_row(f"{link_name}", f"NOT FOUND: {e}")

    # ── 6. Initial qpos ──────────────────────────────────────────────────────
    print_separator("INITIAL QPOS")
    qpos = so101.get_qpos().cpu().numpy().flatten()
    print_row("qpos", np.round(qpos, 4))
    print_row("qpos shape", qpos.shape)
    print_row("Expected DOFs", f"{len(arm_dof_indices)} arm + {len(gripper_dof_indices)} gripper = {len(arm_dof_indices) + len(gripper_dof_indices)} total")
    if qpos.shape[0] != n_dofs:
        print(f"  ⚠️  qpos length ({qpos.shape[0]}) != n_dofs ({n_dofs}) — possible free root joint issue!")
    else:
        print(f"  ✅ qpos length matches n_dofs = {n_dofs}")

    # ── 7. Set PD gains for control ───────────────────────────────────────────
    print_separator("SETTING PD GAINS")
    # Conservative gains for the STS3215 servos
    n = n_dofs
    kp = np.ones(n) * 300.0
    kv = np.ones(n) * 30.0
    force_lo = np.ones(n) * -10.0
    force_hi = np.ones(n) * 10.0

    # Gripper: lighter gains
    for gi in gripper_dof_indices:
        kp[gi] = 50.0
        kv[gi] = 5.0

    so101.set_dofs_kp(kp)
    so101.set_dofs_kv(kv)
    so101.set_dofs_force_range(force_lo, force_hi)
    print_row("kp", kp)
    print_row("kv", kv)
    print(f"  PD gains set successfully.")

    # ── 8. Per-joint movement test ────────────────────────────────────────────
    print_separator("PER-JOINT MOVEMENT TEST")
    print("  Recording video of each joint being exercised...")

    cam.start_recording()

    # Let it settle at home position first
    home_qpos = np.zeros(n)
    for _ in range(SETTLE_STEPS):
        so101.control_dofs_position(home_qpos)
        scene.step()
        cam.render()

    # Sweep each actuated joint one at a time
    for joint in actuated_joints:
        jname = joint.name
        dof_idx = joint.dofs_idx_local[0]  # single DOF per joint for SO101

        if jname in urdf_limits:
            lo, hi = urdf_limits[jname]
        else:
            lo, hi = -1.0, 1.0

        # Sweep range: fraction of full range, centered
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0 * JOINT_SWEEP_FRAC
        target_lo = mid - half_range
        target_hi = mid + half_range

        print(f"  Joint '{jname}' (dof {dof_idx}): sweeping {math.degrees(target_lo):+.1f}° to {math.degrees(target_hi):+.1f}°")

        # Sweep: home → target_hi → target_lo → home
        waypoints = []
        # Phase 1: home → target_hi
        for t in range(STEPS_PER_JOINT):
            frac = t / STEPS_PER_JOINT
            val = home_qpos[dof_idx] + (target_hi - home_qpos[dof_idx]) * frac
            waypoints.append(val)
        # Phase 2: target_hi → target_lo
        for t in range(STEPS_PER_JOINT):
            frac = t / STEPS_PER_JOINT
            val = target_hi + (target_lo - target_hi) * frac
            waypoints.append(val)
        # Phase 3: target_lo → home
        for t in range(STEPS_PER_JOINT):
            frac = t / STEPS_PER_JOINT
            val = target_lo + (home_qpos[dof_idx] - target_lo) * frac
            waypoints.append(val)

        for val in waypoints:
            target = home_qpos.copy()
            target[dof_idx] = val
            so101.control_dofs_position(target)
            scene.step()
            cam.render()

        # Settle back to home
        for _ in range(SETTLE_STEPS // 2):
            so101.control_dofs_position(home_qpos)
            scene.step()
            cam.render()

    # ── 9. Gripper open/close test ────────────────────────────────────────────
    print_separator("GRIPPER OPEN/CLOSE TEST")
    if gripper_dof_indices:
        gi = gripper_dof_indices[0]
        lo, hi = urdf_limits.get("gripper", (-0.17, 1.75))

        # Open
        print(f"  Opening gripper to {math.degrees(hi):.1f}°")
        for t in range(STEPS_PER_JOINT):
            target = home_qpos.copy()
            target[gi] = hi * (t / STEPS_PER_JOINT)
            so101.control_dofs_position(target)
            scene.step()
            cam.render()

        # Hold open
        for _ in range(SETTLE_STEPS):
            target = home_qpos.copy()
            target[gi] = hi
            so101.control_dofs_position(target)
            scene.step()
            cam.render()

        # Close
        print(f"  Closing gripper to {math.degrees(lo):.1f}°")
        for t in range(STEPS_PER_JOINT):
            target = home_qpos.copy()
            target[gi] = hi + (lo - hi) * (t / STEPS_PER_JOINT)
            so101.control_dofs_position(target)
            scene.step()
            cam.render()

        # Settle
        for _ in range(SETTLE_STEPS):
            so101.control_dofs_position(home_qpos)
            scene.step()
            cam.render()
    else:
        print("  No gripper DOF found, skipping.")

    # ── 10. EE pose tracking during movement ──────────────────────────────────
    print_separator("EE POSE TRACKING (moving shoulder_pan)")

    ee_link_name = "gripper_frame_link"
    try:
        ee_link = so101.get_link(ee_link_name)
    except Exception:
        ee_link_name = "gripper_link"
        ee_link = so101.get_link(ee_link_name)

    print(f"  Using EE link: {ee_link_name}")

    # Move shoulder_pan and track EE
    pan_idx = None
    for j in actuated_joints:
        if j.name == "shoulder_pan":
            pan_idx = j.dofs_idx_local[0]
            break

    if pan_idx is not None:
        lo, hi = urdf_limits["shoulder_pan"]
        print(f"  Sweeping shoulder_pan and tracking EE position...")
        ee_positions = []
        for t in range(STEPS_PER_JOINT * 2):
            frac = t / (STEPS_PER_JOINT * 2)
            angle = lo * JOINT_SWEEP_FRAC + (hi - lo) * JOINT_SWEEP_FRAC * frac
            target = home_qpos.copy()
            target[pan_idx] = angle
            so101.control_dofs_position(target)
            scene.step()
            cam.render()

            if t % 20 == 0:
                ee_pos = ee_link.get_pos().cpu().numpy().flatten()
                ee_quat = ee_link.get_quat().cpu().numpy().flatten()
                ee_positions.append((angle, ee_pos.copy(), ee_quat.copy()))
                print(f"    pan={math.degrees(angle):+7.1f}°  ee_pos={np.round(ee_pos, 4)}  ee_quat={np.round(ee_quat, 3)}")

    # ── 11. Multi-joint reaching pose ─────────────────────────────────────────
    print_separator("MULTI-JOINT REACHING POSE")
    # Set a reasonable reaching-forward pose
    reaching_pose = home_qpos.copy()
    # Find DOF indices by joint name
    dof_map = {}
    for j in actuated_joints:
        dof_map[j.name] = j.dofs_idx_local[0]

    # A pose that reaches forward and slightly down
    if "shoulder_lift" in dof_map:
        reaching_pose[dof_map["shoulder_lift"]] = -0.5   # tilt forward
    if "elbow_flex" in dof_map:
        reaching_pose[dof_map["elbow_flex"]] = 0.8       # bend elbow
    if "wrist_flex" in dof_map:
        reaching_pose[dof_map["wrist_flex"]] = -0.3      # tilt wrist

    print(f"  Target reaching pose: {np.round(reaching_pose, 3)}")

    # Interpolate to reaching pose
    for t in range(STEPS_PER_JOINT):
        frac = t / STEPS_PER_JOINT
        target = home_qpos + (reaching_pose - home_qpos) * frac
        so101.control_dofs_position(target)
        scene.step()
        cam.render()

    # Hold and report
    for _ in range(SETTLE_STEPS):
        so101.control_dofs_position(reaching_pose)
        scene.step()
        cam.render()

    ee_pos = ee_link.get_pos().cpu().numpy().flatten()
    ee_quat = ee_link.get_quat().cpu().numpy().flatten()
    actual_qpos = so101.get_qpos().cpu().numpy().flatten()
    print(f"  Actual qpos:  {np.round(actual_qpos, 4)}")
    print(f"  EE position:  {np.round(ee_pos, 4)}")
    print(f"  EE quaternion: {np.round(ee_quat, 4)}")

    # ── 12. IK test ───────────────────────────────────────────────────────────
    print_separator("INVERSE KINEMATICS TEST")

    # Try to reach the current EE position (should be trivial — we're already there)
    target_pos = ee_pos.copy()
    target_quat = ee_quat.copy()

    print(f"  Target pos:  {np.round(target_pos, 4)}")
    print(f"  Target quat: {np.round(target_quat, 4)}")

    # Use the actual arm DOF indices (not just a range — they must match the robot's DOF numbering)
    arm_dof_idx_tensor = torch.tensor(arm_dof_indices, device=gs.device)
    print(f"  Arm DOF indices for IK: {arm_dof_indices}")

    try:
        ik_qpos = so101.inverse_kinematics(
            link=ee_link,
            pos=np.array(target_pos, dtype=np.float32),
            quat=np.array(target_quat, dtype=np.float32),
            dofs_idx_local=arm_dof_idx_tensor,
        )
        print(f"  ✅ IK solution found!")
        ik_np = ik_qpos.cpu().numpy().flatten()
        print(f"  IK qpos (full): {np.round(ik_np, 4)}")
        # Show just the arm joint values
        arm_vals = [ik_np[i] for i in arm_dof_indices]
        print(f"  IK arm joints:  {np.round(arm_vals, 4)}")

        # Apply IK solution
        for t in range(STEPS_PER_JOINT):
            so101.control_dofs_position(ik_qpos)
            scene.step()
            cam.render()

        # Check achieved EE pose
        achieved_pos = ee_link.get_pos().cpu().numpy().flatten()
        achieved_quat = ee_link.get_quat().cpu().numpy().flatten()
        pos_error = np.linalg.norm(achieved_pos - target_pos)
        print(f"  Achieved pos:  {np.round(achieved_pos, 4)}")
        print(f"  Achieved quat: {np.round(achieved_quat, 4)}")
        print(f"  Position error: {pos_error:.6f} m")
        if pos_error < 0.01:
            print(f"  ✅ IK is working well (error < 1cm)")
        else:
            print(f"  ⚠️  IK error is large — may need tuning")
    except Exception as e:
        print(f"  ❌ IK FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ── 12b. IK test: reach a NEW target ──────────────────────────────────────
    print_separator("IK TEST: REACH A NEW TARGET")
    # Try to reach a point in front and slightly below current position
    new_target_pos = np.array([0.20, 0.05, 0.15], dtype=np.float32)
    # Keep same orientation
    new_target_quat = target_quat.copy().astype(np.float32)
    print(f"  New target pos:  {new_target_pos}")
    print(f"  New target quat: {np.round(new_target_quat, 4)}")

    try:
        ik_qpos2 = so101.inverse_kinematics(
            link=ee_link,
            pos=new_target_pos,
            quat=new_target_quat,
            dofs_idx_local=arm_dof_idx_tensor,
        )
        print(f"  ✅ IK solution found!")
        ik_np2 = ik_qpos2.cpu().numpy().flatten()
        arm_vals2 = [ik_np2[i] for i in arm_dof_indices]
        print(f"  IK arm joints: {np.round(arm_vals2, 4)}")

        # Smoothly move to the new target
        current_qpos = so101.get_qpos().cpu().numpy().flatten()
        for t in range(STEPS_PER_JOINT):
            frac = t / STEPS_PER_JOINT
            interp = current_qpos + (ik_np2 - current_qpos) * frac
            so101.control_dofs_position(interp)
            scene.step()
            cam.render()

        # Hold and check
        for _ in range(SETTLE_STEPS):
            so101.control_dofs_position(ik_qpos2)
            scene.step()
            cam.render()

        achieved_pos2 = ee_link.get_pos().cpu().numpy().flatten()
        pos_error2 = np.linalg.norm(achieved_pos2 - new_target_pos)
        print(f"  Achieved pos:  {np.round(achieved_pos2, 4)}")
        print(f"  Position error: {pos_error2:.6f} m")
        if pos_error2 < 0.01:
            print(f"  ✅ IK reaches new target well (error < 1cm)")
        elif pos_error2 < 0.03:
            print(f"  ⚠️  IK close but not perfect — expected for 5-DOF")
        else:
            print(f"  ❌ IK error too large — orientation constraint may be infeasible for 5-DOF")
    except Exception as e:
        print(f"  ❌ IK FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ── 13. DLS IK test (manual, like grasp_env) ──────────────────────────────
    print_separator("DLS IK TEST (manual Jacobian-based)")

    # First, record current EE position for comparison
    pre_dls_ee_pos = ee_link.get_pos().cpu().numpy().flatten()
    print(f"  EE before DLS step: {np.round(pre_dls_ee_pos, 4)}")

    try:
        # Small delta movement: move EE 2cm in x
        delta_pose = torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], device=gs.device, dtype=torch.float32)
        jacobian = so101.get_jacobian(link=ee_link)
        print(f"  Jacobian shape: {jacobian.shape}")

        # Handle both batched [B, 6, N] and unbatched [6, N] Jacobian shapes
        if jacobian.dim() == 2:
            print(f"  Jacobian is 2D (unbatched): [{jacobian.shape[0]}, {jacobian.shape[1]}]")
            J = jacobian  # [6, n_dofs]

            lambda_val = 0.01
            J_T = J.T  # [n_dofs, 6]
            lambda_matrix = (lambda_val ** 2) * torch.eye(n=J.shape[0], device=gs.device)
            # DLS: delta_q = J^T (J J^T + λ²I)^{-1} delta_x
            delta_joint_pos = J_T @ torch.inverse(J @ J_T + lambda_matrix) @ delta_pose
            print(f"  Delta joint pos for 2cm x-move: {delta_joint_pos.cpu().numpy().round(4)}")

            current_qpos = so101.get_qpos()
            new_qpos = current_qpos + delta_joint_pos
        elif jacobian.dim() == 3:
            print(f"  Jacobian is 3D (batched): [{jacobian.shape[0]}, {jacobian.shape[1]}, {jacobian.shape[2]}]")
            J = jacobian  # [B, 6, n_dofs]

            lambda_val = 0.01
            J_T = J.transpose(1, 2)
            lambda_matrix = (lambda_val ** 2) * torch.eye(n=J.shape[1], device=gs.device)
            delta_joint_pos = (
                J_T @ torch.inverse(J @ J_T + lambda_matrix) @ delta_pose.unsqueeze(0).unsqueeze(-1)
            ).squeeze(-1)
            print(f"  Delta joint pos for 2cm x-move: {delta_joint_pos.cpu().numpy().flatten().round(4)}")

            current_qpos = so101.get_qpos()
            new_qpos = current_qpos + delta_joint_pos
        else:
            raise ValueError(f"Unexpected Jacobian dimensions: {jacobian.dim()}")

        # Apply the DLS result
        for t in range(STEPS_PER_JOINT):
            so101.control_dofs_position(new_qpos)
            scene.step()
            cam.render()

        # Hold and settle
        for _ in range(SETTLE_STEPS):
            so101.control_dofs_position(new_qpos)
            scene.step()
            cam.render()

        new_ee_pos = ee_link.get_pos().cpu().numpy().flatten()
        actual_delta = new_ee_pos - pre_dls_ee_pos
        print(f"  EE after DLS IK step: {np.round(new_ee_pos, 4)}")
        print(f"  Actual delta:  x={actual_delta[0]:+.4f}  y={actual_delta[1]:+.4f}  z={actual_delta[2]:+.4f}")
        print(f"  Expected:      x=+0.0200  y=+0.0000  z=+0.0000")
        x_error = abs(actual_delta[0] - 0.02)
        if x_error < 0.005:
            print(f"  ✅ DLS IK is working well (x error < 5mm)")
        else:
            print(f"  ⚠️  DLS IK x error: {x_error:.4f} m")

    except Exception as e:
        print(f"  ❌ DLS IK FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Return to home
    for _ in range(SETTLE_STEPS):
        so101.control_dofs_position(home_qpos)
        scene.step()
        cam.render()

    # ── Save video ────────────────────────────────────────────────────────────
    print_separator("SAVING VIDEO")
    cam.stop_recording(save_to_filename=VIDEO_OUT, fps=FPS)
    print(f"  Video saved to {VIDEO_OUT}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print_separator("SUMMARY FOR INTEGRATION")
    print(f"  Total DOFs:           {n_dofs}")
    print(f"  Arm DOFs:             {len(arm_dof_indices)}  indices={arm_dof_indices}")
    print(f"  Gripper DOFs:         {len(gripper_dof_indices)}  indices={gripper_dof_indices}")
    print(f"  EE link:              {ee_link_name}")
    print(f"  Arm joint names:      {[j.name for j in arm_joints]}")
    print(f"  Gripper joint names:  {[j.name for j in gripper_joints]}")
    print()
    # The gripper joint drives moving_jaw_so101_v1_link
    gripper_link_names = ["moving_jaw_so101_v1_link"]

    print("  Suggested robot_cfg for grasp_env integration:")
    print("  robot_cfg = {")
    print(f'      "ee_link_name": "{ee_link_name}",')
    print(f'      "gripper_link_names": {gripper_link_names},')
    print(f'      "default_arm_dof": {[0.0] * len(arm_dof_indices)},')
    print(f'      "default_gripper_dof": {[0.0] * len(gripper_dof_indices)},')
    print(f'      "ik_method": "dls_ik",')
    print(f'      "urdf_path": "SO101/so101_new_calib.urdf",')
    print("  }")

    print(f"\n{'='*60}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

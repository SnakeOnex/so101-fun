"""
SO-101 Grasp Environment
========================
Adapted from Genesis examples/manipulation/grasp_env.py for the SO-101 5-DOF arm.

Supports multiple action modes (configured via robot_cfg["action_mode"]):
  - "joint_delta":  Direct delta joint-position actions (5D). RL learns kinematics.
                    Best for 5-DOF arms where IK is unreliable.
  - "ee_delta_pos": Delta EE position only (3D) via DLS IK. No orientation control.
  - "ee_delta":     Delta EE pose (6D) via DLS IK. Original Panda-style. Problematic for 5-DOF.

Key differences from Franka Panda version:
  - URDF loading with fixed=True (instead of MJCF)
  - 5 arm DOFs + 1 revolute jaw gripper (instead of 7 + 2 parallel fingers)
  - Smaller workspace (~30cm reach vs large industrial)
  - Hobby servo PD gains (STS3215 ~10Nm vs industrial 87Nm)
  - Single jaw gripper: no left/right finger averaging, use EE link directly
  - Gripper is revolute (radians) not prismatic (meters)
  - Camera positions closer to the robot's workspace
"""

import torch
import math
from typing import Literal

import genesis as gs
from genesis.utils.geom import (
    xyz_to_quat,
    transform_quat_by_quat,
    transform_by_quat,
)


class GraspEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(10, env_cfg["num_envs"])))
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(0.5, -0.4, 0.4),
                camera_lookat=(0.15, 0.0, 0.1),
                camera_fov=50,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            renderer=gs.options.renderers.BatchRenderer(
                use_rasterizer=env_cfg["use_rasterizer"],
            ),
            show_viewer=show_viewer,
        )

        # == add ground ==
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # == add robot ==
        self.robot = SO101Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        # == add object ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["box_size"],
                fixed=env_cfg["box_fixed"],
                collision=env_cfg["box_collision"],
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )
        if self.env_cfg["visualize_camera"]:
            self.vis_cam = self.scene.add_camera(
                res=(1280, 720),
                # Pulled back to see the full grid of envs
                pos=(2.5, -1.5, 3.0),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                GUI=self.env_cfg["visualize_camera"],
                debug=True,
            )

        # == add stereo cameras ==
        self.left_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(0.5, 0.15, 0.25),
            lookat=(0.15, 0.0, 0.05),
            fov=50,
            GUI=self.env_cfg["visualize_camera"],
        )
        self.right_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(0.5, -0.15, 0.25),
            lookat=(0.15, 0.0, 0.05),
            fov=50,
            GUI=self.env_cfg["visualize_camera"],
        )

        # build — use env_spacing to lay out envs in a grid for clear visualization
        # SO-101 workspace is ~0.6m wide, so 0.8m spacing avoids overlap
        self.scene.build(
            n_envs=env_cfg["num_envs"],
            env_spacing=(0.8, 0.8),
        )
        self.robot.set_pd_gains()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        # Keypoints offset kept for optional keypoints reward (not used by default)
        # self.keypoints_offset = self.get_keypoint_offsets(
        #     batch_size=self.num_envs, device=self.device, unit_length=0.3
        # )
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)

        # reset object — within SO-101's reachable workspace
        # EE at home is at x=0.29, so place objects in front of the gripper
        num_reset = len(envs_idx)
        random_x = (
            torch.rand(num_reset, device=self.device) * 0.10 + 0.22
        )  # 0.22 ~ 0.32 (in front of the gripper, not behind it)
        random_y = (
            torch.rand(num_reset, device=self.device) - 0.5
        ) * 0.16  # -0.08 ~ 0.08 (narrower to keep in comfortable reach)
        random_z = torch.ones(num_reset, device=self.device) * 0.02  # on table
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        # Upright quaternion
        q_upright = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            num_reset, 1
        )
        # Random yaw
        random_yaw = (
            torch.rand(num_reset, device=self.device) * 2 * math.pi - math.pi
        ) * 0.25
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(num_reset, device=self.device),
                torch.zeros(num_reset, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_quat = transform_quat_by_quat(q_yaw, q_upright)

        self.goal_pose[envs_idx] = torch.cat([random_pos, goal_quat], dim=-1)
        self.object.set_pos(random_pos, envs_idx=envs_idx)
        self.object.set_quat(goal_quat, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        obs, self.extras = self.get_observations()
        return obs, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.episode_length_buf += 1

        actions = self.rescale_action(actions)
        self.last_actions = actions  # store for action_penalty reward
        self.robot.apply_action(actions, open_gripper=True)
        self.scene.step()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward
        reward = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        obs, self.extras = self.get_observations()
        return obs, reward, self.reset_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None

    def is_episode_complete(self) -> torch.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = time_out_buf

        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        ee_pos = self.robot.ee_pose[:, :3]
        ee_quat = self.robot.ee_pose[:, 3:7]
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()

        # Get current arm joint positions (normalized to [-1, 1] range)
        arm_qpos = self.robot.arm_qpos  # [B, 5]

        obs_components = [
            ee_pos - obj_pos,  # EE-to-object vector (3)
            ee_quat,  # EE orientation (4)
            obj_pos,  # object position (3)
            obj_quat,  # object orientation (4)
            arm_qpos,  # current joint positions (5) -- helps policy know its state
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)  # total: 19
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor, self.extras

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scales

    def get_stereo_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        rgb_left, _, _, _ = self.left_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb_right, _, _, _ = self.right_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb_left = rgb_left.permute(0, 3, 1, 2)[:, :3]
        rgb_right = rgb_right.permute(0, 3, 1, 2)[:, :3]
        if normalize:
            rgb_left = torch.clamp(rgb_left, min=0.0, max=255.0) / 255.0
            rgb_right = torch.clamp(rgb_right, min=0.0, max=255.0) / 255.0
        return torch.cat([rgb_left, rgb_right], dim=1)

    # ------------ reward functions ----------------
    #
    # SO-101 kinematic analysis (from diagnostic):
    #   - The gripper's LOCAL X-AXIS is the "barrel" direction (where it points/grips)
    #   - At home pose, local_x = [0.40, -0.05, 0.91] (pointing forward+up)
    #   - The robot naturally approaches objects from the front, not from above
    #   - Good grasp configs have: local_x pointing toward object, EE at 8-14cm height
    #
    # Desired behavior (staged approach):
    #   1. Move so the EE is directly above the object (XY aligned)
    #   2. Stay at hover height (~12cm) while aligning
    #   3. Point the gripper barrel downward toward the object
    #   4. Only get rewarded for closeness when approaching from above
    #   5. Never bump into the box from the side
    #

    def _reward_xy_align(self) -> torch.Tensor:
        """
        Reward for XY alignment: get directly above the object.
        Only cares about horizontal distance, ignoring height.
        """
        ee_pos = self.robot.ee_pose[:, :3]
        obj_pos = self.object.get_pos()
        xy_dist = torch.norm(ee_pos[:, :2] - obj_pos[:, :2], p=2, dim=-1)
        return torch.exp(-20.0 * xy_dist)

    def _reward_hover_height(self) -> torch.Tensor:
        """
        Reward for hovering at the right height above the object.
        Target: ~12cm absolute (about 10cm above the 2cm box).
        Hard penalty for going below 6cm (risk of collision).
        """
        ee_z = self.robot.ee_pose[:, 2]

        ideal_height = 0.12
        height_error = torch.abs(ee_z - ideal_height)
        height_reward = torch.exp(-20.0 * height_error)

        # Hard penalty for going too low (bumping into box zone)
        ground_penalty = (ee_z < 0.06).float() * -2.0

        return height_reward + ground_penalty

    def _reward_gripper_pointing_down(self) -> torch.Tensor:
        """
        Reward for the gripper barrel (local x-axis) pointing downward.
        For a grasp-ready pose, the barrel should point toward -z (down at the object).

        This is different from 'gripper_pointing_at_object' -- it specifically
        rewards a downward orientation regardless of where the object is,
        encouraging a top-approach posture.

        For quaternion (w, x, y, z), local x-axis in world frame:
          x_world = [1 - 2(y^2 + z^2), 2(xy + wz), 2(xz - wy)]
        Dot with [0, 0, -1] = -(2(xz - wy)) = 2(wy - xz)
        """
        ee_quat = self.robot.ee_pose[:, 3:7]
        w, x, y, z = ee_quat[:, 0], ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3]

        # How much does the gripper barrel point downward?
        # local_x dot [0,0,-1] = -(2(xz - wy)) = 2(wy - xz)
        downward_alignment = 2.0 * (w * y - x * z)

        # Map from [-1, 1] to [0, 1]
        return ((downward_alignment + 1.0) / 2.0).clamp(min=0.0, max=1.0)

    def _reward_reach_from_above(self) -> torch.Tensor:
        """
        Reward closeness to the object ONLY when approaching from above.
        Multiplies distance reward by a factor that's high when EE is above
        the object and low when EE is at the same height or below.
        This prevents getting reward for bumping into the box from the side.
        """
        ee_pos = self.robot.ee_pose[:, :3]
        obj_pos = self.object.get_pos()

        # 3D distance reward
        dist = torch.norm(ee_pos - obj_pos, p=2, dim=-1)
        dist_reward = torch.exp(-10.0 * dist)

        # "From above" factor: 1.0 when well above, 0.0 when at same height or below
        height_above = (ee_pos[:, 2] - obj_pos[:, 2]).clamp(min=0.0)
        # Sigmoid-like: full reward when >= 5cm above, zero when at same height
        from_above_factor = (height_above / 0.05).clamp(min=0.0, max=1.0)

        return dist_reward * from_above_factor

    def _reward_action_penalty(self) -> torch.Tensor:
        """
        Penalize large actions to encourage smooth, controlled movements.
        """
        if not hasattr(self, "last_actions"):
            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        action_norm = torch.norm(self.last_actions, p=2, dim=-1)
        return -action_norm

    # ------------ end reward functions ----------------

    def _to_world_frame(
        self,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        keypoints_offset: torch.Tensor,
    ) -> torch.Tensor:
        world = torch.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(
                keypoints_offset[:, k], quaternion
            )
        return world

    @staticmethod
    def get_keypoint_offsets(
        batch_size: int, device: str, unit_length: float = 0.3
    ) -> torch.Tensor:
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],
                    [-1.0, 0, 0],
                    [1.0, 0, 0],
                    [0, -1.0, 0],
                    [0, 1.0, 0],
                    [0, 0, -1.0],
                    [0, 0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets[None].repeat((batch_size, 1, 1))

    def grasp_and_lift_demo(self) -> None:
        """Demo sequence for SO-101: approach, grasp, lift, return home."""
        total_steps = 500
        goal_pose = self.robot.ee_pose.clone()
        lift_height = 0.15
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.2
        final_pose[:, 1] = 0.0
        final_pose[:, 2] = 0.25
        reset_pose = torch.tensor(
            [0.2, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)
        for i in range(total_steps):
            if i < total_steps / 4:
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps / 2:
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 3 / 4:
                self.robot.go_to_goal(final_pose, open_gripper=False)
            else:
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            self.scene.step()


# ------------ SO-101 Robot ----------------
class SO101Manipulator:
    """
    SO-101 5-DOF arm + 1-DOF revolute jaw gripper.

    Joint layout (6 total DOFs):
      [0] shoulder_pan
      [1] shoulder_lift
      [2] elbow_flex
      [3] wrist_flex
      [4] wrist_roll
      [5] gripper  (revolute jaw, range ~ [-10deg, +100deg])

    Supports multiple action modes:
      - "joint_delta":  actions are delta joint angles for the 5 arm DOFs
      - "ee_delta_pos": actions are 3D delta EE position (no orientation)
      - "ee_delta":     actions are 6D delta EE pose (original, problematic for 5-DOF)
    """

    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Load SO-101 URDF ==
        material = gs.materials.Rigid()
        morph = gs.morphs.URDF(
            file=args["urdf_path"],
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            fixed=True,
        )
        self._robot_entity = scene.add_entity(material=material, morph=morph)

        # Gripper DOF values (radians)
        self._gripper_open_dof = args.get("gripper_open_dof", 1.4)
        self._gripper_close_dof = args.get("gripper_close_dof", 0.0)

        # Action mode
        self._action_mode: str = args.get("action_mode", "joint_delta")

        # Initialize buffers
        self._init()

    def set_pd_gains(self):
        """PD gains for SO-101's STS3215 hobby servos."""
        self._robot_entity.set_dofs_kp(
            torch.tensor([300, 300, 300, 300, 300, 50], dtype=torch.float32),
        )
        self._robot_entity.set_dofs_kv(
            torch.tensor([30, 30, 30, 30, 30, 5], dtype=torch.float32),
        )
        self._robot_entity.set_dofs_force_range(
            torch.tensor([-10, -10, -10, -10, -10, -5], dtype=torch.float32),
            torch.tensor([10, 10, 10, 10, 10, 5], dtype=torch.float32),
        )

    def _init(self):
        """Initialize DOF indices and link references."""
        self._arm_dof_dim = 5
        self._gripper_dim = 1

        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._gripper_dof_idx = torch.tensor([self._arm_dof_dim], device=self._device)

        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._jaw_link = self._robot_entity.get_link(self._args["jaw_link_name"])

        self._default_joint_angles = list(self._args["default_arm_dof"])
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += list(self._args["default_gripper_dof"])

    def reset(self, envs_idx: torch.IntTensor):
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: torch.IntTensor | None = None):
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def apply_action(self, action: torch.Tensor, open_gripper: bool) -> None:
        """
        Apply action based on the configured action mode.

        Action modes:
          - "joint_delta":  action shape [B, 5] — delta joint angles for arm DOFs
          - "ee_delta_pos": action shape [B, 3] — delta EE position via position-only DLS IK
          - "ee_delta":     action shape [B, 6] — delta EE pose (6D) via DLS IK
        """
        if self._action_mode == "joint_delta":
            q_pos = self._joint_delta(action)
        elif self._action_mode == "ee_delta_pos":
            q_pos = self._dls_ik_pos_only(action)
        elif self._action_mode == "ee_delta":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid action mode: {self._action_mode}")

        # Set gripper
        if open_gripper:
            q_pos[:, self._gripper_dof_idx] = self._gripper_open_dof
        else:
            q_pos[:, self._gripper_dof_idx] = self._gripper_close_dof

        self._robot_entity.control_dofs_position(position=q_pos)

    def _joint_delta(self, action: torch.Tensor) -> torch.Tensor:
        """
        Direct joint-space control: action is delta joint angles for the 5 arm DOFs.
        The RL policy learns the kinematics directly — no IK needed.
        """
        q_pos = self._robot_entity.get_qpos().clone()
        q_pos[:, self._arm_dof_idx] += action[:, :5]
        return q_pos

    def _dls_ik_pos_only(self, action: torch.Tensor) -> torch.Tensor:
        """
        Position-only DLS IK: only uses the translational (top 3 rows) of the Jacobian.
        This avoids the 5-DOF orientation problem entirely.
        Action: [dx, dy, dz] (3D)
        """
        delta_pos = action[:, :3]
        lambda_val = 0.05  # higher damping for stability
        jacobian_full = self._robot_entity.get_jacobian(link=self._ee_link)
        # Only use position rows (top 3), and only arm columns (first 5)
        jacobian = jacobian_full[:, :3, : self._arm_dof_dim]  # [B, 3, 5]
        jacobian_T = jacobian.transpose(1, 2)  # [B, 5, 3]
        lambda_matrix = (lambda_val**2) * torch.eye(n=3, device=self._device)
        delta_arm_pos = (
            jacobian_T
            @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
            @ delta_pos.unsqueeze(-1)
        ).squeeze(-1)  # [B, 5]
        q_pos = self._robot_entity.get_qpos().clone()
        q_pos[:, self._arm_dof_idx] += delta_arm_pos
        return q_pos

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Full 6D DLS IK (original approach — problematic for 5-DOF, kept for comparison).
        """
        delta_pose = action[:, :6]
        lambda_val = 0.01
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * torch.eye(
            n=jacobian.shape[1], device=self._device
        )
        delta_joint_pos = (
            jacobian_T
            @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
            @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        return self._robot_entity.get_qpos() + delta_joint_pos

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True):
        """Move to an absolute goal pose using Genesis IK."""
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        if open_gripper:
            q_pos[:, self._gripper_dof_idx] = self._gripper_open_dof
        else:
            q_pos[:, self._gripper_dof_idx] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        """End-effector pose (gripper_link): [pos(3), quat(4)] = 7D."""
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def jaw_pose(self) -> torch.Tensor:
        """Moving jaw pose: [pos(3), quat(4)] = 7D."""
        pos, quat = self._jaw_link.get_pos(), self._jaw_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def arm_qpos(self) -> torch.Tensor:
        """Current arm joint positions [B, 5]."""
        return self._robot_entity.get_qpos()[:, self._arm_dof_idx]

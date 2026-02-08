"""
SO-101 Base Environment
=======================
Shared infrastructure for all SO-101 tasks: scene setup, robot, cameras,
step/reset loop, and reward aggregation.

Subclasses must implement:
  - _setup_task()          : add task-specific entities (objects, markers)
  - _get_observations()    : build observation tensor
  - _reset_task(envs_idx)  : reset task-specific state (target pos, object pos)
  - _reward_*()            : one method per reward term listed in REWARD_SCALES
"""

import math

import torch
import genesis as gs


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
          - "joint_delta":  action shape [B, 5] -- delta joint angles for arm DOFs
          - "ee_delta_pos": action shape [B, 3] -- delta EE position via position-only DLS IK
          - "ee_delta":     action shape [B, 6] -- delta EE pose (6D) via DLS IK
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
        The RL policy learns the kinematics directly -- no IK needed.
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
        Full 6D DLS IK (original approach -- problematic for 5-DOF, kept for comparison).
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


# ---- Default robot config ----

DEFAULT_ROBOT_CFG = {
    "urdf_path": "SO101/so101_new_calib.urdf",
    "ee_link_name": "gripper_link",
    "jaw_link_name": "moving_jaw_so101_v1_link",
    "default_arm_dof": [0.0, 0.0, 0.0, 0.0, 0.0],
    "default_gripper_dof": [0.0],
    "gripper_open_dof": 1.4,
    "gripper_close_dof": 0.0,
    "action_mode": "joint_delta",
}


class SO101BaseEnv:
    """
    Base environment for all SO-101 tasks.

    Handles: scene creation, robot loading, camera setup, step/reset loop,
    reward aggregation, episode tracking.

    Subclasses must implement:
      - _setup_task()           : add objects/markers to the scene (called before build)
      - _get_observations()     : return (obs_tensor, extras_dict)
      - _reset_task(envs_idx)   : reset task-specific state for given envs
      - _reward_<name>()        : one method per key in self.reward_scales
    """

    # Subclasses should override these with their own defaults
    DEFAULT_ENV_CFG: dict = {
        "num_envs": 10,
        "num_obs": 8,
        "num_actions": 5,
        "action_scales": [0.1, 0.1, 0.1, 0.1, 0.1],
        "episode_length_s": 4.0,
        "ctrl_dt": 0.01,
        "image_resolution": (64, 64),
        "use_rasterizer": True,
        "visualize_camera": False,
    }
    DEFAULT_REWARD_SCALES: dict = {}

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

        # Store configs
        self.env_cfg = env_cfg
        self.reward_scales = dict(reward_cfg)  # copy so we can mutate
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        # Expose cfg for OnPolicyRunner wandb integration (self.env.cfg)
        self.cfg = env_cfg

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

        # == task-specific setup (objects, markers, etc.) ==
        self._setup_task()

        # == cameras ==
        if self.env_cfg.get("visualize_camera", False):
            self.vis_cam = self.scene.add_camera(
                res=(256, 256),
                pos=(0.5, -0.3, 0.4),
                lookat=(0.15, 0.0, 0.1),
                fov=50,
                GUI=False,
            )

        self.left_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(0.5, 0.15, 0.25),
            lookat=(0.15, 0.0, 0.05),
            fov=50,
            GUI=False,
        )
        self.right_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(0.5, -0.15, 0.25),
            lookat=(0.15, 0.0, 0.05),
            fov=50,
            GUI=False,
        )

        # == build scene ==
        self.scene.build(
            n_envs=env_cfg["num_envs"],
            env_spacing=(0.8, 0.8),
        )
        self.robot.set_pd_gains()

        # == prepare reward functions ==
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.ctrl_dt
            reward_method = getattr(self, "_reward_" + name, None)
            if reward_method is None:
                raise AttributeError(
                    f"{self.__class__.__name__} has no reward method '_reward_{name}' "
                    f"but '{name}' is in reward_scales. "
                    f"Define the method or remove it from reward_scales."
                )
            self.reward_functions[name] = reward_method
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        self._init_buffers()
        self.reset()

    # ---- Subclass hooks (must override) ----

    def _setup_task(self) -> None:
        """Add task-specific entities to the scene (called before scene.build)."""
        raise NotImplementedError

    def _get_observations(self) -> tuple[torch.Tensor, dict]:
        """Build and return (obs_tensor, extras_dict)."""
        raise NotImplementedError

    def _reset_task(self, envs_idx: torch.Tensor) -> None:
        """Reset task-specific state for the given environment indices."""
        raise NotImplementedError

    # ---- Shared implementation ----

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # Reset robot to home position
        self.robot.reset(envs_idx)

        # Reset task-specific state
        self._reset_task(envs_idx)

        # Log per-reward-term episode sums
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
        self.last_actions = actions
        self.robot.apply_action(actions, open_gripper=self._gripper_open_during_task())
        self.scene.step()

        # Check termination
        env_reset_idx = self._check_termination()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # Compute reward
        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        obs, self.extras = self.get_observations()
        return obs, reward, self.reset_buf, self.extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self._get_observations()

    def get_privileged_observations(self) -> None:
        return None

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scales

    def _gripper_open_during_task(self) -> bool:
        """Whether to keep gripper open during the task. Override for grasp tasks."""
        return True

    def _check_termination(self) -> torch.Tensor:
        """Check for episode termination. Override for early termination conditions."""
        time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = time_out_buf

        time_out_idx = time_out_buf.nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros(
            self.num_envs, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_stereo_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        """Render stereo RGB images from left and right cameras."""
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

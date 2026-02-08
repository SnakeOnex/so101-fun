"""
SO-101 Grasp Environment
========================
Full grasp task: approach an object from above and prepare to grasp it.
This is the most complex task, combining XY alignment, height control,
gripper orientation, and approach-from-above rewards.

Observation (19D):
  ee_pos - obj_pos (3) + ee_quat (4) + obj_pos (3) + obj_quat (4) + arm_qpos (5)

Rewards:
  xy_align:              get directly above the object (horizontal distance)
  hover_height:          maintain ~12cm height, penalty below 6cm
  gripper_pointing_down: orient gripper barrel downward
  reach_from_above:      reward closeness only when approaching from above
  action_penalty:        smooth movements
"""

import math
import torch
import genesis as gs

from genesis.utils.geom import transform_quat_by_quat

from envs.base_env import SO101BaseEnv


class GraspEnv(SO101BaseEnv):

    DEFAULT_ENV_CFG = {
        "num_envs": 10,
        "num_obs": 19,
        "num_actions": 5,
        "action_scales": [0.1, 0.1, 0.1, 0.1, 0.1],
        "episode_length_s": 4.0,
        "ctrl_dt": 0.01,
        "image_resolution": (64, 64),
        "use_rasterizer": True,
        "visualize_camera": False,
        # Object config
        "box_size": [0.04, 0.02, 0.03],
        "box_collision": False,
        "box_fixed": True,
    }

    DEFAULT_REWARD_SCALES = {
        "xy_align": 1.0,
        "hover_height": 1.0,
        "gripper_pointing_down": 0.5,
        "reach_from_above": 0.5,
        "action_penalty": 0.01,
    }

    def _setup_task(self) -> None:
        """Add the target box object to the scene."""
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=self.env_cfg["box_size"],
                fixed=self.env_cfg["box_fixed"],
                collision=self.env_cfg["box_collision"],
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)),
            ),
        )
        self.goal_pose = torch.zeros(self.num_envs, 7, device=self.device)

    def _reset_task(self, envs_idx: torch.Tensor) -> None:
        """Randomize object position and yaw within SO-101's reachable workspace."""
        num_reset = len(envs_idx)

        # Position: in front of gripper, on the table
        x = torch.rand(num_reset, device=self.device) * 0.10 + 0.22   # 0.22 ~ 0.32
        y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.16  # -0.08 ~ 0.08
        z = torch.ones(num_reset, device=self.device) * 0.02           # on table
        pos = torch.stack([x, y, z], dim=-1)

        # Random yaw rotation
        q_upright = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num_reset, -1)
        random_yaw = (torch.rand(num_reset, device=self.device) * 2 * math.pi - math.pi) * 0.25
        q_yaw = torch.stack([
            torch.cos(random_yaw / 2),
            torch.zeros(num_reset, device=self.device),
            torch.zeros(num_reset, device=self.device),
            torch.sin(random_yaw / 2),
        ], dim=-1)
        goal_quat = transform_quat_by_quat(q_yaw, q_upright)

        self.goal_pose[envs_idx] = torch.cat([pos, goal_quat], dim=-1)
        self.object.set_pos(pos, envs_idx=envs_idx)
        self.object.set_quat(goal_quat, envs_idx=envs_idx)

    def _get_observations(self) -> tuple[torch.Tensor, dict]:
        ee_pos = self.robot.ee_pose[:, :3]
        ee_quat = self.robot.ee_pose[:, 3:7]
        obj_pos = self.object.get_pos()
        obj_quat = self.object.get_quat()
        arm_qpos = self.robot.arm_qpos

        obs = torch.cat([
            ee_pos - obj_pos,  # (3)
            ee_quat,           # (4)
            obj_pos,           # (3)
            obj_quat,          # (4)
            arm_qpos,          # (5)
        ], dim=-1)  # total: 19

        extras = {"observations": {"critic": obs}}
        return obs, extras

    # ---- Rewards ----

    def _reward_xy_align(self) -> torch.Tensor:
        """Reward for XY alignment: get directly above the object."""
        ee_pos = self.robot.ee_pose[:, :3]
        obj_pos = self.object.get_pos()
        xy_dist = torch.norm(ee_pos[:, :2] - obj_pos[:, :2], p=2, dim=-1)
        return torch.exp(-20.0 * xy_dist)

    def _reward_hover_height(self) -> torch.Tensor:
        """
        Reward for hovering at ~12cm height.
        Hard penalty for going below 6cm (collision risk).
        """
        ee_z = self.robot.ee_pose[:, 2]
        ideal_height = 0.12
        height_error = torch.abs(ee_z - ideal_height)
        height_reward = torch.exp(-20.0 * height_error)

        ground_penalty = (ee_z < 0.06).float() * -2.0
        return height_reward + ground_penalty

    def _reward_gripper_pointing_down(self) -> torch.Tensor:
        """
        Reward for the gripper barrel (local x-axis) pointing downward.

        For quaternion (w, x, y, z), local x-axis in world frame:
          x_world = [1 - 2(y^2 + z^2), 2(xy + wz), 2(xz - wy)]
        Dot with [0, 0, -1] = -(2(xz - wy)) = 2(wy - xz)
        """
        ee_quat = self.robot.ee_pose[:, 3:7]
        w, x, y, z = ee_quat[:, 0], ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3]
        downward_alignment = 2.0 * (w * y - x * z)
        return ((downward_alignment + 1.0) / 2.0).clamp(min=0.0, max=1.0)

    def _reward_reach_from_above(self) -> torch.Tensor:
        """
        Reward closeness to the object ONLY when approaching from above.
        Prevents getting reward for bumping into the box from the side.
        """
        ee_pos = self.robot.ee_pose[:, :3]
        obj_pos = self.object.get_pos()

        dist = torch.norm(ee_pos - obj_pos, p=2, dim=-1)
        dist_reward = torch.exp(-10.0 * dist)

        height_above = (ee_pos[:, 2] - obj_pos[:, 2]).clamp(min=0.0)
        from_above_factor = (height_above / 0.05).clamp(min=0.0, max=1.0)

        return dist_reward * from_above_factor

    def _reward_action_penalty(self) -> torch.Tensor:
        """Penalize large actions to encourage smooth, controlled movements."""
        if not hasattr(self, "last_actions"):
            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        action_norm = torch.norm(self.last_actions, p=2, dim=-1)
        return -action_norm

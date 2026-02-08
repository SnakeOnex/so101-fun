"""
SO-101 Hover Environment
========================
Intermediate task: position the end-effector directly above a box object
at a safe hover height. Trains XY alignment and height control without
gripper interaction or orientation concerns.

Observation (19D):
  ee_pos - obj_pos (3) + ee_quat (4) + obj_pos (3) + obj_quat (4) + arm_qpos (5)

Rewards:
  xy_align:      get directly above the object (horizontal distance)
  hover_height:  maintain ~12cm height, penalty below 6cm
"""

import math
import torch
import genesis as gs

from envs.base_env import SO101BaseEnv


class HoverEnv(SO101BaseEnv):

    DEFAULT_ENV_CFG = {
        "num_envs": 10,
        "num_obs": 19,
        "num_actions": 5,
        "action_scales": [0.1, 0.1, 0.1, 0.1, 0.1],
        "episode_length_s": 4.0,
        "ctrl_dt": 0.01,
        "image_resolution": (64, 64),
        "use_vision_cameras": False,
        "visualize_camera": False,
        # Object config
        "box_size": [0.04, 0.02, 0.03],
        "box_collision": False,
        "box_fixed": True,
    }

    DEFAULT_REWARD_SCALES = {
        "xy_align": 1.0,
        "hover_height": 1.0,
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

    def _reset_task(self, envs_idx: torch.Tensor) -> None:
        """Randomize object position within SO-101's reachable workspace."""
        num_reset = len(envs_idx)

        x = torch.rand(num_reset, device=self.device) * 0.10 + 0.22   # 0.22 ~ 0.32
        y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.16  # -0.08 ~ 0.08
        z = torch.ones(num_reset, device=self.device) * 0.02           # on table

        pos = torch.stack([x, y, z], dim=-1)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num_reset, -1)

        self.object.set_pos(pos, envs_idx=envs_idx)
        self.object.set_quat(quat, envs_idx=envs_idx)

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

    def _gripper_open_during_task(self) -> bool:
        """Keep gripper closed for hover -- no grasping needed."""
        return False

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

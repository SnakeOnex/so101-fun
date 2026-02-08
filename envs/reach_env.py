"""
SO-101 Reach Environment
========================
Simplest task: move the end-effector to a target position in space.
No object interaction, no gripper control. Pure position reaching.

Supports two modes (via env_cfg["reach_target_mode"]):
  - "fixed":  Target is always at a fixed position (e.g., [0.25, 0.0, 0.15])
  - "random": Target is randomized within the workspace each episode

Observation (8D):
  ee_pos - target_pos (3) + arm_qpos (5)

Rewards:
  reach_target:      exponential decay on distance to target (fingertip)
  overshoot_penalty:  penalize gripper body going inside/past the target
"""

import torch
import genesis as gs

from envs.base_env import SO101BaseEnv


class ReachEnv(SO101BaseEnv):

    DEFAULT_ENV_CFG = {
        "num_envs": 10,
        "num_obs": 8,  # ee_pos - target (3) + arm_qpos (5)
        "num_actions": 5,
        "action_scales": [0.3, 0.3, 0.3, 0.3, 0.3],
        "episode_length_s": 8.0,
        "ctrl_dt": 0.01,
        "image_resolution": (64, 64),
        "use_vision_cameras": False,
        "visualize_camera": False,
        # Reach-specific
        "reach_target_mode": "fixed",  # "fixed" or "random"
        "reach_target_fixed": [0.25, 0.0, 0.15],
    }

    DEFAULT_REWARD_SCALES = {
        "reach_target": 1.0,
        "overshoot_penalty": 0.5,
    }

    def _setup_task(self) -> None:
        """Add a small visual marker sphere for the target (no collision)."""
        self.target_marker = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.03,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)),
            ),
        )
        # Will be set in _reset_task
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _reset_task(self, envs_idx: torch.Tensor) -> None:
        """Set target position for the given envs."""
        num_reset = len(envs_idx)
        mode = self.env_cfg.get("reach_target_mode", "fixed")

        if mode == "fixed":
            fixed = self.env_cfg.get("reach_target_fixed", [0.25, 0.0, 0.15])
            pos = torch.tensor(fixed, device=self.device).unsqueeze(0).expand(num_reset, -1)
        elif mode == "random":
            # Random target within SO-101's comfortable workspace
            # Full workspace: X[-0.24,0.38] Y[-0.34,0.34] Z[-0.07,0.43]
            # We use a generous subset, avoiding extreme joint configs
            x = torch.rand(num_reset, device=self.device) * 0.30 + 0.05   # 0.05 ~ 0.35
            y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.50  # -0.25 ~ 0.25
            z = torch.rand(num_reset, device=self.device) * 0.25 + 0.05   # 0.05 ~ 0.30
            pos = torch.stack([x, y, z], dim=-1)
        else:
            raise ValueError(f"Unknown reach_target_mode: {mode}")

        self.target_pos[envs_idx] = pos
        self.target_marker.set_pos(pos, envs_idx=envs_idx)

    def _get_observations(self) -> tuple[torch.Tensor, dict]:
        ee_pos = self.robot.ee_pose[:, :3]
        arm_qpos = self.robot.arm_qpos  # [B, 5]

        obs = torch.cat([
            ee_pos - self.target_pos,  # (3) direction to target
            arm_qpos,                   # (5) joint state
        ], dim=-1)  # total: 8

        extras = {"observations": {"critic": obs}}
        return obs, extras

    # ---- Rewards ----

    def _gripper_open_during_task(self) -> bool:
        """Keep gripper closed for reach -- no grasping needed."""
        return False

    def _reward_reach_target(self) -> torch.Tensor:
        """
        Exponential reward for reaching the target position.
        Maxes out at 1.0 when at target, decays with distance.
        """
        ee_pos = self.robot.ee_pose[:, :3]
        dist = torch.norm(ee_pos - self.target_pos, p=2, dim=-1)
        return torch.exp(-20.0 * dist)

    def _reward_overshoot_penalty(self) -> torch.Tensor:
        """
        Penalize the gripper body going past/inside the target.

        The fingertip (ee_pose) is ~9.8cm ahead of the gripper motor
        (gripper_link). If the motor gets closer to the target than the
        fingertip, it means the gripper has rammed through the target.

        penalty = -max(0, fingertip_dist - motor_dist)

        This is 0 when approaching correctly (motor far, fingertip close)
        and negative when overshooting (motor closer than fingertip).
        """
        ee_pos = self.robot.ee_pose[:, :3]       # fingertip position
        motor_pos = self.robot._ee_link.get_pos()  # gripper_link (motor)

        fingertip_dist = torch.norm(ee_pos - self.target_pos, p=2, dim=-1)
        motor_dist = torch.norm(motor_pos - self.target_pos, p=2, dim=-1)

        # Overshoot: fingertip is further from target than the motor
        overshoot = torch.clamp(fingertip_dist - motor_dist, min=0.0)
        return -overshoot

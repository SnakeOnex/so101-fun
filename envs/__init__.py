"""
SO-101 Environment Registry
============================
Task registry and factory for creating environments by name.

Usage:
  from envs import TASK_REGISTRY, make_env

  env = make_env("reach", num_envs=2048, action_mode="joint_delta")
"""

from envs.reach_env import ReachEnv
from envs.hover_env import HoverEnv
from envs.grasp_env import GraspEnv
from envs.base_env import SO101BaseEnv, SO101Manipulator, DEFAULT_ROBOT_CFG

TASK_REGISTRY: dict[str, type[SO101BaseEnv]] = {
    "reach": ReachEnv,
    "hover": HoverEnv,
    "grasp": GraspEnv,
}


def make_env(
    task: str,
    num_envs: int = 2048,
    action_mode: str = "joint_delta",
    show_viewer: bool = False,
    env_overrides: dict | None = None,
    reward_overrides: dict | None = None,
    robot_overrides: dict | None = None,
) -> SO101BaseEnv:
    """
    Create an environment by task name.

    Args:
        task: One of "reach", "hover", "grasp"
        num_envs: Number of parallel environments
        action_mode: "joint_delta", "ee_delta_pos", or "ee_delta"
        show_viewer: Whether to show the Genesis viewer
        env_overrides: Dict of env_cfg overrides (merged on top of defaults)
        reward_overrides: Dict of reward_scale overrides (merged on top of defaults)
        robot_overrides: Dict of robot_cfg overrides (merged on top of defaults)

    Returns:
        The constructed environment instance.
    """
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task!r}. Available: {list(TASK_REGISTRY.keys())}")

    env_cls = TASK_REGISTRY[task]

    # Build configs from defaults + overrides
    env_cfg = dict(env_cls.DEFAULT_ENV_CFG)
    env_cfg["num_envs"] = num_envs
    if env_overrides:
        env_cfg.update(env_overrides)

    # Action mode affects num_actions and action_scales
    if action_mode == "joint_delta":
        env_cfg["num_actions"] = 5
        env_cfg["action_scales"] = [0.1, 0.1, 0.1, 0.1, 0.1]
    elif action_mode == "ee_delta_pos":
        env_cfg["num_actions"] = 3
        env_cfg["action_scales"] = [0.03, 0.03, 0.03]
    elif action_mode == "ee_delta":
        env_cfg["num_actions"] = 6
        env_cfg["action_scales"] = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    else:
        raise ValueError(f"Unknown action_mode: {action_mode!r}")

    # For reach env, obs dim depends on action mode indirectly (always 8 for reach)
    # so we don't override num_obs based on action_mode

    reward_scales = dict(env_cls.DEFAULT_REWARD_SCALES)
    if reward_overrides:
        reward_scales.update(reward_overrides)

    robot_cfg = dict(DEFAULT_ROBOT_CFG)
    robot_cfg["action_mode"] = action_mode
    if robot_overrides:
        robot_cfg.update(robot_overrides)

    return env_cls(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=show_viewer,
    )

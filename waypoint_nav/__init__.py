"""Waypoint navigation environment — Stage 1.5 of drone curriculum."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-WaypointNav-Direct-v0",
    entry_point=f"{__name__}.waypoint_nav_env:WaypointNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.waypoint_nav_env:WaypointNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaypointNavPPORunnerCfg",
    },
)

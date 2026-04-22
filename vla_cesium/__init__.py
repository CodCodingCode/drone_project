"""Cesium-world VLA drone navigation — real-city 3D Tiles variant of vla/."""

import gymnasium as gym

from . import agents  # noqa: F401

gym.register(
    id="Isaac-VLADrone-Cesium-v0",
    entry_point=f"{__name__}.vla_cesium_env:VLACesiumDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vla_cesium_env:VLACesiumDroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VLACesiumPPORunnerCfg",
    },
)

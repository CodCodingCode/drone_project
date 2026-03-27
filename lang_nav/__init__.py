"""Language-grounded drone navigation RL task for IsaacLab."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-LangDrone-Direct-v0",
    entry_point=f"{__name__}.lang_drone_env:LangDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lang_drone_env:LangDroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LangDronePPORunnerCfg",
    },
)

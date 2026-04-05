"""Language-grounded drone navigation (SigLIP + cosine-sim variant) for IsaacLab.

This is the SigLIP-based VLA with a text<->image cosine similarity scalar added
to both the observation and the reward, as an alternative to the PaliGemma
variant in sibling package `lang_nav`. Registered under a separate gym id and
log directory so the two can coexist without collisions.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-LangDroneSigLIP-Direct-v0",
    entry_point=f"{__name__}.lang_drone_env:LangDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lang_drone_env:LangDroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LangDronePPORunnerCfg",
    },
)

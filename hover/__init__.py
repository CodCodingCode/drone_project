"""Hover pre-training environment — Stage 1 of drone curriculum."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Hover-Direct-v0",
    entry_point=f"{__name__}.hover_env:HoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_env:HoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HoverPPORunnerCfg",
    },
)

"""Pi0-based drone navigation environment.

Reuses the VLA drone environment with Pi0's PaliGemma backbone
(fine-tuned on embodied robot control) instead of vanilla PaliGemma.

Observation (multi-group dict) -- identical to VLA:
  "policy"      (N, 9)          flight state (lin_vel, ang_vel, gravity)
  "rgb"         (N, 64, 64, 3)  raw camera image
  "text_tokens" (N, 32)         tokenized text command
  "text_mask"   (N, 32)         attention mask for text

Action (4-dim): normalised thrust + 3-axis moment.
"""

import gymnasium as gym

from . import agents  # noqa: F401

gym.register(
    id="Isaac-Pi0Drone-Direct-v0",
    entry_point="vla.vla_drone_env:VLADroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "vla.vla_drone_env:VLADroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Pi0DronePPORunnerCfg",
    },
)

"""VLA (Vision-Language-Action) drone navigation environment.

Stage 4 of curriculum: PaliGemma 3B VLA with LoRA fine-tuning.
The drone navigates to objects based on natural language commands using
an end-to-end vision-language model — not frozen CLIP features.

Observation (multi-group dict):
  "policy"      (N, 9)     flight state (lin_vel, ang_vel, gravity)
  "rgb"         (N, 64, 64, 3)  raw camera image
  "text_tokens" (N, 32)    tokenized text command
  "text_mask"   (N, 32)    attention mask for text

Action (4-dim): normalised thrust + 3-axis moment.
"""

import gymnasium as gym

from . import agents  # noqa: F401

gym.register(
    id="Isaac-VLADrone-Direct-v0",
    entry_point=f"{__name__}.vla_drone_env:VLADroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vla_drone_env:VLADroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VLADronePPORunnerCfg",
    },
)

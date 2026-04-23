"""VLA Universal — offline scene scan + open-vocabulary navigation.

One drone, any Isaac Sim USD scene. The workflow:
    1. `scan.py`   — flythrough + PaliGemma detection + 3D projection → semantic_map.json
    2. `navigate.py` — load map, match user prompt, fly to target via frozen waypoint policy

See README.md for usage.
"""

import gymnasium as gym

gym.register(
    id="Isaac-VLADrone-Universal-v0",
    entry_point=f"{__name__}.universal_env:UniversalDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.universal_env:UniversalDroneEnvCfg",
    },
)

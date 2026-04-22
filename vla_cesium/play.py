"""Play back a VLA policy inside the Cesium real-world scene.

Thin shim over vla/play.py — reuses the playback/video-recording loop
verbatim and only swaps the environment via sys.modules patching.

Launch:
    cd /home/ubuntu/IsaacLab
    export CESIUM_ION_TOKEN=<your-token>
    ./isaaclab.sh -p /home/ubuntu/drone_project/vla_cesium/play.py \
        --checkpoint <path/to/model.pt> --num_steps 1500 --video
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_PROJECT = os.path.dirname(_HERE)
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

# Phase 1: import originals before shadowing
import vla.vla_drone_env                      # noqa: F401
import vla.agents.rsl_rl_ppo_cfg              # noqa: F401

import vla_cesium.vla_cesium_env as _cesium_env_mod
import vla_cesium.agents.rsl_rl_ppo_cfg as _cesium_ppo_mod
sys.modules["vla.vla_drone_env"] = _cesium_env_mod
sys.modules["vla.agents.rsl_rl_ppo_cfg"] = _cesium_ppo_mod

# Phase 2: redirect gym.make
import gymnasium as gym
import vla_cesium  # noqa: F401

_real_gym_make = gym.make


def _patched_gym_make(task_id, **kwargs):
    if task_id == "Isaac-VLADrone-Direct-v0":
        task_id = "Isaac-VLADrone-Cesium-v0"
    return _real_gym_make(task_id, **kwargs)


gym.make = _patched_gym_make

# Phase 3: AppLauncher starts when we import vla.play
import vla.play  # noqa: F401

# Phase 4: enable Cesium extension
from vla_cesium.cesium_setup import enable_cesium_extension
if not enable_cesium_extension():
    print("[vla_cesium] FATAL: could not enable cesium.omniverse.")
    vla.play.simulation_app.close()
    sys.exit(1)

# Phase 5: run playback
if __name__ == "__main__":
    vla.play.main()
    vla.play.simulation_app.close()

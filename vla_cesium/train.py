"""Train the VLA drone in a Cesium real-world scene.

Thin shim over vla/train.py — we reuse the 600-line training loop verbatim
and only swap the environment via sys.modules patching.

Launch:
    cd /home/ubuntu/IsaacLab
    export CESIUM_ION_TOKEN=<your-token>
    ./isaaclab.sh -p /home/ubuntu/drone_project/vla_cesium/train.py \
        --num_envs 128 --max_iterations 3000 --headless --enable_cameras \
        --resume_path /home/ubuntu/drone_project/logs/rsl_rl/vla_drone_direct/<run>/model_XXXX.pt

Notes
-----
* You MUST set CESIUM_ION_TOKEN (free token from cesium.com/ion/tokens).
* Checkpoints + TensorBoard logs go to logs/rsl_rl/vla_drone_cesium/ so they
  don't clobber the empty-env VLA logs under logs/rsl_rl/vla_drone_direct/.
* --resume_path's shape-mismatch-tolerant loader lets you warm-start from
  your existing vla/ checkpoint; the Cesium env produces the exact same
  observation schema so all compatible weights transfer.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_PROJECT = os.path.dirname(_HERE)
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

# --- Phase 1: import originals so we can subclass, THEN shadow them ---------
# vla_cesium.vla_cesium_env does `from vla.vla_drone_env import VLADroneEnv`,
# so the original must be importable before the patch flips sys.modules.
import vla.vla_drone_env                      # noqa: F401 — load original first
import vla.agents.rsl_rl_ppo_cfg              # noqa: F401

import vla_cesium.vla_cesium_env as _cesium_env_mod
import vla_cesium.agents.rsl_rl_ppo_cfg as _cesium_ppo_mod

# Shadow the original modules so `from vla.vla_drone_env import VLADroneEnvCfg`
# in vla/train.py resolves to our Cesium subclasses.
sys.modules["vla.vla_drone_env"] = _cesium_env_mod
sys.modules["vla.agents.rsl_rl_ppo_cfg"] = _cesium_ppo_mod

# --- Phase 2: redirect gym.make to the Cesium task id -----------------------
import gymnasium as gym
import vla_cesium  # noqa: F401 — registers Isaac-VLADrone-Cesium-v0

_real_gym_make = gym.make


def _patched_gym_make(task_id, **kwargs):
    if task_id == "Isaac-VLADrone-Direct-v0":
        task_id = "Isaac-VLADrone-Cesium-v0"
    return _real_gym_make(task_id, **kwargs)


gym.make = _patched_gym_make

# --- Phase 3: defer to vla.train, which starts AppLauncher at module load ---
# Importing vla.train triggers AppLauncher and argparse parsing at module top.
import vla.train  # noqa: F401

# --- Phase 4: enable the Cesium extension now that SimulationApp is live ----
from vla_cesium.cesium_setup import enable_cesium_extension

if not enable_cesium_extension():
    print("[vla_cesium] FATAL: could not enable cesium.omniverse. "
          "Run vla_cesium/install_cesium.sh first.")
    vla.train.simulation_app.close()
    sys.exit(1)

# --- Phase 5: run the training loop -----------------------------------------
if __name__ == "__main__":
    vla.train.main()
    vla.train.simulation_app.close()

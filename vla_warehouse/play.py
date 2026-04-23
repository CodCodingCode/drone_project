"""Play back a VLA policy in an Isaac Sim sample environment.

Thin shim over vla/play.py — swaps the environment via sys.modules patch.
AppLauncher ordering is the same as train.py's shim; see that file's
docstring for the phase layout.

Launch:
    bash ~/drone_project/vla_warehouse/run_play.sh --checkpoint <ckpt>
"""

import argparse
import os
import sys

# ----- Phase 0: args + AppLauncher first -----------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play VLA drone in Isaac Sim sample scene.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=750)
parser.add_argument("--video", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True

_app_launcher = AppLauncher(args_cli)
_simulation_app = _app_launcher.app

# ----- Phase 1: sys.modules patch ------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_PROJECT = os.path.dirname(_HERE)
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla.vla_drone_env                      # noqa: F401
import vla.agents.rsl_rl_ppo_cfg              # noqa: F401

import vla_warehouse.vla_warehouse_env as _wh_env_mod
import vla_warehouse.agents.rsl_rl_ppo_cfg as _wh_ppo_mod
sys.modules["vla.vla_drone_env"] = _wh_env_mod
sys.modules["vla.agents.rsl_rl_ppo_cfg"] = _wh_ppo_mod

# ----- Phase 2: redirect gym.make ------------------------------------------
import gymnasium as gym
import vla_warehouse  # noqa: F401

_real_gym_make = gym.make


def _patched_gym_make(task_id, **kwargs):
    if task_id == "Isaac-VLADrone-Direct-v0":
        task_id = "Isaac-VLADrone-Warehouse-v0"
    return _real_gym_make(task_id, **kwargs)


gym.make = _patched_gym_make

# ----- Phase 3: neutralize the AppLauncher inside vla.play -----------------
class _NoopAppLauncher:
    def __init__(self, *_args, **_kwargs):
        self.app = _simulation_app

    @staticmethod
    def add_app_launcher_args(_parser):
        pass


import isaaclab.app
isaaclab.app.AppLauncher = _NoopAppLauncher

# ----- Phase 4: defer to vla.play ------------------------------------------
import vla.play  # noqa: F401


if __name__ == "__main__":
    vla.play.main()
    _simulation_app.close()

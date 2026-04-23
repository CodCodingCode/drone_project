"""Train the VLA drone in an Isaac Sim sample environment (warehouse/hospital/office).

Thin shim over vla/train.py — reuses the 600-line training loop verbatim
and only swaps the environment via sys.modules patching.

Launch:
    cd /home/ubuntu/IsaacLab
    bash ~/drone_project/vla_warehouse/run_train.sh --num_envs 64 --max_iterations 3000 --resume_path <ckpt>

Phases (order matters — pxr/isaaclab only load once AppLauncher runs):
    0. Parse CLI + start AppLauncher first (loads pxr, isaaclab, Kit)
    1. sys.modules patch so `from vla.vla_drone_env import ...` resolves to
       our Warehouse subclass
    2. Redirect gym.make("Isaac-VLADrone-Direct-v0") -> Warehouse task id
    3. Neutralize vla.train's top-level AppLauncher so importing it doesn't
       try to boot a second Kit app
    4. Import vla.train and run its main()
"""

import argparse
import os
import sys

# ----- Phase 0: args + AppLauncher BEFORE any isaaclab/pxr imports ---------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train VLA drone in Isaac Sim sample scene.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume_path", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_app_launcher = AppLauncher(args_cli)
_simulation_app = _app_launcher.app

# ----- Phase 1: sys.modules patch (now that isaaclab is bootstrapped) ------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_PROJECT = os.path.dirname(_HERE)
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla.vla_drone_env                      # noqa: F401 — load original first
import vla.agents.rsl_rl_ppo_cfg              # noqa: F401

import vla_warehouse.vla_warehouse_env as _wh_env_mod
import vla_warehouse.agents.rsl_rl_ppo_cfg as _wh_ppo_mod
sys.modules["vla.vla_drone_env"] = _wh_env_mod
sys.modules["vla.agents.rsl_rl_ppo_cfg"] = _wh_ppo_mod

# ----- Phase 2: redirect gym.make to the Warehouse task id -----------------
import gymnasium as gym
import vla_warehouse  # noqa: F401 — registers Isaac-VLADrone-Warehouse-v0

_real_gym_make = gym.make


def _patched_gym_make(task_id, **kwargs):
    if task_id == "Isaac-VLADrone-Direct-v0":
        task_id = "Isaac-VLADrone-Warehouse-v0"
    return _real_gym_make(task_id, **kwargs)


gym.make = _patched_gym_make

# ----- Phase 3: neutralize the AppLauncher inside vla.train ----------------
# vla/train.py starts its own AppLauncher at module load. We already have
# one running, so swap __init__ for a no-op that returns our existing app.
# BUT we MUST keep add_app_launcher_args functional — vla.train's parser
# uses it to register --device, --headless, --enable_cameras, etc., which
# main() reads via args_cli.device.
import isaaclab.app

_OriginalAppLauncher = isaaclab.app.AppLauncher


class _NoopAppLauncher:
    def __init__(self, *_args, **_kwargs):
        self.app = _simulation_app

    @staticmethod
    def add_app_launcher_args(parser):
        _OriginalAppLauncher.add_app_launcher_args(parser)


isaaclab.app.AppLauncher = _NoopAppLauncher

# ----- Phase 4: defer to vla.train -----------------------------------------
import vla.train  # noqa: F401


if __name__ == "__main__":
    vla.train.main()
    _simulation_app.close()

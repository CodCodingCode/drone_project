"""Train the language-grounded drone navigation policy with RSL-RL PPO.

Must be launched via IsaacLab's Python wrapper so Isaac Sim loads correctly:

    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/lang_nav/train.py

Optional args:
    --num_envs      Number of parallel environments (default: 1024)
    --max_iterations  PPO iterations (default: 500)
    --seed          Random seed (default: 42)
    --device        Torch device string, e.g. "cuda:0" (default: cuda:0)
    --headless      Run without GUI (passed through to AppLauncher)

Smoke-test (no real training):
    ./isaaclab.sh -p /home/ubuntu/drone_project/lang_nav/train.py \
        --num_envs 4 --max_iterations 1 --headless
"""

import argparse
import os
import sys
import warnings

# Suppress noisy Isaac Sim deprecation/extension warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rsl_rl")
os.environ["OMNI_LOG_LEVEL"] = "ERROR"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train language drone nav with RSL-RL PPO.")
parser.add_argument("--num_envs", type=int, default=1024, help="Parallel environments.")
parser.add_argument("--max_iterations", type=int, default=3000, help="PPO iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--resume_path", type=str, default=None, help="Path to pretrained checkpoint (.pt) to init from.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim — must happen before any omni/isaaclab imports
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------
# Everything below runs after Isaac Sim is initialised
# -----------------------------------------------------------------------

import os
import sys
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Add drone_project/ to path so `import lang_nav` works
_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

# Importing lang_nav registers Isaac-LangDrone-Direct-v0 with gymnasium
import lang_nav  # noqa: F401

from lang_nav.lang_drone_env import LangDroneEnvCfg
from lang_nav.agents.rsl_rl_ppo_cfg import LangDronePPORunnerCfg


def main():
    # --- Config ---
    env_cfg = LangDroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    agent_cfg = LangDronePPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.seed = args_cli.seed

    log_dir = os.path.abspath(
        os.path.join(
            _DRONE_PROJECT,
            "logs",
            "rsl_rl",
            agent_cfg.experiment_name,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
    )
    print(f"[INFO] Logging to: {log_dir}")

    # Handle deprecated rsl-rl config fields for version compatibility
    from importlib.metadata import version as pkg_version
    handle_deprecated_rsl_rl_cfg(agent_cfg, pkg_version("rsl-rl-lib"))

    # --- Environment ---
    env = gym.make("Isaac-LangDrone-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # --- Runner ---
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Load pretrained weights (e.g. from hover pre-training)
    if args_cli.resume_path:
        print(f"[INFO] Loading pretrained checkpoint: {args_cli.resume_path}")
        ckpt = torch.load(args_cli.resume_path, map_location=agent_cfg.device, weights_only=False)
        runner.alg.actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
        runner.alg.critic.load_state_dict(ckpt["critic_state_dict"], strict=False)
        print("[INFO] Pretrained weights loaded (strict=False — missing/extra keys ignored)")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

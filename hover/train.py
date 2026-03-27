"""Train the hover (flight control) policy — Stage 1 of curriculum.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/hover/train.py

Optional args:
    --num_envs        Number of parallel environments (default: 1024)
    --max_iterations  PPO iterations (default: 300)
    --seed            Random seed (default: 42)
    --device          Torch device string (default: cuda:0)
    --headless        Run without GUI

The trained checkpoint is saved under drone_project/logs/rsl_rl/hover_pretrain/.
Use it to initialise the lang_nav policy in Stage 2.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train hover policy with RSL-RL PPO.")
parser.add_argument("--num_envs", type=int, default=1024, help="Parallel environments.")
parser.add_argument("--max_iterations", type=int, default=1500, help="PPO iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------

import os
import sys
from datetime import datetime

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import hover  # noqa: F401 — registers Isaac-Hover-Direct-v0

from hover.hover_env import HoverEnvCfg
from hover.agents.rsl_rl_ppo_cfg import HoverPPORunnerCfg


def main():
    env_cfg = HoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    agent_cfg = HoverPPORunnerCfg()
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

    from importlib.metadata import version as pkg_version
    handle_deprecated_rsl_rl_cfg(agent_cfg, pkg_version("rsl-rl-lib"))

    env = gym.make("Isaac-Hover-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Train VLA (PaliGemma 3B + LoRA) drone navigation policy.

This uses a custom training loop because the VLA actor/critic are not
standard RSL-RL MLPModel instances — they wrap PaliGemma with LoRA
and need manual construction + shared backbone linking.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/vla/train.py \
        --num_envs 256 --max_iterations 5000 --headless --enable_cameras

Optional:
    --resume_path  Path to transferred waypoint checkpoint for action head seeding
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train VLA drone navigation policy.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs (256 for A10, 512 for H100)")
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume_path", type=str, default=None, help="Path to action head checkpoint (.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------

import os
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla  # noqa: F401 — registers Isaac-VLADrone-Direct-v0

from vla.vla_drone_env import VLADroneEnvCfg
from vla.vla_policy import VLAActorModel, VLACriticModel
from vla.agents.rsl_rl_ppo_cfg import VLADronePPORunnerCfg


def main():
    env_cfg = VLADroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    agent_cfg = VLADronePPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.seed = args_cli.seed

    log_dir = os.path.abspath(os.path.join(
        _DRONE_PROJECT, "logs", "rsl_rl", agent_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    ))
    print(f"[INFO] Logging to: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = gym.make("Isaac-VLADrone-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    device = agent_cfg.device

    # ---------------------------------------------------------------
    # Construct VLA actor + critic manually (not via OnPolicyRunner)
    # ---------------------------------------------------------------
    print("[INFO] Constructing VLA actor (PaliGemma 3B + LoRA)...")
    actor = VLAActorModel(
        flight_state_dim=9,
        action_dim=4,
        hidden_dims=[256, 256],
        activation="elu",
        init_std=0.6,
        paligemma_model_name="google/paligemma-3b-pt-224",
        lora_rank=8,
        lora_alpha=16.0,
    ).to(device)

    print("[INFO] Constructing VLA critic...")
    critic = VLACriticModel(
        flight_state_dim=9,
        hidden_dims=[256, 256],
        activation="elu",
    ).to(device)

    # Link shared PaliGemma backbone (avoids double forward)
    critic._shared_paligemma = actor.paligemma
    print("[INFO] Linked shared PaliGemma backbone between actor and critic.")

    # Load action head weights from waypoint nav checkpoint
    if args_cli.resume_path:
        print(f"[INFO] Loading action head from: {args_cli.resume_path}")
        ckpt = torch.load(args_cli.resume_path, map_location=device, weights_only=False)
        # Load with strict=False — PaliGemma weights are NOT in the checkpoint
        missing_a, unexpected_a = actor.load_state_dict(ckpt.get("actor_state_dict", {}), strict=False)
        missing_c, unexpected_c = critic.load_state_dict(ckpt.get("critic_state_dict", {}), strict=False)
        print(f"  Actor: {len(missing_a)} missing, {len(unexpected_a)} unexpected keys")
        print(f"  Critic: {len(missing_c)} missing, {len(unexpected_c)} unexpected keys")

    # Print trainable parameter summary
    n_actor_train = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    n_actor_total = sum(p.numel() for p in actor.parameters())
    n_critic_train = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"[INFO] Actor: {n_actor_train:,} trainable / {n_actor_total:,} total params")
    print(f"[INFO] Critic: {n_critic_train:,} trainable params")

    # ---------------------------------------------------------------
    # Create PPO algorithm with custom actor/critic
    # ---------------------------------------------------------------
    alg_cfg = agent_cfg.algorithm.to_dict()
    alg_cfg["actor"] = actor
    alg_cfg["critic"] = critic

    # Construct PPO manually
    ppo = PPO(
        actor=actor,
        critic=critic,
        num_learning_epochs=alg_cfg["num_learning_epochs"],
        num_mini_batches=alg_cfg["num_mini_batches"],
        clip_param=alg_cfg["clip_param"],
        gamma=alg_cfg["gamma"],
        lam=alg_cfg["lam"],
        value_loss_coef=alg_cfg["value_loss_coef"],
        entropy_coef=alg_cfg["entropy_coef"],
        learning_rate=alg_cfg["learning_rate"],
        max_grad_norm=alg_cfg["max_grad_norm"],
        use_clipped_value_loss=alg_cfg["use_clipped_value_loss"],
        schedule=alg_cfg["schedule"],
        desired_kl=alg_cfg["desired_kl"],
        device=device,
    )

    # Create runner with our custom PPO
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    runner.alg = ppo  # Override with our custom PPO

    # Train
    print(f"[INFO] Starting VLA training: {agent_cfg.max_iterations} iterations, {args_cli.num_envs} envs")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

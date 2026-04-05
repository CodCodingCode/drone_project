"""Train Pi0-based drone navigation policy.

Uses Pi0's PaliGemma backbone (frozen) with a trainable MLP action head.
No LoRA -- only the MLP + action std are trained via PPO.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/pi/train.py \
        --num_envs 512 --max_iterations 5000 --headless --enable_cameras

Optional:
    --resume_path  Path to transferred waypoint checkpoint for action head seeding
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Pi0 drone navigation policy.")
parser.add_argument("--num_envs", type=int, default=512, help="Parallel envs (512 for GH200)")
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume_path", type=str, default=None, help="Path to action head checkpoint (.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------

import os
import time
from collections import deque
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.algorithms import PPO

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import pi  # noqa: F401 -- registers Isaac-Pi0Drone-Direct-v0

from vla.vla_drone_env import VLADroneEnvCfg
from pi.pi0_policy import Pi0ActorModel, Pi0CriticModel
from pi.agents.rsl_rl_ppo_cfg import Pi0DronePPORunnerCfg


class Pi0Policy(nn.Module):
    """Wrapper that makes Pi0 actor+critic look like RSL-RL's ActorCritic."""

    is_recurrent = False

    def __init__(self, actor: Pi0ActorModel, critic: Pi0CriticModel):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def _obs_dict(self, obs: TensorDict) -> dict:
        return {k: obs[k] for k in obs.keys()}

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        d = self._obs_dict(obs)
        return self.actor(d, stochastic_output=True)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        d = self._obs_dict(obs)
        return self.critic(d)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.actor.get_output_log_prob(actions)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.actor.output_distribution_params[0]

    @property
    def action_std(self) -> torch.Tensor:
        return self.actor.output_distribution_params[1]

    @property
    def entropy(self) -> torch.Tensor:
        return self.actor.output_entropy

    def update_normalization(self, obs: TensorDict) -> None:
        d = self._obs_dict(obs)
        self.actor.update_normalization(d)
        self.critic.update_normalization(d)

    def reset(self, dones=None):
        self.actor.reset()
        self.critic.reset()

    def act_inference(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        d = self._obs_dict(obs)
        return self.actor(d, stochastic_output=False)

    def get_hidden_states(self):
        return None


def main():
    env_cfg = VLADroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    agent_cfg = Pi0DronePPORunnerCfg()
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
    env = gym.make("Isaac-Pi0Drone-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    device = agent_cfg.device

    # ---------------------------------------------------------------
    # Construct Pi0 actor + critic
    # ---------------------------------------------------------------
    print("[INFO] Constructing Pi0 actor (frozen backbone + MLP action head)...")
    actor = Pi0ActorModel(
        flight_state_dim=9,
        action_dim=4,
        hidden_dims=[256, 256],
        activation="elu",
        init_std=0.6,
    ).to(device)

    print("[INFO] Constructing Pi0 critic...")
    critic = Pi0CriticModel(
        flight_state_dim=9,
        hidden_dims=[256, 256],
        activation="elu",
    ).to(device)

    critic._shared_pi0 = actor.pi0
    print("[INFO] Linked shared Pi0 backbone between actor and critic.")

    # Load action head weights
    if args_cli.resume_path:
        print(f"[INFO] Loading action head from: {args_cli.resume_path}")
        ckpt = torch.load(args_cli.resume_path, map_location=device, weights_only=False)
        missing_a, unexpected_a = actor.load_state_dict(ckpt.get("actor_state_dict", {}), strict=False)
        missing_c, unexpected_c = critic.load_state_dict(ckpt.get("critic_state_dict", {}), strict=False)
        print(f"  Actor: {len(missing_a)} missing, {len(unexpected_a)} unexpected keys")
        print(f"  Critic: {len(missing_c)} missing, {len(unexpected_c)} unexpected keys")

    n_actor_train = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    n_actor_total = sum(p.numel() for p in actor.parameters())
    n_critic_train = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"[INFO] Actor: {n_actor_train:,} trainable / {n_actor_total:,} total params")
    print(f"[INFO] Critic: {n_critic_train:,} trainable params")

    # ---------------------------------------------------------------
    # Create PPO (single optimizer for MLP + std only)
    # ---------------------------------------------------------------
    policy = Pi0Policy(actor, critic).to(device)

    alg_cfg = agent_cfg.algorithm.to_dict()
    alg_cfg.pop("class_name", None)

    ppo = PPO(policy=policy, device=device, **alg_cfg)

    # PPO's optimizer automatically picks up only requires_grad=True params
    # (MLP weights + _std_param), since Pi0 backbone is fully frozen.
    n_optim = sum(p.numel() for group in ppo.optimizer.param_groups for p in group["params"])
    print(f"[INFO] PPO optimizer: {n_optim:,} params (MLP + std only)")

    # Init rollout storage
    obs = env.get_observations()
    ppo.init_storage(
        "rl",
        env.num_envs,
        agent_cfg.num_steps_per_env,
        obs,
        [env.num_actions],
    )

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    num_steps_per_env = agent_cfg.num_steps_per_env
    max_iterations = agent_cfg.max_iterations
    save_interval = agent_cfg.save_interval

    # Tensorboard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    obs = env.get_observations().to(device)
    policy.train()

    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    tot_timesteps = 0
    tot_time = 0.0

    # Randomize initial episode lengths
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))

    print(f"[INFO] Starting Pi0 training: {max_iterations} iterations, {args_cli.num_envs} envs")

    for it in range(max_iterations):
        start = time.time()

        # -- Rollout --
        with torch.inference_mode():
            for _ in range(num_steps_per_env):
                actions = ppo.act(obs)
                # Stash Pi0 features into stored obs so PPO replay skips the 3B model
                cached_feat = policy.actor.pi0._cache_val
                if cached_feat is not None:
                    ppo.transition.observations["vla_features"] = cached_feat.detach()
                policy.actor.pi0.clear_cache()

                obs, rewards, dones, extras = env.step(actions.to(env.device))
                obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
                ppo.process_env_step(obs, rewards, dones, extras)

                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

        # -- PPO Update (no LoRA step) --
        ppo.compute_returns(obs)
        loss_dict = ppo.update()

        stop = time.time()

        step_time = stop - start
        tot_timesteps += num_steps_per_env * env.num_envs
        tot_time += step_time
        fps = num_steps_per_env * env.num_envs / step_time

        # -- Logging --
        if len(rewbuffer) > 0:
            mean_reward = sum(rewbuffer) / len(rewbuffer)
            mean_ep_len = sum(lenbuffer) / len(lenbuffer)
        else:
            mean_reward = 0.0
            mean_ep_len = 0.0

        writer.add_scalar("Train/mean_reward", mean_reward, it)
        writer.add_scalar("Train/mean_episode_length", mean_ep_len, it)
        writer.add_scalar("Train/fps", fps, it)
        for k, v in loss_dict.items():
            writer.add_scalar(f"Loss/{k}", v, it)

        # Log extras from env
        if "log" in extras:
            for k, v in extras["log"].items():
                val = v.item() if hasattr(v, "item") else v
                writer.add_scalar(f"Env/{k}", val, it)

        if it % 10 == 0:
            print(
                f"[{it:5d}/{max_iterations}]  reward={mean_reward:7.2f}  "
                f"ep_len={mean_ep_len:6.1f}  fps={fps:7.0f}  "
                f"surr={loss_dict.get('surrogate', 0):.4f}  val={loss_dict.get('value_function', 0):.4f}"
            )

        # -- Save --
        if it % save_interval == 0 or it == max_iterations - 1:
            path = os.path.join(log_dir, f"model_{it}.pt")
            torch.save({
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": ppo.optimizer.state_dict(),
                "iter": it,
            }, path)
            if it % save_interval == 0:
                print(f"[INFO] Saved checkpoint: {path}")

    writer.close()
    env.close()
    print(f"[INFO] Training complete. Total time: {tot_time:.0f}s, Total steps: {tot_timesteps:,}")


if __name__ == "__main__":
    main()
    simulation_app.close()

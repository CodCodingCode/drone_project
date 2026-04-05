"""Train VLA (PaliGemma 3B + LoRA) drone navigation policy.

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

import vla  # noqa: F401 — registers Isaac-VLADrone-Direct-v0

from vla.vla_drone_env import VLADroneEnvCfg
from vla.vla_policy import VLAActorModel, VLACriticModel, HierarchicalVLAActor, HierarchicalVLACritic
from vla.agents.rsl_rl_ppo_cfg import VLADronePPORunnerCfg


class VLAPolicy(nn.Module):
    """Wrapper that makes VLA actor+critic look like RSL-RL's ActorCritic."""

    is_recurrent = False

    def __init__(self, actor: VLAActorModel, critic: VLACriticModel):
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
    # Construct Hierarchical VLA actor + critic
    # ---------------------------------------------------------------
    waypoint_ckpt = os.path.join(_DRONE_PROJECT, "model_2998_waypoint.pt")
    print(f"[INFO] Constructing Hierarchical VLA actor (PaliGemma frozen + frozen waypoint policy)...")
    actor = HierarchicalVLAActor(
        waypoint_checkpoint_path=waypoint_ckpt,
        paligemma_model_name="google/paligemma-3b-pt-224",
        init_std=0.3,
        target_range=1.5,
    ).to(device)

    print("[INFO] Constructing Hierarchical VLA critic...")
    critic = HierarchicalVLACritic(flight_state_dim=9).to(device)
    critic._shared_paligemma = actor.paligemma
    print("[INFO] Linked shared PaliGemma backbone between actor and critic.")

    n_actor_train = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    n_critic_train = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"[INFO] Actor trainable: {n_actor_train:,} params")
    print(f"[INFO] Critic trainable: {n_critic_train:,} params")

    # ---------------------------------------------------------------
    # Create PPO
    # ---------------------------------------------------------------
    policy = VLAPolicy(actor, critic).to(device)

    alg_cfg = agent_cfg.algorithm.to_dict()
    alg_cfg.pop("class_name", None)

    ppo = PPO(policy=policy, device=device, **alg_cfg)

    # Only train non-PaliGemma, non-frozen-waypoint-buffer params
    trainable_params = [p for name, p in policy.named_parameters() if p.requires_grad]
    ppo.optimizer = torch.optim.Adam(trainable_params, lr=alg_cfg["learning_rate"])
    print(f"[INFO] PPO optimizer: {sum(p.numel() for p in trainable_params):,} trainable params")

    # Resume from checkpoint if provided
    if args_cli.resume_path:
        print(f"[INFO] Resuming from checkpoint: {args_cli.resume_path}")
        ckpt = torch.load(args_cli.resume_path, map_location=device, weights_only=False)
        missing, unexpected = policy.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
        print(f"  Loaded policy: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if "optimizer_state_dict" in ckpt:
            try:
                ppo.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print(f"  Loaded optimizer state")
            except Exception as e:
                print(f"  Skipping optimizer state: {e}")

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
    # Training loop (replaces OnPolicyRunner.learn)
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

    print(f"[INFO] Starting VLA training: {max_iterations} iterations, {args_cli.num_envs} envs")

    for it in range(max_iterations):
        start = time.time()

        # -- Rollout --
        with torch.inference_mode():
            for _ in range(num_steps_per_env):
                actions = ppo.act(obs)
                # Stash token features into rollout buffer so PPO update skips PaliGemma
                cached = policy.actor.paligemma._cache_val
                if cached is not None:
                    ppo.transition.observations["vla_token_features"] = cached.detach().half()
                policy.actor.paligemma.clear_cache()

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

        # -- PPO Update --
        ppo.compute_returns(obs)
        loss_dict = ppo.update()

        # -- Auxiliary supervised loss: predict ground-truth target from PaliGemma features --
        # Gives the attention head a dense signal for "where should I point" in addition to PPO.
        aux_batch_size = 32
        storage = ppo.storage
        flat_obs = storage.observations.flatten(0, 1)
        total_samples = flat_obs.batch_size[0]
        aux_idx = torch.randint(0, total_samples, (aux_batch_size,), device=device)
        aux_gt = flat_obs["target_gt_body"][aux_idx].clone()  # (B, 3)

        # Use cached token features from rollout (no PaliGemma re-forward)
        import torch.nn.functional as F_aux
        ppo.optimizer.zero_grad()
        token_features = flat_obs["vla_token_features"][aux_idx].clone().float()  # fp16 → fp32
        text_mask = flat_obs["text_mask"][aux_idx].clone()
        flight_state = flat_obs["policy"][aux_idx].clone()

        # Attention head forward WITH gradients
        target_pred = actor._compute_target_from_tokens(
            token_features, text_mask, flight_state
        )
        aux_loss = F_aux.mse_loss(target_pred, aux_gt)
        aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in actor.parameters() if p.requires_grad], 0.5
        )
        ppo.optimizer.step()
        loss_dict["aux_target_mse"] = aux_loss.item()
        actor.paligemma.clear_cache()

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
                f"surr={loss_dict.get('surrogate', 0):.4f}  val={loss_dict.get('value_function', 0):.4f}  "
                f"aux={loss_dict.get('aux_target_mse', 0):.3f}"
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

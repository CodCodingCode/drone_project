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
import torch.nn.functional as F
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

    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def _obs_dict(self, obs: TensorDict) -> dict:
        return {k: obs[k] for k in obs.keys()}

    def act(self, obs: TensorDict, masks=None, hidden_state=None, **kwargs) -> torch.Tensor:
        d = self._obs_dict(obs)
        return self.actor(d, masks=masks, hidden_state=hidden_state, stochastic_output=True)

    def evaluate(self, obs: TensorDict, masks=None, hidden_state=None, **kwargs) -> torch.Tensor:
        # Pass actor's cached scene features to critic (avoids re-running PaliGemma on 4 views)
        if hasattr(self.actor, '_critic_features') and self.actor._critic_features is not None:
            self.critic._cached_scene_features = self.actor._critic_features
        d = self._obs_dict(obs)
        result = self.critic(d, masks=masks, hidden_state=hidden_state)
        self.critic._cached_scene_features = None  # clear after use
        return result

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
        self.actor.reset(dones)
        self.critic.reset(dones)

    def act_inference(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        d = self._obs_dict(obs)
        return self.actor(d, stochastic_output=False)

    def get_hidden_states(self):
        return None


def compute_aux_weight(iteration: int, cfg) -> float:
    """Anneal auxiliary target supervision weight over training.

    Schedule: full weight → linear decay → residual floor.
    """
    if iteration < cfg.aux_warmup_end:
        return cfg.aux_loss_weight
    elif iteration < cfg.aux_decay_end:
        progress = (iteration - cfg.aux_warmup_end) / (cfg.aux_decay_end - cfg.aux_warmup_end)
        return cfg.aux_loss_weight - progress * (cfg.aux_loss_weight - cfg.aux_min_weight)
    else:
        return cfg.aux_min_weight


def ppo_update_with_aux(ppo, policy, aux_weight: float, aux_optimizer=None, aux_params=None) -> dict[str, float]:
    """PPO update with auxiliary target supervision loss on the cross-attention head."""
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_aux_loss = 0.0
    mean_cls_loss = 0.0
    mean_cls_accuracy = 0.0
    mean_target_l2_error = 0.0
    mean_err_x = 0.0
    mean_err_y = 0.0
    mean_err_z = 0.0
    mean_gt_dist = 0.0  # how far away the GT target actually is
    mean_attn_spatial_err = 0.0  # raw geometric readout error (before LSTM/MLP)

    policy.actor._force_lstm_reset = True  # prevent stale LSTM state during PPO update
    generator = ppo.storage.mini_batch_generator(ppo.num_mini_batches, ppo.num_learning_epochs)

    for (
        obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hidden_states_batch,
        masks_batch,
    ) in generator:
        original_batch_size = obs_batch.batch_size[0]

        # Recompute actions/values for current policy params
        policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
        actions_log_prob_batch = policy.get_actions_log_prob(actions_batch)
        value_batch = policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
        mu_batch = policy.action_mean[:original_batch_size]
        sigma_batch = policy.action_std[:original_batch_size]
        entropy_batch = policy.entropy[:original_batch_size]

        # Adaptive LR
        if ppo.desired_kl is not None and ppo.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1e-5)
                    + (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())
                    / (2.0 * sigma_batch.square())
                    - 0.5,
                    axis=-1,
                )
                kl_mean = kl.mean()
                if kl_mean > ppo.desired_kl * 2.0:
                    ppo.learning_rate = max(5e-6, ppo.learning_rate / 1.5)
                elif kl_mean < ppo.desired_kl / 2.0 and kl_mean > 0.0:
                    ppo.learning_rate = min(1e-2, ppo.learning_rate * 1.5)
                for pg in ppo.optimizer.param_groups:
                    pg["lr"] = ppo.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze())
        surrogate = -advantages_batch.squeeze() * ratio
        surrogate_clipped = -advantages_batch.squeeze() * ratio.clamp(
            1.0 - ppo.clip_param, 1.0 + ppo.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value loss
        if ppo.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -ppo.clip_param, ppo.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        # Auxiliary target supervision loss (computed in pre-tanh space to avoid saturation)
        predicted_logits = policy.actor._last_target_logits  # (N, 3) pre-tanh, has grad_fn
        gt_target = obs_batch["target_gt_body"]              # ground truth (may be padded in recurrent mode)
        obj_target_raw = obs_batch["target_obj_idx"]         # object class (may be padded)

        obj_logits = policy.actor._last_obj_logits

        # Convert ground truth to pre-tanh space: atanh(gt / target_range), clamped to avoid inf
        gt_normalized = (gt_target / policy.actor.target_range).clamp(-0.999, 0.999)
        gt_logits = torch.atanh(gt_normalized)
        aux_loss = F.mse_loss(predicted_logits, gt_logits)

        # Object classification loss (which of 3 objects does the command refer to?)
        obj_target = obj_target_raw.long()                   # (N,) class indices
        cls_loss = F.cross_entropy(obj_logits, obj_target)
        combined_aux_loss = aux_loss + cls_loss

        # PPO loss (critic + std_param; cross-attention head is detached from surrogate)
        ppo_loss = (
            surrogate_loss
            + ppo.value_loss_coef * value_loss
            - ppo.entropy_coef * entropy_batch.mean()
        )

        # Two separate optimizer steps so PPO's adaptive LR doesn't throttle aux learning
        ppo.optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()
            ppo_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(policy.parameters(), ppo.max_grad_norm)
            ppo.optimizer.step()
            # Aux step: position MSE + object classification CE
            # NOTE: PPO grads are still on aux_params (not zeroed), so
            # aux_optimizer steps combined RL + aux gradients.
            (aux_weight * combined_aux_loss).backward()
            if aux_params is not None:
                nn.utils.clip_grad_norm_(aux_params, ppo.max_grad_norm)
            aux_optimizer.step()
        else:
            # Fallback: single optimizer (original behavior)
            loss = ppo_loss + aux_weight * combined_aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), ppo.max_grad_norm)
            ppo.optimizer.step()

        # Diagnostic: per-axis and L2 error between predicted and GT target
        with torch.no_grad():
            predicted_target = policy.actor._last_target_body  # (B, 3) post-tanh
            err = predicted_target - gt_target                  # (B, 3)
            target_l2 = err.norm(dim=-1).mean()
            axis_err = err.abs().mean(dim=0)                    # (3,) mean absolute error per axis
            # Diagnostic: how good is the raw geometric attention readout?
            if hasattr(policy.actor, '_last_attn_spatial') and policy.actor._last_attn_spatial is not None:
                attn_err = (policy.actor._last_attn_spatial - gt_target).norm(dim=-1).mean()
            else:
                attn_err = torch.tensor(0.0)

        # Classification accuracy
        with torch.no_grad():
            obj_pred = obj_logits.argmax(dim=-1)
            cls_acc = (obj_pred == obj_target).float().mean()

        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_entropy += entropy_batch.mean().item()
        mean_aux_loss += aux_loss.item()
        mean_cls_loss += cls_loss.item()
        mean_cls_accuracy += cls_acc.item()
        mean_target_l2_error += target_l2.item()
        mean_err_x += axis_err[0].item()
        mean_err_y += axis_err[1].item()
        mean_err_z += axis_err[2].item()
        mean_gt_dist += gt_target.norm(dim=-1).mean().item()
        mean_attn_spatial_err += attn_err.item()

    num_updates = ppo.num_learning_epochs * ppo.num_mini_batches
    ppo.storage.clear()
    policy.actor._force_lstm_reset = False  # restore LSTM temporal tracking for rollout

    return {
        "value_function": mean_value_loss / num_updates,
        "surrogate": mean_surrogate_loss / num_updates,
        "entropy": mean_entropy / num_updates,
        "aux_target_mse": mean_aux_loss / num_updates,
        "cls_loss": mean_cls_loss / num_updates,
        "cls_accuracy": mean_cls_accuracy / num_updates,
        "target_l2_error_m": mean_target_l2_error / num_updates,
        "target_err_x": mean_err_x / num_updates,
        "target_err_y": mean_err_y / num_updates,
        "target_err_z": mean_err_z / num_updates,
        "gt_target_dist_m": mean_gt_dist / num_updates,
        "attn_spatial_err_m": mean_attn_spatial_err / num_updates,
    }


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
        target_range=3.0,  # needs to cover full arena diagonal (~2.5m)
        lstm_hidden_dim=agent_cfg.lstm_hidden_dim,
    ).to(device)

    print("[INFO] Constructing Hierarchical VLA critic...")
    critic = HierarchicalVLACritic(
        flight_state_dim=9,
        lstm_hidden_dim=agent_cfg.lstm_hidden_dim,
    ).to(device)
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

    # PPO optimizer: critic + std_param only (cross-attention head trained by aux optimizer)
    ppo_param_names = set()
    ppo_params = []
    aux_params = []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        # Cross-attention head params go to aux optimizer
        if any(k in name for k in ["image_proj", "text_proj", "cross_attn", "target_mlp", "obj_classifier", "actor.lstm", "depth_encoder"]):
            aux_params.append(p)
        else:
            ppo_params.append(p)
            ppo_param_names.add(name)
    ppo.optimizer = torch.optim.Adam(ppo_params, lr=alg_cfg["learning_rate"])
    aux_optimizer = torch.optim.Adam(aux_params, lr=3e-4)  # faster LR for direct supervision
    print(f"[INFO] PPO optimizer: {sum(p.numel() for p in ppo_params):,} params (critic + std)")
    print(f"[INFO] Aux optimizer: {sum(p.numel() for p in aux_params):,} params (cross-attention head) @ lr=3e-4")

    # LoRA optimizer: PaliGemma LoRA adapters at very low LR
    lora_params = [p for n, p in policy.named_parameters() if "lora_" in n and p.requires_grad]
    lora_optimizer = None
    if lora_params:
        lora_optimizer = torch.optim.Adam(lora_params, lr=agent_cfg.lora_learning_rate)
        print(f"[INFO] LoRA optimizer: {sum(p.numel() for p in lora_params):,} params @ lr={agent_cfg.lora_learning_rate}")
    else:
        print("[WARN] No LoRA params found — LoRA training disabled")

    # Resume from checkpoint if provided. Tolerates architecture changes by
    # dropping keys whose shapes no longer match (e.g. LSTM/target_mlp after
    # the FALCON-style spatial-pathway refactor) so the compatible parts
    # (PaliGemma LoRA, cross-attention, projections, aux heads) warm-start.
    if args_cli.resume_path:
        print(f"[INFO] Resuming from checkpoint: {args_cli.resume_path}")
        ckpt = torch.load(args_cli.resume_path, map_location=device, weights_only=False)
        src_sd = ckpt.get("model_state_dict", {})
        tgt_sd = policy.state_dict()
        skipped_shape = []
        filtered_sd = {}
        for k, v in src_sd.items():
            if k in tgt_sd and tgt_sd[k].shape != v.shape:
                skipped_shape.append((k, tuple(v.shape), tuple(tgt_sd[k].shape)))
                continue
            filtered_sd[k] = v
        missing, unexpected = policy.load_state_dict(filtered_sd, strict=False)
        print(f"  Loaded policy: {len(missing)} missing, {len(unexpected)} unexpected, {len(skipped_shape)} shape-mismatched")
        for k, src_shape, tgt_shape in skipped_shape:
            print(f"    skip (shape) {k}: ckpt {src_shape} vs model {tgt_shape}")
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
    best_reward = -float("inf")  # track best reward for best-checkpoint saving
    tot_timesteps = 0
    tot_time = 0.0

    # Randomize initial episode lengths
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))

    print(f"[INFO] Starting VLA training: {max_iterations} iterations, {args_cli.num_envs} envs")

    for it in range(max_iterations):
        start = time.time()

        # -- Rollout --
        with torch.no_grad():  # no_grad (not inference_mode) for LSTM hidden state compatibility
            for _ in range(num_steps_per_env):
                actions = ppo.act(obs)
                # Stash concatenated image+text token features (fp16) so PPO replay skips PaliGemma
                if hasattr(policy.actor, '_cached_all_tokens') and policy.actor._cached_all_tokens is not None:
                    cached = policy.actor._cached_all_tokens  # (B, 1024+text_len, 2048)
                    # Pad or truncate to match placeholder size (1048 tokens)
                    T = ppo.transition.observations["vla_token_features"].shape[1]
                    if cached.shape[1] < T:
                        pad = torch.zeros(cached.shape[0], T - cached.shape[1], cached.shape[2], device=cached.device, dtype=cached.dtype)
                        cached = torch.cat([cached, pad], dim=1)
                    ppo.transition.observations["vla_token_features"] = cached[:, :T].half()
                    policy.actor._cached_all_tokens = None  # free memory

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

        # -- PPO Update with auxiliary target supervision --
        # Clone obs to escape inference_mode (LSTM needs autograd-compatible tensors)
        obs = obs.clone()
        ppo.compute_returns(obs)
        aux_weight = compute_aux_weight(it, agent_cfg)
        loss_dict = ppo_update_with_aux(ppo, policy, aux_weight, aux_optimizer, aux_params=aux_params)

        # -- LoRA fine-tuning step (small mini-batch, full grad through PaliGemma) --
        if lora_optimizer is not None and it >= agent_cfg.lora_warmup_iterations:
            lora_optimizer.zero_grad()
            lora_bs = min(agent_cfg.lora_mini_batch_size, env.num_envs)
            idx = torch.randperm(env.num_envs, device=device)[:lora_bs]
            lora_obs = {k: v[idx] if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            # Zero out cached tokens to force fresh PaliGemma forward with grad
            if "vla_token_features" in lora_obs:
                lora_obs["vla_token_features"] = torch.zeros_like(lora_obs["vla_token_features"])

            target_logits, obj_logits = policy.actor.forward_lora_grad(lora_obs)
            gt_target = lora_obs["target_gt_body"]
            gt_normalized = (gt_target / policy.actor.target_range).clamp(-0.999, 0.999)
            gt_logits = torch.atanh(gt_normalized)
            lora_loss = F.mse_loss(target_logits, gt_logits)
            obj_target = lora_obs["target_obj_idx"].long()
            lora_loss = lora_loss + F.cross_entropy(obj_logits, obj_target)

            lora_loss.backward()
            nn.utils.clip_grad_norm_(lora_params, agent_cfg.lora_max_grad_norm)
            lora_optimizer.step()

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
        writer.add_scalar("Curriculum/aux_weight", aux_weight, it)
        for k, v in loss_dict.items():
            writer.add_scalar(f"Loss/{k}", v, it)

        # Log extras from env
        if "log" in extras:
            for k, v in extras["log"].items():
                val = v.item() if hasattr(v, "item") else v
                writer.add_scalar(f"Env/{k}", val, it)

        if it % 10 == 0:
            # Extract key env diagnostics for console
            log = extras.get("log", {})
            succ_rate = log.get("Termination/success_rate", 0)
            wrong_rate = log.get("Termination/wrong_object_rate", 0)
            final_dist = log.get("Diagnostic/final_dist_to_target", 0)
            succ_rate = succ_rate.item() if hasattr(succ_rate, "item") else succ_rate
            wrong_rate = wrong_rate.item() if hasattr(wrong_rate, "item") else wrong_rate
            final_dist = final_dist.item() if hasattr(final_dist, "item") else final_dist

            ex = loss_dict.get('target_err_x', 0)
            ey = loss_dict.get('target_err_y', 0)
            ez = loss_dict.get('target_err_z', 0)
            gt_d = loss_dict.get('gt_target_dist_m', 0)
            cls_acc = loss_dict.get('cls_accuracy', 0)
            attn_err = loss_dict.get('attn_spatial_err_m', 0)

            # Per-object success rates
            def _g(key):
                v = log.get(key, 0)
                return v.item() if hasattr(v, "item") else v
            cube_s = _g("PerObject/cube_success_rate")
            sph_s = _g("PerObject/sphere_success_rate")
            cyl_s = _g("PerObject/cylinder_success_rate")
            cube_w = _g("PerObject/cube_wrong_rate")
            sph_w = _g("PerObject/sphere_wrong_rate")
            cyl_w = _g("PerObject/cylinder_wrong_rate")

            print(
                f"[{it:5d}/{max_iterations}]  reward={mean_reward:7.2f}  "
                f"ep_len={mean_ep_len:6.1f}  fps={fps:7.0f}  "
                f"aux={loss_dict.get('aux_target_mse', 0):.4f}  tgt_err={loss_dict.get('target_l2_error_m', 0):.3f}m  "
                f"attn_err={attn_err:.3f}m  "
                f"cls_acc={cls_acc:.0%}  "
                f"err_xyz=({ex:.2f},{ey:.2f},{ez:.2f})  gt_dist={gt_d:.2f}m  "
                f"succ={succ_rate:.0%}  wrong={wrong_rate:.0%}  dist={final_dist:.2f}m"
            )
            print(
                f"          per-obj succ: cube={cube_s:.0%} sph={sph_s:.0%} cyl={cyl_s:.0%}  "
                f"wrong: cube={cube_w:.0%} sph={sph_w:.0%} cyl={cyl_w:.0%}"
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

        # -- Save best checkpoint whenever mean_reward exceeds prior best --
        if mean_reward > best_reward and len(rewbuffer) >= 20:
            best_reward = mean_reward
            best_path = os.path.join(log_dir, "best_model.pt")
            torch.save({
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": ppo.optimizer.state_dict(),
                "iter": it,
                "best_reward": best_reward,
            }, best_path)

    writer.close()
    env.close()
    print(f"[INFO] Training complete. Total time: {tot_time:.0f}s, Total steps: {tot_timesteps:,}")


if __name__ == "__main__":
    main()
    simulation_app.close()

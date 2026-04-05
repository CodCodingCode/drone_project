"""Pi0-based drone navigation policy.

Uses Pi0's PaliGemma backbone (fine-tuned on embodied robot control)
as a frozen feature extractor, fuses vision-language features with
flight state, and outputs continuous actions via an MLP action head.

Architecture:
    Raw RGB (64x64) + Text tokens --> Pi0 PaliGemma (FROZEN)
                                           |
                                    Features (2048-dim)
                                           |
                                    Concat with flight state (9-dim)
                                           |
                                    MLP action head (2057->256->256->4)
"""

from __future__ import annotations

import gc
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pi0 feature extractor (fully frozen, no LoRA)
# ---------------------------------------------------------------------------

class Pi0FeatureExtractor(nn.Module):
    """Frozen Pi0 PaliGemma backbone as a feature extractor.

    Loads Pi0 open weights from HuggingFace (lerobot/pi0_base),
    extracts the PaliGemma component (fine-tuned on embodied tasks),
    and discards the action expert.

    Takes raw RGB images and tokenized text, returns 2048-dim features
    from the last hidden state (last-token extraction).
    """

    FEATURE_DIM = 2048  # PaliGemma's hidden size (same as vanilla)

    def __init__(
        self,
        model_id: str = "lerobot/pi0_base",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        print(f"[Pi0] Loading Pi0 from {model_id}...")
        self._load_pi0_backbone(model_id, dtype)

        self._dtype = dtype
        self._img_size = 224

        n_total = sum(p.numel() for p in self.parameters())
        print(f"[Pi0] PaliGemma backbone loaded. {n_total:,} params (all frozen)")

        # Feature cache (avoids double forward for actor + critic)
        self._cache_key: int | None = None
        self._cache_val: torch.Tensor | None = None

    def _load_pi0_backbone(self, model_id: str, dtype: torch.dtype):
        """Load Pi0, extract PaliGemma backbone, discard action expert."""
        try:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        except ImportError:
            raise ImportError(
                "LeRobot is required for Pi0. Install with:\n"
                '  pip install "lerobot[pi]"'
            )

        pi0_policy = PI0Policy.from_pretrained(model_id)

        # Extract the PaliGemma backbone from Pi0's two-expert architecture
        # Access path: PI0Policy.model.paligemma_with_expert.paligemma
        pi0_model = pi0_policy.model
        if hasattr(pi0_model, "paligemma_with_expert"):
            self.paligemma = pi0_model.paligemma_with_expert.paligemma
        elif hasattr(pi0_model, "paligemma"):
            self.paligemma = pi0_model.paligemma
        else:
            # Fallback: inspect available attributes
            attrs = [a for a in dir(pi0_model) if not a.startswith("_")]
            raise AttributeError(
                f"Cannot find PaliGemma backbone in Pi0 model. "
                f"Available attributes: {attrs}"
            )

        # Freeze everything
        for p in self.paligemma.parameters():
            p.requires_grad = False
        self.paligemma.eval()

        # Convert to desired dtype
        self.paligemma = self.paligemma.to(dtype)

        # Delete the rest of Pi0 to free memory
        del pi0_policy, pi0_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[Pi0] Extracted PaliGemma backbone, discarded action expert.")

    def preprocess_images(self, rgb: torch.Tensor) -> torch.Tensor:
        """GPU-side image preprocessing.

        Args:
            rgb: (N, 64, 64, 3) float [0, 1]

        Returns:
            (N, 3, 224, 224) normalized for SigLIP
        """
        x = rgb.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = F.interpolate(x, size=(self._img_size, self._img_size), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5  # SigLIP uses [-1, 1] normalization
        return x.to(self._dtype)

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features from Pi0's PaliGemma (always no_grad).

        Returns:
            (batch_size, 2048) float32 tensor -- last-token hidden state, detached.
        """
        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.paligemma.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        # Last token hidden state -- captures full image+text context via causal attention
        hidden = outputs.last_hidden_state  # (B, seq_len, 2048)
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        features = hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lengths]
        return features.float()

    def get_features(self, rgb: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Cached, mini-batched feature extraction (always detached)."""
        cache_key = rgb.data_ptr()
        if self._cache_key == cache_key and self._cache_val is not None:
            return self._cache_val

        pixel_values = self.preprocess_images(rgb)

        batch_size = pixel_values.shape[0]
        chunk_size = 64
        if batch_size <= chunk_size:
            features = self.forward(pixel_values, input_ids, attention_mask)
        else:
            chunks = []
            for i in range(0, batch_size, chunk_size):
                j = min(i + chunk_size, batch_size)
                chunk_feat = self.forward(
                    pixel_values[i:j], input_ids[i:j], attention_mask[i:j]
                )
                chunks.append(chunk_feat)
            features = torch.cat(chunks, dim=0)

        self._cache_key = cache_key
        self._cache_val = features.detach()
        return self._cache_val

    def clear_cache(self):
        self._cache_key = None
        self._cache_val = None


# ---------------------------------------------------------------------------
# MLP builder (same as VLA)
# ---------------------------------------------------------------------------

def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str = "elu") -> nn.Sequential:
    """Build a simple MLP matching RSL-RL's pattern."""
    act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Pi0 Actor
# ---------------------------------------------------------------------------

class Pi0ActorModel(nn.Module):
    """Pi0 actor: frozen Pi0 features + flight state -> continuous actions.

    Implements the interface RSL-RL's PPO expects:
      - forward(obs) -> actions
      - get_output_log_prob(outputs) -> log probs
      - output_distribution_params -> (mean, std)
      - output_entropy -> entropy
      - update_normalization(obs)
      - get_hidden_state() / reset()
    """

    is_recurrent = False

    def __init__(
        self,
        flight_state_dim: int = 9,
        action_dim: int = 4,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_std: float = 0.6,
        pi0_model_id: str = "lerobot/pi0_base",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.flight_state_dim = flight_state_dim
        self.action_dim = action_dim
        fused_dim = Pi0FeatureExtractor.FEATURE_DIM + flight_state_dim  # 2048 + 9

        # Pi0 feature extractor (fully frozen)
        self.pi0 = Pi0FeatureExtractor(model_id=pi0_model_id)

        # Flight state normalizer (running mean/std, only for the 9-dim state)
        self._obs_mean = nn.Parameter(torch.zeros(1, flight_state_dim), requires_grad=False)
        self._obs_var = nn.Parameter(torch.ones(1, flight_state_dim), requires_grad=False)
        self._obs_count = nn.Parameter(torch.tensor(1e-4), requires_grad=False)

        # Action head MLP
        self.mlp = _build_mlp(fused_dim, hidden_dims, action_dim, activation)

        # Gaussian distribution (learnable std)
        self._std_param = nn.Parameter(torch.ones(action_dim) * math.log(init_std))
        self._action_mean: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None

    def _normalize_flight_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self._obs_mean) / (self._obs_var.sqrt() + 1e-8)

    def update_normalization(self, obs: dict[str, torch.Tensor]):
        """Update running statistics for flight state only."""
        state = obs["policy"]
        batch_mean = state.mean(dim=0, keepdim=True)
        batch_var = state.var(dim=0, keepdim=True)
        batch_count = state.shape[0]

        delta = batch_mean - self._obs_mean
        total = self._obs_count + batch_count
        self._obs_mean.data = self._obs_mean + delta * batch_count / total
        self._obs_var.data = (
            self._obs_var * self._obs_count + batch_var * batch_count + delta**2 * self._obs_count * batch_count / total
        ) / total
        self._obs_count.data = total

    def forward(self, obs: dict[str, torch.Tensor], masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        """Forward pass: extract features, fuse, produce actions."""
        flight_state = obs["policy"]          # (B, 9)

        # Normalize flight state
        flight_norm = self._normalize_flight_state(flight_state)

        # Use pre-computed features if available (from rollout buffer), else compute
        if "vla_features" in obs and obs["vla_features"].any():
            vla_features = obs["vla_features"]
        else:
            rgb = obs["rgb"]
            text_tokens = obs["text_tokens"]
            text_mask = obs["text_mask"]
            vla_features = self.pi0.get_features(rgb, text_tokens.long(), text_mask.long())

        # Fuse and run action head
        fused = torch.cat([vla_features, flight_norm], dim=-1)  # (B, 2057)
        self._action_mean = self.mlp(fused)
        self._action_std = self._std_param.exp().expand_as(self._action_mean)

        if stochastic_output:
            dist = torch.distributions.Normal(self._action_mean, self._action_std)
            return dist.sample()
        return self._action_mean

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._action_mean, self._action_std

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Normal(self._action_mean, self._action_std)
        return dist.log_prob(outputs).sum(dim=-1)

    @property
    def output_entropy(self) -> torch.Tensor:
        dist = torch.distributions.Normal(self._action_mean, self._action_std)
        return dist.entropy().sum(dim=-1)

    def get_hidden_state(self):
        return None

    def reset(self, env_ids=None):
        self.pi0.clear_cache()


# ---------------------------------------------------------------------------
# Pi0 Critic
# ---------------------------------------------------------------------------

class Pi0CriticModel(nn.Module):
    """Pi0 critic: shares Pi0 backbone with actor, separate value MLP.

    The Pi0 reference is set AFTER construction via:
        critic._shared_pi0 = actor.pi0
    """

    is_recurrent = False

    def __init__(
        self,
        flight_state_dim: int = 9,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.flight_state_dim = flight_state_dim
        fused_dim = Pi0FeatureExtractor.FEATURE_DIM + flight_state_dim

        # Shared Pi0 (set after construction, NOT a submodule)
        self._shared_pi0: Pi0FeatureExtractor | None = None

        # Flight state normalizer
        self._obs_mean = nn.Parameter(torch.zeros(1, flight_state_dim), requires_grad=False)
        self._obs_var = nn.Parameter(torch.ones(1, flight_state_dim), requires_grad=False)
        self._obs_count = nn.Parameter(torch.tensor(1e-4), requires_grad=False)

        # Value head MLP
        self.mlp = _build_mlp(fused_dim, hidden_dims, 1, activation)

    def _normalize_flight_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self._obs_mean) / (self._obs_var.sqrt() + 1e-8)

    def update_normalization(self, obs: dict[str, torch.Tensor]):
        state = obs["policy"]
        batch_mean = state.mean(dim=0, keepdim=True)
        batch_var = state.var(dim=0, keepdim=True)
        batch_count = state.shape[0]
        delta = batch_mean - self._obs_mean
        total = self._obs_count + batch_count
        self._obs_mean.data = self._obs_mean + delta * batch_count / total
        self._obs_var.data = (
            self._obs_var * self._obs_count + batch_var * batch_count + delta**2 * self._obs_count * batch_count / total
        ) / total
        self._obs_count.data = total

    def forward(self, obs: dict[str, torch.Tensor], masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        flight_state = obs["policy"]
        flight_norm = self._normalize_flight_state(flight_state)

        # Use pre-computed features if available (from rollout buffer), else compute
        if "vla_features" in obs and obs["vla_features"].any():
            vla_features = obs["vla_features"]
        else:
            rgb = obs["rgb"]
            text_tokens = obs["text_tokens"]
            text_mask = obs["text_mask"]
            assert self._shared_pi0 is not None, "Call critic._shared_pi0 = actor.pi0 first"
            vla_features = self._shared_pi0.get_features(rgb, text_tokens.long(), text_mask.long())

        fused = torch.cat([vla_features, flight_norm], dim=-1)
        return self.mlp(fused)

    def get_hidden_state(self):
        return None

    def reset(self, env_ids=None):
        pass

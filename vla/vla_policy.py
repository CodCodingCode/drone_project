"""PaliGemma 3B VLA policy for drone navigation.

Wraps PaliGemma as a frozen feature extractor with LoRA adapters,
fuses vision-language features with flight state, and outputs
continuous actions via an MLP action head.

Architecture:
    Raw RGB (64×64) + Text tokens ──→ PaliGemma 3B (frozen + LoRA)
                                           ↓
                                    Features (2048-dim)
                                           ↓
                                    Concat with flight state (9-dim)
                                           ↓
                                    MLP action head (2057→256→256→4)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA adapter (manual implementation, no PEFT dependency)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a frozen linear layer."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # A: Kaiming init for good gradient flow, B: zero init so LoRA starts as identity
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale


# ---------------------------------------------------------------------------
# PaliGemma feature extractor
# ---------------------------------------------------------------------------

class PaliGemmaFeatureExtractor(nn.Module):
    """Frozen PaliGemma 3B with LoRA adapters as a feature extractor.

    Takes raw RGB images and tokenized text, returns 2048-dim features
    from the last hidden state (mean-pooled over sequence length).
    """

    FEATURE_DIM = 2048  # PaliGemma's hidden size

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-pt-224",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        from transformers import PaliGemmaForConditionalGeneration

        print(f"[VLA] Loading PaliGemma from {model_name}...")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype,
        )

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # Apply LoRA to targeted layers
        self._apply_lora(lora_rank, lora_alpha, lora_targets)

        # Enable gradient checkpointing to save memory
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self._dtype = dtype
        self._img_size = 224

        n_lora = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[VLA] PaliGemma loaded. LoRA params: {n_lora:,} / {n_total:,} total ({100*n_lora/n_total:.2f}%)")

        # Feature cache (avoids double forward for actor + critic)
        self._cache_key: int | None = None
        self._cache_val: torch.Tensor | None = None

    def _apply_lora(self, rank: int, alpha: float, targets: tuple[str, ...]):
        """Replace targeted projection layers with LoRA-wrapped versions."""
        replaced = 0
        for name, module in list(self.model.named_modules()):
            if not any(t in name for t in targets):
                continue
            if not isinstance(module, nn.Linear):
                continue
            # Navigate to parent and replace
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(self.model.named_modules())[parts[0]]
                setattr(parent, parts[1], LoRALinear(module, rank, alpha))
                replaced += 1
        print(f"[VLA] Applied LoRA to {replaced} layers (rank={rank}, alpha={alpha})")

    def preprocess_images(self, rgb: torch.Tensor) -> torch.Tensor:
        """GPU-side image preprocessing.

        Args:
            rgb: (N, 64, 64, 3) float [0, 1]

        Returns:
            (N, 3, 224, 224) normalized for SigLIP
        """
        x = rgb.permute(0, 3, 1, 2)  # NHWC → NCHW
        x = F.interpolate(x, size=(self._img_size, self._img_size), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5  # SigLIP uses [-1, 1] normalization
        return x.to(self._dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features from PaliGemma.

        Returns:
            (batch_size, 2048) float32 tensor — mean-pooled last hidden state.
        """
        with torch.cuda.amp.autocast(dtype=self._dtype):
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        # Mean-pool over sequence length (mask-aware)
        hidden = outputs.last_hidden_state  # (B, seq_len, 2048)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, seq_len, 1)
        features = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return features.float()  # back to fp32

    def get_features(self, rgb: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Cached feature extraction — avoids double forward for actor + critic."""
        cache_key = id(rgb)
        if self._cache_key == cache_key and self._cache_val is not None:
            return self._cache_val

        pixel_values = self.preprocess_images(rgb)
        features = self.forward(pixel_values, input_ids, attention_mask)

        self._cache_key = cache_key
        self._cache_val = features
        return features

    def clear_cache(self):
        self._cache_key = None
        self._cache_val = None


# ---------------------------------------------------------------------------
# VLA Actor (subclasses RSL-RL's MLPModel)
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


class VLAActorModel(nn.Module):
    """VLA actor: PaliGemma features + flight state → continuous actions.

    This is a standalone nn.Module (not subclassing MLPModel) that implements
    the interface RSL-RL's PPO expects:
      - forward(obs) → actions
      - get_output_log_prob(outputs) → log probs
      - output_distribution_params → (mean, std)
      - output_entropy → entropy
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
        # PaliGemma config
        paligemma_model_name: str = "google/paligemma-3b-pt-224",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.flight_state_dim = flight_state_dim
        self.action_dim = action_dim
        fused_dim = PaliGemmaFeatureExtractor.FEATURE_DIM + flight_state_dim  # 2048 + 9

        # PaliGemma feature extractor (frozen + LoRA)
        self.paligemma = PaliGemmaFeatureExtractor(
            model_name=paligemma_model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

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
        rgb = obs["rgb"]                      # (B, 64, 64, 3)
        text_tokens = obs["text_tokens"]      # (B, 32)
        text_mask = obs["text_mask"]          # (B, 32)

        # Normalize flight state
        flight_norm = self._normalize_flight_state(flight_state)

        # PaliGemma features (cached for critic reuse)
        vla_features = self.paligemma.get_features(rgb, text_tokens.long(), text_mask.long())  # (B, 2048)

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
        self.paligemma.clear_cache()


class VLACriticModel(nn.Module):
    """VLA critic: shares PaliGemma backbone with actor, separate value MLP.

    The PaliGemma reference is set AFTER construction via:
        critic._shared_paligemma = actor.paligemma
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
        fused_dim = PaliGemmaFeatureExtractor.FEATURE_DIM + flight_state_dim

        # Shared PaliGemma (set after construction, NOT a submodule)
        self._shared_paligemma: PaliGemmaFeatureExtractor | None = None

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
        rgb = obs["rgb"]
        text_tokens = obs["text_tokens"]
        text_mask = obs["text_mask"]

        flight_norm = self._normalize_flight_state(flight_state)

        # Reuse cached PaliGemma features from actor's forward pass
        assert self._shared_paligemma is not None, "Call critic._shared_paligemma = actor.paligemma first"
        vla_features = self._shared_paligemma.get_features(rgb, text_tokens.long(), text_mask.long())

        fused = torch.cat([vla_features, flight_norm], dim=-1)
        return self.mlp(fused)

    def get_hidden_state(self):
        return None

    def reset(self, env_ids=None):
        pass

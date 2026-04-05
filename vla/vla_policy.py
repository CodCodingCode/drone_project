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
            model_name, torch_dtype=dtype, attn_implementation="eager",
        )

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # Apply LoRA to targeted layers
        self._apply_lora(lora_rank, lora_alpha, lora_targets)

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

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features from PaliGemma (always no_grad to avoid OOM).

        Returns:
            (batch_size, 2048) float32 tensor — mean-pooled last hidden state, detached.
        """
        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        # Last token hidden state — captures full image+text context via causal attention
        # (mean-pooling dilutes the text signal across 256 image tokens)
        hidden = outputs.last_hidden_state  # (B, seq_len, 2048)
        # Find last non-padding token per batch element
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

    @torch.no_grad()
    def forward_tokens(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return full token-level hidden states (not pooled).

        Returns:
            (B, seq_len, 2048) float32 tensor — full sequence, detached.
        """
        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
        return outputs.last_hidden_state.float()  # (B, seq_len, 2048)

    def get_token_features(self, rgb: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Cached, mini-batched token-level feature extraction.

        Returns token sequence (B, seq_len, 2048) for attention-based heads.
        """
        cache_key = ("tokens", rgb.data_ptr())
        if self._cache_key == cache_key and self._cache_val is not None:
            return self._cache_val

        pixel_values = self.preprocess_images(rgb)

        batch_size = pixel_values.shape[0]
        chunk_size = 32  # smaller chunks since we're keeping full sequence (more memory)
        if batch_size <= chunk_size:
            features = self.forward_tokens(pixel_values, input_ids, attention_mask)
        else:
            chunks = []
            for i in range(0, batch_size, chunk_size):
                j = min(i + chunk_size, batch_size)
                chunk_feat = self.forward_tokens(
                    pixel_values[i:j], input_ids[i:j], attention_mask[i:j]
                )
                chunks.append(chunk_feat)
            features = torch.cat(chunks, dim=0)

        self._cache_key = cache_key
        self._cache_val = features.detach()
        return self._cache_val

    def forward_with_grad(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features WITH gradient tracking for LoRA fine-tuning.

        Same as forward() but allows backpropagation through LoRA adapters.
        Only used during the LoRA update step on small mini-batches.
        """
        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        # Last token hidden state — same as forward() but with gradients
        hidden = outputs.last_hidden_state
        seq_lengths = attention_mask.sum(dim=1) - 1
        features = hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lengths]
        return features.float()  # NOT detached — gradients flow through LoRA

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

        # Normalize flight state
        flight_norm = self._normalize_flight_state(flight_state)

        # Use pre-computed features if available (from rollout buffer), else compute
        if "vla_features" in obs and obs["vla_features"].any():
            vla_features = obs["vla_features"]
        else:
            rgb = obs["rgb"]
            text_tokens = obs["text_tokens"]
            text_mask = obs["text_mask"]
            vla_features = self.paligemma.get_features(rgb, text_tokens.long(), text_mask.long())

        # Fuse and run action head
        fused = torch.cat([vla_features, flight_norm], dim=-1)  # (B, 2057)
        self._action_mean = self.mlp(fused)
        self._action_std = self._std_param.exp().expand_as(self._action_mean)

        if stochastic_output:
            dist = torch.distributions.Normal(self._action_mean, self._action_std)
            return dist.sample()
        return self._action_mean

    def forward_with_grad_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with gradient flow through PaliGemma LoRA.

        Used ONLY during the LoRA fine-tuning step. Returns action means
        with a computation graph that connects back through LoRA adapters.
        """
        flight_state = obs["policy"]
        flight_norm = self._normalize_flight_state(flight_state)

        rgb = obs["rgb"]
        text_tokens = obs["text_tokens"]
        text_mask = obs["text_mask"]

        pixel_values = self.paligemma.preprocess_images(rgb)
        vla_features = self.paligemma.forward_with_grad(
            pixel_values, text_tokens.long(), text_mask.long()
        )

        fused = torch.cat([vla_features, flight_norm], dim=-1)
        self._action_mean = self.mlp(fused)
        self._action_std = self._std_param.exp().expand_as(self._action_mean)

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
        flight_norm = self._normalize_flight_state(flight_state)

        # Use pre-computed features if available (from rollout buffer), else compute
        if "vla_features" in obs and obs["vla_features"].any():
            vla_features = obs["vla_features"]
        else:
            rgb = obs["rgb"]
            text_tokens = obs["text_tokens"]
            text_mask = obs["text_mask"]
            assert self._shared_paligemma is not None, "Call critic._shared_paligemma = actor.paligemma first"
            vla_features = self._shared_paligemma.get_features(rgb, text_tokens.long(), text_mask.long())

        fused = torch.cat([vla_features, flight_norm], dim=-1)
        return self.mlp(fused)

    def get_hidden_state(self):
        return None

    def reset(self, env_ids=None):
        pass


# ---------------------------------------------------------------------------
# Hierarchical VLA Actor — PaliGemma → Waypoint Head → Frozen Waypoint Policy
# ---------------------------------------------------------------------------

class HierarchicalVLAActor(nn.Module):
    """Hierarchical VLA: PaliGemma → target waypoint → frozen waypoint nav policy.

    Architecture:
        RGB + text ──→ PaliGemma (frozen) ──→ 2048 features
                                                    ↓
                                              Waypoint head (trainable: 2048→128→3)
                                                    ↓
                                              target_pos_b (body frame, 3-dim)
                                                    ↓
                                              [flight_state_9 | target_pos_b | pos_error (=target_pos_b)]  ──→ (15-dim)
                                                    ↓
                                              Frozen waypoint nav policy (256×256 ELU)
                                                    ↓
                                              4-dim action (thrust + moment)

    Only the waypoint head (~263K params) is trainable.
    """

    is_recurrent = False

    def __init__(
        self,
        waypoint_checkpoint_path: str,
        paligemma_model_name: str = "google/paligemma-3b-pt-224",
        lora_rank: int = 0,  # ignored for compatibility
        lora_alpha: float = 0.0,
        init_std: float = 0.3,
        target_range: float = 2.5,  # max |target_pos_b| output
    ):
        super().__init__()
        self.target_range = target_range

        # 1. PaliGemma feature extractor (fully frozen — no LoRA)
        self.paligemma = PaliGemmaFeatureExtractor(
            model_name=paligemma_model_name,
            lora_rank=1,  # dummy, will be frozen
            lora_alpha=1.0,
        )
        for p in self.paligemma.parameters():
            p.requires_grad = False

        # 2. Attention-based waypoint head
        # Input: PaliGemma token sequence (B, seq_len, 2048) split into image tokens (first 256)
        # and text tokens (remaining). Text tokens attend over image tokens via cross-attention,
        # producing a command-conditioned scene summary.
        self._num_image_tokens = 256
        embed_dim = 256
        self.image_proj = nn.Linear(PaliGemmaFeatureExtractor.FEATURE_DIM, embed_dim)
        self.text_proj = nn.Linear(PaliGemmaFeatureExtractor.FEATURE_DIM, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )
        self.target_mlp = nn.Sequential(
            nn.Linear(embed_dim + 9, 128),
            nn.ELU(),
            nn.Linear(128, 3),
        )
        # Zero-init final layer so the drone starts by hovering
        nn.init.zeros_(self.target_mlp[-1].weight)
        nn.init.zeros_(self.target_mlp[-1].bias)

        # 3. Frozen waypoint nav policy (loaded from checkpoint)
        self._build_waypoint_policy(waypoint_checkpoint_path)

        # Gaussian action distribution for PPO
        self._std_param = nn.Parameter(torch.ones(4) * math.log(init_std))
        self._action_mean: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None

        # Log param counts
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[VLA-Hier] Trainable: {n_train:,} / Total: {n_total:,} ({100*n_train/n_total:.3f}%)")

    def _build_waypoint_policy(self, ckpt_path: str):
        """Load waypoint nav policy weights as frozen buffers."""
        print(f"[VLA-Hier] Loading frozen waypoint policy from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["actor_state_dict"]
        self.register_buffer("wp_w0", state["mlp.0.weight"].clone())
        self.register_buffer("wp_b0", state["mlp.0.bias"].clone())
        self.register_buffer("wp_w1", state["mlp.2.weight"].clone())
        self.register_buffer("wp_b1", state["mlp.2.bias"].clone())
        self.register_buffer("wp_w2", state["mlp.4.weight"].clone())
        self.register_buffer("wp_b2", state["mlp.4.bias"].clone())
        self.register_buffer("wp_obs_mean", state["obs_normalizer._mean"].clone())
        self.register_buffer("wp_obs_std", state["obs_normalizer._std"].clone())

    def _waypoint_policy_forward(self, obs_15: torch.Tensor) -> torch.Tensor:
        """Frozen waypoint policy forward pass."""
        x = (obs_15 - self.wp_obs_mean) / (self.wp_obs_std + 1e-8)
        x = F.elu(F.linear(x, self.wp_w0, self.wp_b0))
        x = F.elu(F.linear(x, self.wp_w1, self.wp_b1))
        return F.linear(x, self.wp_w2, self.wp_b2)

    def _compute_target_from_tokens(self, token_features: torch.Tensor, text_mask: torch.Tensor, flight_state: torch.Tensor) -> torch.Tensor:
        """Attention-based target prediction from PaliGemma token sequence.

        Args:
            token_features: (B, seq_len, 2048)
            text_mask: (B, seq_len)  — 1 for real tokens, 0 for padding
            flight_state: (B, 9)

        Returns:
            target_body: (B, 3) in body frame, bounded by tanh * target_range
        """
        n_img = self._num_image_tokens  # 256
        image_tokens = token_features[:, :n_img]       # (B, 256, 2048)
        text_tokens = token_features[:, n_img:]        # (B, text_len, 2048)
        text_mask_only = text_mask[:, n_img:]          # (B, text_len)

        image_emb = self.image_proj(image_tokens)      # (B, 256, embed_dim)
        text_emb = self.text_proj(text_tokens)         # (B, text_len, embed_dim)

        # Cross-attention: text attends over image tokens
        # key_padding_mask: True where PADDING (opposite of attention_mask)
        key_padding_mask = None  # image tokens are all valid
        fused, _ = self.cross_attn(
            query=text_emb, key=image_emb, value=image_emb,
            key_padding_mask=key_padding_mask,
        )  # (B, text_len, embed_dim)

        # Masked mean over valid text tokens
        mask = text_mask_only.unsqueeze(-1).to(fused.dtype)  # (B, text_len, 1)
        scene_summary = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, embed_dim)

        # Concat with flight state → predict target
        head_input = torch.cat([scene_summary, flight_state], dim=-1)
        target_body = torch.tanh(self.target_mlp(head_input)) * self.target_range  # (B, 3)
        return target_body

    def forward(self, obs: dict[str, torch.Tensor], masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        # 1. Get PaliGemma token sequence (frozen, no grad)
        if "vla_token_features" in obs and obs["vla_token_features"].abs().sum() > 0:
            token_features = obs["vla_token_features"]
        else:
            rgb = obs["rgb"]
            text_tokens = obs["text_tokens"].long()
            text_mask = obs["text_mask"].long()
            token_features = self.paligemma.get_token_features(rgb, text_tokens, text_mask)

        # 2. Attention-based target prediction
        flight_state = obs["policy"]  # (B, 9)
        text_mask = obs["text_mask"]  # (B, seq_len)
        target_body = self._compute_target_from_tokens(token_features, text_mask, flight_state)
        self._last_target = target_body  # for auxiliary loss

        # 3. Build 15-dim observation for waypoint policy
        wp_obs = torch.cat([flight_state, target_body, target_body], dim=-1)  # (B, 15)

        # 4. Run frozen waypoint policy
        action_mean = self._waypoint_policy_forward(wp_obs)
        self._action_mean = action_mean
        self._action_std = self._std_param.exp().expand_as(action_mean)

        if stochastic_output:
            dist = torch.distributions.Normal(action_mean, self._action_std)
            return dist.sample()
        return action_mean

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

    def update_normalization(self, obs: dict[str, torch.Tensor]):
        # No-op: waypoint policy has its own frozen normalizer, and PaliGemma
        # features don't need normalization (already normalized internally).
        pass

    def get_hidden_state(self):
        return None

    def reset(self, env_ids=None):
        self.paligemma.clear_cache()


class HierarchicalVLACritic(nn.Module):
    """Value head on PaliGemma features + flight state.

    The PaliGemma backbone is shared with the actor (set after construction).
    Only this MLP is trainable.
    """

    is_recurrent = False

    def __init__(self, flight_state_dim: int = 9):
        super().__init__()
        fused_dim = PaliGemmaFeatureExtractor.FEATURE_DIM + flight_state_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )
        self._shared_paligemma: PaliGemmaFeatureExtractor | None = None

    def forward(self, obs: dict[str, torch.Tensor], masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        # Use token-level features (shared with actor cache) then pool
        if "vla_token_features" in obs and obs["vla_token_features"].abs().sum() > 0:
            token_features = obs["vla_token_features"]  # (B, seq_len, 2048)
        else:
            assert self._shared_paligemma is not None
            rgb = obs["rgb"]
            text_tokens = obs["text_tokens"].long()
            text_mask = obs["text_mask"].long()
            token_features = self._shared_paligemma.get_token_features(rgb, text_tokens, text_mask)

        # Pool token features for critic (simple mean over non-padded tokens)
        text_mask = obs["text_mask"]  # (B, seq_len)
        mask = text_mask.unsqueeze(-1).to(token_features.dtype)
        vla_features = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        flight_state = obs["policy"]
        fused = torch.cat([vla_features, flight_state], dim=-1)
        return self.mlp(fused)

    def update_normalization(self, obs: dict[str, torch.Tensor]):
        pass

    def get_hidden_state(self):
        return None

    def reset(self, env_ids=None):
        pass

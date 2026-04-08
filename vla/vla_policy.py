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
# Note: NOT using rsl_rl Memory module — manual LSTM state management is simpler
# for our multi-stage architecture (PaliGemma + cross-attn + LSTM + waypoint).


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
            rgb: (N, H, W, 3) float [0, 1]

        Returns:
            (N, 3, 224, 224) normalized for SigLIP
        """
        x = rgb.permute(0, 3, 1, 2)  # NHWC → NCHW
        if x.shape[-1] != self._img_size or x.shape[-2] != self._img_size:
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

    def forward_tokens_with_grad(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Like forward_tokens but with LoRA gradients enabled (no @torch.no_grad).

        Use only on small mini-batches for LoRA fine-tuning.
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
        lstm_hidden_dim: int = 128,
    ):
        super().__init__()
        self.target_range = target_range
        self._lstm_hidden_dim = lstm_hidden_dim

        # 1. PaliGemma feature extractor (frozen backbone with LoRA adapters)
        self.paligemma = PaliGemmaFeatureExtractor(
            model_name=paligemma_model_name,
            lora_rank=1,
            lora_alpha=1.0,
        )
        for p in self.paligemma.parameters():
            p.requires_grad = False
        # Selectively unfreeze LoRA adapters for backbone fine-tuning
        for name, p in self.paligemma.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

        # 2. Cross-attention head: text tokens attend over 4×256=1024 image tokens
        # 4 cameras (front/right/back/left), each producing 256 image tokens
        self._num_cameras = 4
        self._num_image_tokens_per_cam = 256
        self._num_image_tokens_total = self._num_cameras * self._num_image_tokens_per_cam  # 1024
        embed_dim = 256
        self.image_proj = nn.Linear(PaliGemmaFeatureExtractor.FEATURE_DIM, embed_dim)
        self.text_proj = nn.Linear(PaliGemmaFeatureExtractor.FEATURE_DIM, embed_dim)

        # Positional encoding: 2D spatial (16x16 grid) + camera direction embedding
        self._grid_size = 16
        pos_enc_2d = self._build_2d_sinusoidal_encoding(self._grid_size, embed_dim)  # (256, embed_dim)
        # Repeat for 4 cameras + add learned camera direction embedding
        self.register_buffer("_image_pos_enc_2d", pos_enc_2d)  # (256, embed_dim)
        self.camera_embed = nn.Embedding(4, embed_dim)  # learned: which camera (front/right/back/left)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True
        )

        # Patch grid coordinates for attention-weighted spatial readout (per camera, 256 each)
        grid = self._grid_size
        patch_rows = torch.arange(grid).repeat_interleave(grid).float() / (grid - 1)  # (256,)
        patch_cols = torch.arange(grid).repeat(grid).float() / (grid - 1)              # (256,)
        # Tile for 4 cameras → (1024,)
        self.register_buffer("_patch_rows", patch_rows.repeat(4))
        self.register_buffer("_patch_cols", patch_cols.repeat(4))
        # Camera index for each of 1024 tokens (0=front, 1=right, 2=back, 3=left)
        cam_ids = torch.arange(4).repeat_interleave(256)  # (1024,)
        self.register_buffer("_patch_cam_ids", cam_ids)
        # Known camera body-frame directions (from env config)
        # Front: +X, Right: +Y, Back: -X, Left: -Y
        cam_forward = torch.tensor([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
        ], dtype=torch.float32)
        cam_right_dir = torch.tensor([
            [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0],
        ], dtype=torch.float32)
        cam_up_dir = torch.tensor([
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        ], dtype=torch.float32)
        self.register_buffer("_cam_forward", cam_forward)
        self.register_buffer("_cam_right_dir", cam_right_dir)
        self.register_buffer("_cam_up_dir", cam_up_dir)
        # FOV geometry: focal_length=10, horizontal_aperture=20 → ~90° FOV
        self._pixel_to_ray_scale = 20.0 / 10.0  # aperture / focal_length

        # Object classification head: which of 3 objects does the command refer to?
        # Branches from scene_summary — provides direct "wrong object" gradient to cross-attention.
        self.obj_classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3),  # 3 classes: cube=0, sphere=1, cylinder=2
        )

        # Depth pooling: 224x224 depth map → 16x16 → 256 per-patch depth values
        self._depth_pool = nn.AvgPool2d(kernel_size=14, stride=14)  # 224/14 = 16

        # LSTM temporal memory: accumulates spatial context across frames.
        # Input: scene_summary(256) + attn_spatial(3: row,col,depth) + flight_state(9) = 268
        # attn_spatial gives explicit WHERE from attention-weighted patch coordinates + depth.
        self.lstm = nn.LSTM(
            input_size=embed_dim + 3 + 9,  # scene_summary + attn_spatial + flight_state
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # Hidden state maintained across steps during rollout, reset on episode done
        self._lstm_h: torch.Tensor | None = None  # (1, num_envs, hidden)
        self._lstm_c: torch.Tensor | None = None  # (1, num_envs, hidden)
        self._force_lstm_reset = False  # set True during PPO update to avoid stale state

        # Target prediction MLP: reads LSTM output (temporal spatial memory)
        self.target_mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3),
        )
        # Small random init on final layer
        nn.init.uniform_(self.target_mlp[-1].weight, -0.01, 0.01)
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

    @staticmethod
    def _build_2d_sinusoidal_encoding(grid_size: int, embed_dim: int) -> torch.Tensor:
        """2D sinusoidal positional encoding for a grid_size x grid_size patch grid."""
        half = embed_dim // 2
        positions = torch.arange(grid_size, dtype=torch.float32)
        dim_indices = torch.arange(half // 2, dtype=torch.float32)
        freqs = 1.0 / (10000.0 ** (2 * dim_indices / half))
        # Row encoding (first half of dims)
        row_idx = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (G, half//2)
        row_enc = torch.cat([row_idx.sin(), row_idx.cos()], dim=-1)  # (G, half)
        # Col encoding (second half of dims)
        col_idx = positions.unsqueeze(1) * freqs.unsqueeze(0)
        col_enc = torch.cat([col_idx.sin(), col_idx.cos()], dim=-1)  # (G, half)
        # Combine: patch (i,j) gets [row_enc[i], col_enc[j]]
        pos = torch.zeros(grid_size * grid_size, embed_dim)
        for i in range(grid_size):
            for j in range(grid_size):
                pos[i * grid_size + j, :half] = row_enc[i]
                pos[i * grid_size + j, half:] = col_enc[j]
        return pos

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

    def _compute_scene_summary(self, all_image_tokens: torch.Tensor, text_tokens_feat: torch.Tensor, text_mask_only: torch.Tensor, patch_depths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-attention scene summary + attention-weighted spatial coordinates.

        Args:
            all_image_tokens: (B, 1024, 2048) — 4 cameras × 256 tokens each
            text_tokens_feat: (B, text_len, 2048) — text token features
            text_mask_only: (B, text_len) — text attention mask
            patch_depths: (B, 1024) — per-patch depth values across all 4 cameras

        Returns:
            scene_summary: (B, embed_dim) — WHAT object (semantic)
            attn_spatial: (B, 3) — WHERE: attention-weighted (row, col, depth)
        """
        B = all_image_tokens.shape[0]

        # Project image tokens and add positional + camera embeddings
        image_emb = self.image_proj(all_image_tokens)  # (B, 1024, embed_dim)
        # Add 2D spatial encoding (same pattern repeated 4 times)
        pos_enc = self._image_pos_enc_2d.repeat(self._num_cameras, 1)  # (1024, embed_dim)
        image_emb = image_emb + pos_enc
        # Add camera direction embedding (which of 4 cameras each token is from)
        cam_emb = self.camera_embed(self._patch_cam_ids)  # (1024, embed_dim)
        image_emb = image_emb + cam_emb

        text_emb = self.text_proj(text_tokens_feat)  # (B, text_len, embed_dim)

        # Cross-attention: text attends over ALL 1024 image tokens (full 360°)
        fused, attn_weights = self.cross_attn(
            query=text_emb, key=image_emb, value=image_emb,
            average_attn_weights=True,
        )  # fused: (B, text_len, embed_dim), attn_weights: (B, text_len, 1024)

        # Scene summary: mean-pool fused output over valid text tokens (semantic: WHAT)
        mask = text_mask_only.unsqueeze(-1).to(fused.dtype)  # (B, text_len, 1)
        scene_summary = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, embed_dim)

        # Camera-aware geometric 3D projection (WHERE)
        mask_2d = text_mask_only.to(attn_weights.dtype)
        avg_attn = (attn_weights * mask_2d.unsqueeze(-1)).sum(dim=1) / mask_2d.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1024)

        # Split attention by camera: (B, 1024) → (B, 4, 256)
        cam_attns = avg_attn.reshape(B, 4, 256)
        patch_rows_single = self._patch_rows[:256]
        patch_cols_single = self._patch_cols[:256]

        # Per-camera attention-weighted pixel coords and depth
        cam_rows = (cam_attns * patch_rows_single).sum(dim=-1)   # (B, 4)
        cam_cols = (cam_attns * patch_cols_single).sum(dim=-1)   # (B, 4)
        cam_depths_m = (cam_attns * patch_depths.reshape(B, 4, 256)).sum(dim=-1) * 20.0  # (B, 4) denormalized to meters

        # Convert pixel coords to ray offsets (centered at 0.5, scaled by FOV)
        ray_h = (cam_cols - 0.5) * self._pixel_to_ray_scale   # horizontal
        ray_v = (cam_rows - 0.5) * self._pixel_to_ray_scale   # vertical (row 0 = top)

        # 3D ray per camera in body frame, scaled by depth
        rays = (self._cam_forward.unsqueeze(0)
                + ray_h.unsqueeze(-1) * self._cam_right_dir.unsqueeze(0)
                - ray_v.unsqueeze(-1) * self._cam_up_dir.unsqueeze(0))  # (B, 4, 3)
        rays = F.normalize(rays, dim=-1)
        body_points = rays * cam_depths_m.unsqueeze(-1)  # (B, 4, 3)

        # Weighted average by per-camera attention mass
        cam_mass = cam_attns.sum(dim=-1)  # (B, 4)
        attn_spatial = (body_points * cam_mass.unsqueeze(-1)).sum(dim=1)  # (B, 3)

        # Object classification logits
        self._last_obj_logits = self.obj_classifier(scene_summary)

        return scene_summary, attn_spatial

    def forward(self, obs: dict[str, torch.Tensor], masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        # 1. PaliGemma token features for 4 camera views
        rgb = obs["rgb"]  # (B, 4, 224, 224, 3)
        text_tokens = obs["text_tokens"].long()  # (B, 280)
        text_mask = obs["text_mask"].long()       # (B, 280)
        B = rgb.shape[0]
        n_img = self._num_image_tokens_per_cam  # 256

        # Check for cached token features from rollout buffer (skips PaliGemma during PPO update)
        if "vla_token_features" in obs and obs["vla_token_features"].abs().sum() > 0:
            cached = obs["vla_token_features"].float()  # (B, 1024, 2048) fp16→fp32
            all_image_tokens = cached[:, :self._num_image_tokens_total]  # (B, 1024, 2048)
            text_tokens_feat = cached[:, self._num_image_tokens_total:]  # (B, text_len, 2048)
            text_mask_only = text_mask[:, n_img:]  # (B, text_len)
        else:
            # Process each camera view through PaliGemma → collect image tokens
            all_image_tokens = []
            for cam_idx in range(self._num_cameras):
                cam_rgb = rgb[:, cam_idx]  # (B, 224, 224, 3)
                cam_tokens = self.paligemma.get_token_features(cam_rgb, text_tokens, text_mask)  # (B, 280, 2048)
                cam_image_tokens = cam_tokens[:, :n_img]  # (B, 256, 2048) — image tokens only
                all_image_tokens.append(cam_image_tokens)
                self.paligemma.clear_cache()  # clear between views
            all_image_tokens = torch.cat(all_image_tokens, dim=1)  # (B, 1024, 2048)

            # Extract text token features from the last PaliGemma pass (text is the same for all views)
            text_tokens_feat = cam_tokens[:, n_img:]  # (B, text_len, 2048)
            text_mask_only = text_mask[:, n_img:]     # (B, text_len)

            # Cache concatenated features so train loop can store them in rollout buffer
            self._cached_all_tokens = torch.cat([all_image_tokens, text_tokens_feat], dim=1).detach()  # (B, 1024+text_len, 2048)

        # 2. Pool depth maps to per-patch values — 4 cameras × 256 patches = 1024
        flight_state = obs["policy"]  # (B, 9)
        depth = obs["depth"]  # (B, 4, 224, 224)
        # Pool each view: (B, 4, 224, 224) → (B, 4, 16, 16) → (B, 1024)
        depth_flat = depth.reshape(B * 4, 1, 224, 224)
        patch_depths = self._depth_pool(depth_flat).reshape(B, -1)  # (B, 1024)

        # 3. Cross-attention over 1024 image tokens (full 360° view)
        scene_summary, attn_spatial = self._compute_scene_summary(
            all_image_tokens, text_tokens_feat, text_mask_only, patch_depths
        )

        # Stash mean-pooled features for critic (avoids re-running PaliGemma)
        self._critic_features = all_image_tokens.mean(dim=1).detach()  # (B, 2048)

        # 4. LSTM temporal memory — accumulates spatial context across frames
        lstm_input = torch.cat([scene_summary, attn_spatial, flight_state], dim=-1)  # (B, 268)
        lstm_input = lstm_input.unsqueeze(1)  # (B, 1, 265) — single timestep, batch_first

        if self._force_lstm_reset:
            # PPO update: observations are shuffled, temporal state is meaningless
            lstm_out, (self._lstm_h, self._lstm_c) = self.lstm(lstm_input)
        elif self._lstm_h is not None and self._lstm_h.shape[1] == lstm_input.shape[0]:
            lstm_out, (self._lstm_h, self._lstm_c) = self.lstm(
                lstm_input, (self._lstm_h.detach(), self._lstm_c.detach())
            )
        else:
            # First step or batch size changed — no prior state
            lstm_out, (self._lstm_h, self._lstm_c) = self.lstm(lstm_input)

        memory_out = lstm_out.squeeze(1)  # (B, hidden)

        # 5. Target prediction: residual correction on top of geometric attention readout
        # attn_spatial is already a decent 3D estimate from attention-weighted depth + camera rays.
        # The MLP only needs to learn a small correction, not the full position from scratch.
        attn_logits = torch.atanh((attn_spatial / self.target_range).clamp(-0.999, 0.999))  # (B, 3)
        correction_logits = self.target_mlp(memory_out)                 # (B, 3) small correction
        target_logits = attn_logits + correction_logits                 # (B, 3) pre-tanh
        target_body = torch.tanh(target_logits) * self.target_range     # (B, 3) bounded
        self._last_target_logits = target_logits
        self._last_target_body = target_body

        # 6. Build 15-dim observation for frozen waypoint policy
        pos_error_w = obs["pos_error_w"]  # (B, 3)
        wp_obs = torch.cat([flight_state, target_body, pos_error_w], dim=-1)  # (B, 15)

        # 7. Run frozen waypoint policy
        action_mean = self._waypoint_policy_forward(wp_obs)
        self._action_mean = action_mean
        self._action_std = self._std_param.exp().expand_as(action_mean)

        if stochastic_output:
            dist = torch.distributions.Normal(action_mean, self._action_std)
            return dist.sample()
        return action_mean

    def forward_lora_grad(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with LoRA gradients for backbone fine-tuning. Small batch only.

        Returns (target_logits, obj_logits) with grad through LoRA adapters.
        """
        rgb = obs["rgb"]  # (B, 4, 224, 224, 3)
        text_tokens = obs["text_tokens"].long()
        text_mask = obs["text_mask"].long()
        B = rgb.shape[0]
        n_img = self._num_image_tokens_per_cam

        # Run PaliGemma WITH grad (through LoRA) for each camera view
        all_image_tokens = []
        for cam_idx in range(self._num_cameras):
            cam_rgb = rgb[:, cam_idx]
            pixel_values = self.paligemma.preprocess_images(cam_rgb)
            cam_tokens = self.paligemma.forward_tokens_with_grad(
                pixel_values, text_tokens, text_mask
            )
            all_image_tokens.append(cam_tokens[:, :n_img])
            self.paligemma.clear_cache()
        all_image_tokens = torch.cat(all_image_tokens, dim=1)  # (B, 1024, 2048)

        # Text tokens from last camera pass
        text_tokens_feat = cam_tokens[:, n_img:]
        text_mask_only = text_mask[:, n_img:]

        depth = obs["depth"]
        depth_flat = depth.reshape(B * self._num_cameras, 1, 224, 224)
        patch_depths = self._depth_pool(depth_flat).reshape(B, -1)

        scene_summary, attn_spatial = self._compute_scene_summary(
            all_image_tokens, text_tokens_feat, text_mask_only, patch_depths
        )

        flight_state = obs["policy"]
        lstm_input = torch.cat([scene_summary, attn_spatial, flight_state], dim=-1).unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_input)  # fresh LSTM, no hidden state
        memory_out = lstm_out.squeeze(1)

        attn_logits = torch.atanh((attn_spatial / self.target_range).clamp(-0.999, 0.999))
        correction_logits = self.target_mlp(memory_out)
        target_logits = attn_logits + correction_logits
        return target_logits, self._last_obj_logits

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

    def reset(self, dones=None):
        # Reset LSTM hidden state for done environments
        if dones is not None and self._lstm_h is not None:
            done_mask = (dones == 1) if dones.dtype != torch.bool else dones
            self._lstm_h[:, done_mask, :] = 0.0
            self._lstm_c[:, done_mask, :] = 0.0
        elif dones is None:
            self._lstm_h = None
            self._lstm_c = None
        self.paligemma.clear_cache()


class HierarchicalVLACritic(nn.Module):
    """Value head on PaliGemma features + flight state.

    The PaliGemma backbone is shared with the actor (set after construction).
    Feedforward — no LSTM needed for value estimation (actor LSTM provides
    the temporal context that matters for target prediction).
    """

    is_recurrent = False

    def __init__(self, flight_state_dim: int = 9, lstm_hidden_dim: int = 128):
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
        # Use cached scene features from actor (avoids re-running PaliGemma on 4 views)
        # During rollout: actor runs first and stashes _cached_scene_features
        # During PPO update: vla_token_features from rollout buffer are available
        if hasattr(self, '_cached_scene_features') and self._cached_scene_features is not None:
            vla_features = self._cached_scene_features
        elif "vla_token_features" in obs and obs["vla_token_features"].abs().sum() > 0:
            token_features = obs["vla_token_features"].float()
            text_mask = obs["text_mask"]
            mask = text_mask.unsqueeze(-1).to(token_features.dtype)
            vla_features = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            # Fallback: zero features (should not happen in normal flow)
            vla_features = torch.zeros(obs["policy"].shape[0], PaliGemmaFeatureExtractor.FEATURE_DIM, device=obs["policy"].device)

        flight_state = obs["policy"]
        fused = torch.cat([vla_features, flight_state], dim=-1)
        return self.mlp(fused)

    def update_normalization(self, obs: dict[str, torch.Tensor]):
        pass

    def get_hidden_state(self):
        return None

    def reset(self, dones=None):
        pass

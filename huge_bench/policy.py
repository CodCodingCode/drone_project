"""BC policy: PaliGemma encoder (first_image + current_image + instruction) + MLP head.

Reuses `PaliGemmaFeatureExtractor` and `LoRALinear` from the existing VLA stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Reuse the existing PaliGemma feature extractor from vla/
_DRONE_ROOT = Path(__file__).resolve().parent.parent
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))
from vla.vla_policy import PaliGemmaFeatureExtractor  # noqa: E402

NUM_IMAGE_TOKENS = 256  # PaliGemma's image-placeholder count
TOKENIZER_NAME = "google/paligemma-3b-pt-224"


class HugeBCPolicy(nn.Module):
    """PaliGemma(first_image, instruction) || PaliGemma(image, instruction) || state -> action.

    PaliGemma backbone is frozen with LoRA on q_proj/v_proj. Only the LoRA
    adapters and the small MLP head are trainable.
    """

    STATE_DIM = 4
    ACTION_DIM = 4

    def __init__(
        self,
        max_text_length: int = 64,   # text-only token budget; total seq = 256 + max_text_length
        hidden_dims: tuple[int, ...] = (512, 512),
        dtype: torch.dtype = torch.float16,
        lora_rank: int = 8,
    ):
        super().__init__()
        from transformers import AutoProcessor

        self.max_text_length = max_text_length
        self.feat = PaliGemmaFeatureExtractor(
            model_name=TOKENIZER_NAME, lora_rank=lora_rank, dtype=dtype,
        )
        self.processor = AutoProcessor.from_pretrained(TOKENIZER_NAME)
        # Force right-padding so attention_mask.sum()-1 indexes the actual last
        # text token (Gemma's default is left-padding, which would break
        # PaliGemmaFeatureExtractor's last-token extraction).
        self.processor.tokenizer.padding_side = "right"
        self._image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")

        feat_dim = self.feat.FEATURE_DIM  # 2048
        head_in = 2 * feat_dim + self.STATE_DIM  # first + current + state

        # Small head MLP — only trainable params besides LoRA
        layers: list[nn.Module] = []
        prev = head_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, self.ACTION_DIM))
        self.head = nn.Sequential(*layers)

    # ---------------------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------------------
    def tokenize(self, instructions: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, attention_mask) of shape (B, 256 + max_text_length).

        Mirrors the pattern in vla/vla_drone_env.py: tokenize text with a small
        budget, then prepend 256 image-placeholder tokens.
        """
        prefixed = ["\n" + s for s in instructions]
        tok = self.processor.tokenizer(
            prefixed,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )
        b = len(instructions)
        img_ids = torch.full((b, NUM_IMAGE_TOKENS), self._image_token_id, dtype=torch.long)
        img_mask = torch.ones(b, NUM_IMAGE_TOKENS, dtype=torch.long)
        input_ids = torch.cat([img_ids, tok["input_ids"]], dim=1).to(device)
        attention_mask = torch.cat([img_mask, tok["attention_mask"]], dim=1).to(device)
        return input_ids, attention_mask

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def _features(self, rgb: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  with_grad: bool) -> torch.Tensor:
        # rgb: (B, H, W, 3) in [-1, 1] (already SigLIP-normalized by dataset)
        # PaliGemmaFeatureExtractor.preprocess_images expects [0,1] then re-normalizes.
        # Convert back: [-1,1] -> [0,1].
        rgb01 = (rgb + 1.0) * 0.5
        pixel_values = self.feat.preprocess_images(rgb01)
        if with_grad:
            return self.feat.forward_with_grad(pixel_values, input_ids, attention_mask)
        return self.feat.forward(pixel_values, input_ids, attention_mask)

    def forward(self, batch: dict, with_grad_through_lora: bool = True) -> torch.Tensor:
        """Predict actions (B, 4)."""
        device = next(self.head.parameters()).device
        instructions: list[str] = batch["instruction"]
        first_image = batch["first_image"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)

        input_ids, attention_mask = self.tokenize(instructions, device)
        # Clear feat cache to avoid stale data (we run two different RGB tensors back-to-back)
        self.feat.clear_cache()
        feat_first = self._features(first_image, input_ids, attention_mask, with_grad_through_lora)
        self.feat.clear_cache()
        feat_curr = self._features(image, input_ids, attention_mask, with_grad_through_lora)

        x = torch.cat([feat_first, feat_curr, state], dim=-1)
        return self.head(x)

    # ---------------------------------------------------------------------
    # Param groups
    # ---------------------------------------------------------------------
    def lora_parameters(self):
        for p in self.feat.parameters():
            if p.requires_grad:
                yield p

    def head_parameters(self):
        return self.head.parameters()

"""SigLIP-based language and vision grounder for object identification.

Loads google/siglip-base-patch16-224 and encodes text commands and RGB
images into 768-dim normalized embeddings. Both encoders are frozen —
no gradients.

Name is kept as `CLIPGrounder` for drop-in compatibility with the rest
of the codebase; internally it uses SigLIP which handles low-resolution
inputs better than CLIP.

Usage:
    grounder = CLIPGrounder(device="cuda")
    embs = grounder.encode_texts(["go to the square", "fly to the ball"])
    # embs: Tensor(2, 768), normalized, on device

    img_embs = grounder.encode_images(rgb_tensor)
    # rgb_tensor: (N, H, W, 3) uint8, img_embs: (N, 768), normalized
"""

import torch
import torch.nn.functional as F

# SigLIP embedding dimension (google/siglip-base-patch16-224)
EMBEDDING_DIM = 768

# SigLIP image normalization constants
_SIGLIP_IMAGE_MEAN = [0.5, 0.5, 0.5]
_SIGLIP_IMAGE_STD = [0.5, 0.5, 0.5]

# SigLIP was trained at 224x224 — resize camera images to match
_SIGLIP_INPUT_SIZE = 224

# SigLIP tokenizer max length
_SIGLIP_MAX_LEN = 64


class CLIPGrounder:
    """Frozen SigLIP text + vision encoder for language-to-object grounding.

    Encodes natural language navigation commands and camera images into
    SigLIP embeddings (768-dim). These embeddings are fed to the RL policy
    as part of the observation so it can match language commands to visual
    objects.
    """

    def __init__(self, device: str = "cuda"):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for CLIPGrounder. "
                "Install with: pip install transformers sentencepiece"
            )

        self.device = device
        print("[SigLIPGrounder] Loading google/siglip-base-patch16-224...")
        self._model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
        self._processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self._tokenizer = self._processor.tokenizer

        # Freeze all parameters — we only use it as a feature extractor
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

        # Pre-compute image normalization tensors on device: shape (1, 3, 1, 1)
        self._img_mean = torch.tensor(_SIGLIP_IMAGE_MEAN, device=device).view(1, 3, 1, 1)
        self._img_std = torch.tensor(_SIGLIP_IMAGE_STD, device=device).view(1, 3, 1, 1)

        print(f"[SigLIPGrounder] Ready (text + vision, {EMBEDDING_DIM}-dim).")

    @torch.no_grad()
    def encode_texts(self, commands: list[str]) -> torch.Tensor:
        """Encode a batch of text commands into normalized SigLIP embeddings.

        Args:
            commands: List of natural language strings, e.g. ["go to the square"]

        Returns:
            Tensor of shape (len(commands), 768), L2-normalized, on self.device.
        """
        # SigLIP requires fixed-length padding (unlike CLIP which supports padding=True)
        inputs = self._tokenizer(
            commands,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=_SIGLIP_MAX_LEN,
        ).to(self.device)
        embs = self._model.get_text_features(**inputs)  # (N, 768)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs

    @torch.no_grad()
    def encode_images(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode a batch of RGB images into normalized SigLIP embeddings.

        All preprocessing runs on GPU — no CPU round-trip.

        Args:
            rgb: (N, H, W, 3) uint8 tensor from TiledCamera.

        Returns:
            Tensor of shape (N, 768), L2-normalized, on self.device.
        """
        # (N, H, W, 3) uint8 -> (N, 3, H, W) float32 [0, 1]
        x = rgb.to(dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
        # Bilinear resize to 224x224 (SigLIP's expected input)
        x = F.interpolate(x, size=(_SIGLIP_INPUT_SIZE, _SIGLIP_INPUT_SIZE), mode="bilinear", align_corners=False)
        # Normalize with SigLIP constants (mean=std=0.5, so this maps [0,1] -> [-1,1])
        x = (x - self._img_mean) / self._img_std
        # Encode
        embs = self._model.get_image_features(pixel_values=x)  # (N, 768)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs

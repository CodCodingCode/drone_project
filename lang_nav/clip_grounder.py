"""CLIP-based language and vision grounder for object identification.

Loads openai/clip-vit-base-patch32 (~350MB) and encodes text commands
and RGB images into 512-dim normalized embeddings. Both encoders are
frozen — no gradients.

Usage:
    grounder = CLIPGrounder(device="cuda")
    embs = grounder.encode_texts(["go to the square", "fly to the ball"])
    # embs: Tensor(2, 512), normalized, on device

    img_embs = grounder.encode_images(rgb_tensor)
    # rgb_tensor: (N, H, W, 3) uint8, img_embs: (N, 512), normalized
"""

import torch
import torch.nn.functional as F

# CLIP image normalization constants (openai/clip-vit-base-patch32)
_CLIP_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


class CLIPGrounder:
    """Frozen CLIP text + vision encoder for language-to-object grounding.

    Encodes natural language navigation commands and camera images into
    CLIP embeddings. These embeddings are fed to the RL policy as part
    of the observation so it can match language commands to visual objects.
    """

    def __init__(self, device: str = "cuda"):
        try:
            from transformers import CLIPModel, CLIPTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for CLIPGrounder. "
                "Install with: pip install transformers"
            )

        self.device = device
        print("[CLIPGrounder] Loading openai/clip-vit-base-patch32...")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self._tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze all CLIP parameters — we only use it as a feature extractor
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

        # Pre-compute image normalization tensors on device: shape (1, 3, 1, 1)
        self._img_mean = torch.tensor(_CLIP_IMAGE_MEAN, device=device).view(1, 3, 1, 1)
        self._img_std = torch.tensor(_CLIP_IMAGE_STD, device=device).view(1, 3, 1, 1)

        print("[CLIPGrounder] Ready (text + vision).")

    @torch.no_grad()
    def encode_texts(self, commands: list[str]) -> torch.Tensor:
        """Encode a batch of text commands into normalized CLIP embeddings.

        Args:
            commands: List of natural language strings, e.g. ["go to the square"]

        Returns:
            Tensor of shape (len(commands), 512), L2-normalized, on self.device.
        """
        inputs = self._tokenizer(
            commands,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        embs = self._model.get_text_features(**inputs)  # (N, 512)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs

    @torch.no_grad()
    def encode_images(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode a batch of RGB images into normalized CLIP embeddings.

        All preprocessing runs on GPU — no CPU round-trip.

        Args:
            rgb: (N, H, W, 3) uint8 tensor from TiledCamera.

        Returns:
            Tensor of shape (N, 512), L2-normalized, on self.device.
        """
        # (N, H, W, 3) uint8 -> (N, 3, H, W) float32 [0, 1]
        x = rgb.to(dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
        # Bilinear resize to 224x224 (CLIP's expected input)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Normalize with CLIP constants
        x = (x - self._img_mean) / self._img_std
        # Encode
        embs = self._model.get_image_features(pixel_values=x)  # (N, 512)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs

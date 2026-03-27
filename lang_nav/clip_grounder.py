"""CLIP-based language grounder for object identification.

Loads openai/clip-vit-base-patch32 (~350MB) and encodes text commands
into 512-dim normalized embeddings. The encoder is frozen — no gradients.

Usage:
    grounder = CLIPGrounder(device="cuda")
    embs = grounder.encode_texts(["go to the square", "fly to the ball"])
    # embs: Tensor(2, 512), normalized, on device
"""

import torch


class CLIPGrounder:
    """Frozen CLIP text encoder for language-to-object grounding.

    Encodes natural language navigation commands into CLIP text embeddings.
    These embeddings are fed to the RL policy as part of the observation so
    the policy can generalize across different phrasings of the same command.
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
        print("[CLIPGrounder] Ready.")

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

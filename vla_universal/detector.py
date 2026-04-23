"""PaliGemma-based open-vocabulary object detector.

Uses google/paligemma-3b-pt-224 in its native "detect <class>" mode:
  Prompt:  "detect forklift ; pallet ; shelf\n"  (with <image> token sequence prepended)
  Output:  tokens <locYYYY><locXXXX><locYYYY><locXXXX> class_name ... (one quadruplet per bbox)
  Coords are normalized to [0, 1023] on a 1024×1024 grid (top-left origin).

We batch all 4 onboard camera views per inference call.

Default class list covers the 4 sample Isaac Sim scenes we care about
(warehouse / hospital / office / simple_room). Users can extend via
--extra_classes on the scan CLI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


_LOG = logging.getLogger("vla_universal.detector")

# Classes queried per call. Keep ≤ ~25 so the prompt + detection tokens fit
# in max_new_tokens=128. User can supply --extra_classes to expand.
DEFAULT_CLASSES: list[str] = [
    # warehouse
    "forklift", "pallet", "shelf", "rack", "cardboard box",
    "loading dock", "crate", "barrel",
    # hospital
    "hospital bed", "wheelchair", "medical cart", "monitor", "doorway",
    # office
    "desk", "chair", "conference table", "whiteboard", "plant", "computer",
    # universal
    "person", "ladder", "trash can", "cone", "door",
]


@dataclass
class Detection:
    cls: str
    bbox_xyxy: tuple[int, int, int, int]  # pixel coords in [0, img_size-1]
    cam_idx: int
    frame_idx: int
    conf: float = 1.0  # PaliGemma detect mode doesn't emit per-box conf

    @property
    def center(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.bbox_xyxy
        return ((x0 + x1) // 2, (y0 + y1) // 2)


# Matches: <loc####><loc####><loc####><loc####> <class name>
# PaliGemma orders coordinates as ymin, xmin, ymax, xmax normalized to [0,1023].
# Class name runs from after the 4 locs up to the next <loc...> or EOS.
_LOC_QUAD_RE = re.compile(
    r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^<\n]+?)(?=\s*<loc|\s*$)"
)


def parse_detections(
    raw_text: str,
    img_size: int = 224,
    valid_classes: Optional[set[str]] = None,
    cam_idx: int = 0,
    frame_idx: int = 0,
) -> list[Detection]:
    """Parse PaliGemma's detection output for one image.

    PaliGemma emits location tokens in [0, 1023] on a 1024-grid; we scale
    to the actual image size. If `valid_classes` is given, we only keep
    detections whose class substring-matches one of them (PaliGemma may
    return variant spellings; we match loose).
    """
    out: list[Detection] = []
    for match in _LOC_QUAD_RE.finditer(raw_text):
        ymin_raw, xmin_raw, ymax_raw, xmax_raw, cls_raw = match.groups()
        try:
            ymin = int(ymin_raw); xmin = int(xmin_raw)
            ymax = int(ymax_raw); xmax = int(xmax_raw)
        except ValueError:
            continue
        if not (0 <= ymin <= 1023 and 0 <= xmin <= 1023
                and 0 <= ymax <= 1023 and 0 <= xmax <= 1023):
            continue
        if ymax <= ymin or xmax <= xmin:
            continue

        # Rescale from [0, 1023] grid to actual pixel coords [0, img_size-1]
        s = (img_size - 1) / 1023.0
        x0 = int(round(xmin * s)); y0 = int(round(ymin * s))
        x1 = int(round(xmax * s)); y1 = int(round(ymax * s))

        cls_name = cls_raw.strip().lower()
        if valid_classes is not None:
            matched = next(
                (c for c in valid_classes
                 if c.lower() in cls_name or cls_name in c.lower()),
                None,
            )
            if matched is None:
                continue
            cls_name = matched  # normalize to canonical class name

        out.append(Detection(
            cls=cls_name,
            bbox_xyxy=(x0, y0, x1, y1),
            cam_idx=cam_idx,
            frame_idx=frame_idx,
        ))
    return out


class PaliGemmaDetector:
    """Batched detection wrapper. Loads PaliGemma once, detects on a batch
    of images per call. ~1s per forward on an H100/GH200 at batch 4."""

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-pt-224",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        print(f"[detector] loading {model_name} ...")
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="eager"
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.img_size = 224
        print("[detector] ready")

    @torch.no_grad()
    def detect_batch(
        self,
        rgbs: np.ndarray,          # (N, H, W, 3) float in [0, 1] or uint8 in [0, 255]
        classes: list[str],
        max_new_tokens: int = 128,
        cam_idx_offset: int = 0,   # cam_idx value in resulting Detection objects
        frame_idx: int = 0,
    ) -> list[list[Detection]]:
        """Detect all listed classes on N images in a single forward call.
        Returns a list (length N) of detection lists (one per image).
        """
        if rgbs.dtype != np.uint8:
            rgbs_u8 = (np.clip(rgbs, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            rgbs_u8 = rgbs

        # PaliGemma detect prompt — semicolon-separated class list
        prompt = "detect " + " ; ".join(classes) + "\n"
        prompts = [prompt] * len(rgbs_u8)

        # AutoProcessor handles <image> token prep + image resize automatically
        from PIL import Image
        pil_images = [Image.fromarray(img) for img in rgbs_u8]
        inputs = self.processor(
            text=prompts, images=pil_images, return_tensors="pt", padding="longest"
        ).to(self.device)

        # Generate — greedy decoding is sufficient for detect mode
        gen = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
        # Strip the prompt tokens from the generation
        prompt_len = inputs["input_ids"].shape[1]
        gen_only = gen[:, prompt_len:]
        texts = self.processor.batch_decode(gen_only, skip_special_tokens=False)

        valid_cls_set = set(classes)
        all_dets: list[list[Detection]] = []
        for i, text in enumerate(texts):
            dets = parse_detections(
                text, img_size=self.img_size,
                valid_classes=valid_cls_set,
                cam_idx=cam_idx_offset + i,
                frame_idx=frame_idx,
            )
            all_dets.append(dets)

        # Log malformed-frame rate for diagnostics
        empty = sum(1 for d in all_dets if not d)
        if empty == len(all_dets):
            _LOG.debug(f"frame {frame_idx}: no detections across {len(all_dets)} cams")

        return all_dets

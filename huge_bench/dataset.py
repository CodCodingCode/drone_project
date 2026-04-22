"""LeRobot-format loader for HUGE_Dataset_task0.

The HF dataset has three splits laid out as separate LeRobot trees:
    train/, test_seen/, test_unseen/
each with meta/{info,episodes,tasks}.jsonl, norm_stats.json, and
data/chunk-{NNN}/episode_{NNNNNN}.parquet (one episode per file).

We bypass the `datasets` library here because:
  - HF's auto-config would only see the top-level "train" parquet glob
    (missing test_seen/test_unseen which live in their own meta trees).
  - We want lazy per-episode parquet download, not the full 4 GB upfront.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

REPO_ID = "yu781986168/HUGE_Dataset_task0"
SPLITS = ("train", "test_seen", "test_unseen")
IMG_SIZE = 224  # matches PaliGemma-3b-pt-224


def _hf_get(path: str) -> str:
    return hf_hub_download(REPO_ID, path, repo_type="dataset")


@dataclass
class _EpisodeMeta:
    episode_index: int
    length: int
    task: str          # natural-language instruction
    env_id: str
    parquet_path: str  # repo-relative


class HugeTask0(Dataset):
    """One sample = one (episode, frame). Lazy parquet download per episode."""

    def __init__(self, split: str = "train", normalize_actions: bool = True):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self.normalize_actions = normalize_actions

        # --- metadata -----------------------------------------------------
        info = json.loads(Path(_hf_get(f"{split}/meta/info.json")).read_text())
        self.fps = info["fps"]
        self.data_path_tmpl = info["data_path"]   # "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        self.chunks_size = info["chunks_size"]    # 1000

        episodes_jsonl = Path(_hf_get(f"{split}/meta/episodes.jsonl")).read_text().strip().splitlines()
        episodes = [json.loads(line) for line in episodes_jsonl]

        tasks_jsonl = Path(_hf_get(f"{split}/meta/tasks.jsonl")).read_text().strip().splitlines()
        self.task_by_index: dict[int, str] = {}
        for line in tasks_jsonl:
            t = json.loads(line)
            self.task_by_index[int(t["task_index"])] = t["task"]

        norm = json.loads(Path(_hf_get(f"{split}/norm_stats.json")).read_text())["norm_stats"]
        self.action_mean = np.asarray(norm["actions"]["mean"], dtype=np.float32)
        self.action_std = np.asarray(norm["actions"]["std"], dtype=np.float32) + 1e-6
        self.state_mean = np.asarray(norm["state"]["mean"], dtype=np.float32)
        self.state_std = np.asarray(norm["state"]["std"], dtype=np.float32) + 1e-6

        # --- episode index ------------------------------------------------
        self.episodes: list[_EpisodeMeta] = []
        for ep in episodes:
            ep_idx = int(ep["episode_index"])
            chunk = ep_idx // self.chunks_size
            rel = self.data_path_tmpl.format(episode_chunk=chunk, episode_index=ep_idx)
            self.episodes.append(_EpisodeMeta(
                episode_index=ep_idx,
                length=int(ep["length"]),
                task=ep["tasks"][0] if ep.get("tasks") else self.task_by_index.get(0, ""),
                env_id=str(ep.get("env_id", "")),
                parquet_path=f"{split}/{rel}",
            ))

        # Flat (ep_pos, frame_pos) index — every frame is a training sample
        self._flat: list[tuple[int, int]] = []
        for ep_pos, ep in enumerate(self.episodes):
            for f in range(ep.length):
                self._flat.append((ep_pos, f))

        # In-memory parquet cache (small: max 4 episodes worth of decoded rows)
        self._cache: dict[int, pd.DataFrame] = {}
        self._cache_order: list[int] = []
        self._cache_capacity = 8

    def __len__(self) -> int:
        return len(self._flat)

    def num_episodes(self) -> int:
        return len(self.episodes)

    # ---------------------------------------------------------------------
    # Episode loading (lazy, LRU-cached)
    # ---------------------------------------------------------------------
    def _load_episode(self, ep_pos: int) -> pd.DataFrame:
        if ep_pos in self._cache:
            return self._cache[ep_pos]
        ep = self.episodes[ep_pos]
        local = hf_hub_download(REPO_ID, ep.parquet_path, repo_type="dataset")
        df = pd.read_parquet(local)
        self._cache[ep_pos] = df
        self._cache_order.append(ep_pos)
        if len(self._cache_order) > self._cache_capacity:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return df

    @staticmethod
    def _decode_image(cell) -> np.ndarray:
        # cell is dict {'bytes': png-bytes, 'path': str}
        b = cell["bytes"] if isinstance(cell, dict) else cell
        img = Image.open(io.BytesIO(b)).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        # SigLIP normalization: [0,255] -> [-1,1]
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        return arr  # (H, W, 3)

    def _normalize_state(self, s: np.ndarray) -> np.ndarray:
        return (s - self.state_mean) / self.state_std

    def _normalize_action(self, a: np.ndarray) -> np.ndarray:
        if not self.normalize_actions:
            return a
        return (a - self.action_mean) / self.action_std

    def denormalize_action(self, a: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(a, torch.Tensor):
            mean = torch.as_tensor(self.action_mean, device=a.device, dtype=a.dtype)
            std = torch.as_tensor(self.action_std, device=a.device, dtype=a.dtype)
            return a * std + mean
        return a * self.action_std + self.action_mean

    # ---------------------------------------------------------------------
    # Sample interface
    # ---------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        ep_pos, frame_pos = self._flat[idx]
        ep = self.episodes[ep_pos]
        df = self._load_episode(ep_pos)
        row = df.iloc[frame_pos]

        # Pre-normalized RGB float32 (H, W, 3) in [-1, 1]
        image = self._decode_image(row["image"])
        first_image = self._decode_image(row["first_image"])

        state = np.asarray(row["state"], dtype=np.float32).copy()
        action = np.asarray(row["actions"], dtype=np.float32).copy()

        return {
            "image": torch.from_numpy(image),                        # (224, 224, 3) float32 [-1,1]
            "first_image": torch.from_numpy(first_image),            # (224, 224, 3) float32 [-1,1]
            "state": torch.from_numpy(self._normalize_state(state)), # (4,) float32
            "action": torch.from_numpy(self._normalize_action(action)),  # (4,) float32
            "raw_action": torch.from_numpy(action),                  # (4,) float32 (un-normalized)
            "instruction": ep.task,                                  # str
            "task_index": int(row["task_index"]),
            "env_id": ep.env_id,
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
        }


# -------------------------------------------------------------------------
# Collation
# -------------------------------------------------------------------------
def collate_bc(batch: list[dict]) -> dict:
    """Stack tensors and keep instruction strings as a list."""
    out: dict = {}
    tensor_keys = ("image", "first_image", "state", "action", "raw_action")
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["instruction"] = [b["instruction"] for b in batch]
    out["task_index"] = torch.tensor([b["task_index"] for b in batch], dtype=torch.long)
    out["episode_index"] = torch.tensor([b["episode_index"] for b in batch], dtype=torch.long)
    out["frame_index"] = torch.tensor([b["frame_index"] for b in batch], dtype=torch.long)
    out["env_id"] = [b["env_id"] for b in batch]
    return out

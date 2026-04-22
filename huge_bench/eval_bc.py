"""Evaluate a BC checkpoint on a held-out split.

Reports action MSE overall and broken down by env_id and task_index.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from huge_bench.dataset import HugeTask0, collate_bc
from huge_bench.policy import HugeBCPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test_seen", choices=["train", "test_seen", "test_unseen"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=-1)  # -1 = full split
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    model = HugeBCPolicy(
        max_text_length=cfg.get("max_text_length", 64),
        lora_rank=cfg.get("lora_rank", 8),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = HugeTask0(split=args.split)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_bc, drop_last=False,
    )

    per_dim_sq = np.zeros(4, dtype=np.float64)  # sum of squared errors per dim (normalized space)
    raw_per_dim_sq = np.zeros(4, dtype=np.float64)
    n = 0
    env_stats: dict[str, list[float]] = defaultdict(list)
    task_stats: dict[int, list[float]] = defaultdict(list)

    for i, batch in enumerate(loader):
        if args.max_batches >= 0 and i >= args.max_batches:
            break
        pred = model(batch, with_grad_through_lora=False)  # (B, 4) in normalized space
        target = batch["action"].to(device)                # normalized
        raw_target = batch["raw_action"].to(device)        # un-normalized
        raw_pred = ds.denormalize_action(pred)             # un-normalized prediction

        sq = ((pred - target) ** 2).cpu().numpy()          # (B, 4)
        raw_sq = ((raw_pred - raw_target) ** 2).cpu().numpy()

        per_dim_sq += sq.sum(axis=0)
        raw_per_dim_sq += raw_sq.sum(axis=0)
        n += sq.shape[0]

        for j, env_id in enumerate(batch["env_id"]):
            env_stats[env_id].append(float(sq[j].mean()))
        for j, t_idx in enumerate(batch["task_index"].tolist()):
            task_stats[int(t_idx)].append(float(sq[j].mean()))

        if (i + 1) % 20 == 0:
            print(f"  batch {i + 1}: running total {n} samples")

    mse_per_dim = (per_dim_sq / max(1, n)).tolist()
    raw_mse_per_dim = (raw_per_dim_sq / max(1, n)).tolist()
    overall = float(np.mean(mse_per_dim))
    raw_overall = float(np.mean(raw_mse_per_dim))

    print(f"\n=== Results ({args.split}, {n} samples) ===")
    print("Normalized action MSE:")
    print(f"  overall: {overall:.5f}")
    for name, v in zip(["dx", "dy", "dz", "dyaw"], mse_per_dim):
        print(f"    {name}: {v:.5f}")
    print("Raw (un-normalized) action MSE:")
    print(f"  overall: {raw_overall:.5f}")
    for name, v in zip(["dx", "dy", "dz", "dyaw"], raw_mse_per_dim):
        print(f"    {name}: {v:.5f}")

    print("\nPer env_id mean MSE (normalized):")
    env_summary = {e: float(np.mean(v)) for e, v in sorted(env_stats.items())}
    for e, v in env_summary.items():
        print(f"  {e}: {v:.5f}  (n={len(env_stats[e])})")

    print("\nTop-5 worst task_index by mean MSE:")
    task_mean = {t: float(np.mean(v)) for t, v in task_stats.items()}
    for t, v in sorted(task_mean.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  task {t}: mse={v:.5f} n={len(task_stats[t])}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "split": args.split,
            "n_samples": n,
            "mse_normalized": overall,
            "mse_per_dim_normalized": mse_per_dim,
            "mse_raw": raw_overall,
            "mse_per_dim_raw": raw_mse_per_dim,
            "mse_per_env_normalized": env_summary,
        }, indent=2))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()

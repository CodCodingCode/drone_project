"""One-shot probe of the HUGE_Dataset_task0 HuggingFace dataset.

Run first. Prints schema, one row per split, and lists repo files to
look for a sidecar instruction mapping (meta.json / tasks.json / etc.).
"""

from __future__ import annotations

import argparse
import json

REPO_ID = "yu781986168/HUGE_Dataset_task0"


def probe_repo_files():
    from huggingface_hub import list_repo_files

    print(f"\n=== Files in {REPO_ID} ===")
    files = list_repo_files(REPO_ID, repo_type="dataset")
    for f in files:
        print(f"  {f}")
    return files


def probe_dataset():
    from datasets import load_dataset

    print(f"\n=== Loading {REPO_ID} ===")
    ds = load_dataset(REPO_ID)

    for split in ds:
        print(f"\n--- split: {split} ---")
        print(f"  num_rows: {len(ds[split])}")
        print(f"  features: {ds[split].features}")
        row = ds[split][0]
        print(f"  row keys: {list(row.keys())}")
        for k, v in row.items():
            if hasattr(v, "size") and hasattr(v, "mode"):  # PIL
                print(f"    {k}: PIL {v.mode} {v.size}")
            elif isinstance(v, list):
                print(f"    {k}: list len={len(v)} sample={v[:4]}")
            else:
                print(f"    {k}: {type(v).__name__} = {v}")

        # Scan a slice of rows to collect unique env_ids and task_index range
        sample = ds[split].select(range(min(500, len(ds[split]))))
        env_ids = set()
        task_indices = set()
        for r in sample:
            env_ids.add(r["env_id"])
            task_indices.add(r["task_index"])
        print(f"  env_ids in first 500: {sorted(env_ids)}")
        print(f"  task_index range in first 500: min={min(task_indices)} max={max(task_indices)} count={len(task_indices)}")

    return ds


def probe_global_stats(ds):
    print("\n=== Global stats across full train split ===")
    split = ds["train"]
    env_ids = set()
    task_indices = set()
    episodes = set()
    n = len(split)
    step = max(1, n // 5000)
    for i in range(0, n, step):
        r = split[i]
        env_ids.add(r["env_id"])
        task_indices.add(r["task_index"])
        episodes.add(r["episode_index"])
    print(f"  env_ids (sampled): {sorted(env_ids)}")
    print(f"  task_indices count (sampled): {len(task_indices)} range=[{min(task_indices)}, {max(task_indices)}]")
    print(f"  episodes count (sampled): {len(episodes)} range=[{min(episodes)}, {max(episodes)}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_download", action="store_true",
                        help="Only list repo files, don't load dataset")
    args = parser.parse_args()

    files = probe_repo_files()

    # Highlight any non-parquet sidecar metadata
    suspicious = [f for f in files if not f.endswith(".parquet")
                  and not f.startswith(".")
                  and f not in ("README.md",)]
    if suspicious:
        print(f"\n=== Non-parquet files (potential instruction sidecars) ===")
        for f in suspicious:
            print(f"  {f}")

    if args.skip_download:
        return

    ds = probe_dataset()
    probe_global_stats(ds)


if __name__ == "__main__":
    main()

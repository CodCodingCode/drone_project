# huge_bench — offline BC on HUGE-Bench task0

Trains a PaliGemma-3B (frozen + LoRA) + MLP behavior-cloning policy on the
[`yu781986168/HUGE_Dataset_task0`](https://huggingface.co/datasets/yu781986168/HUGE_Dataset_task0)
LeRobot-format dataset.

## What the dataset looks like
- 3 splits on disk: `train/` (1733 episodes, 171k frames), `test_seen/`, `test_unseen/`.
- Per-frame fields: `image` and `first_image` (256×256 PNG), `state` = `(x, y, z, yaw_rad)`,
  `actions` = per-step deltas `(dx, dy, dz, dyaw)` at 5 Hz.
- Natural-language instructions live in `{split}/meta/tasks.jsonl`, mapped via `task_index`.
  Example: `"Fly to 60 meters above the twin curved office building complex."`
- Normalization stats in `{split}/norm_stats.json` (mean/std/q01/q99 per dim).

## Pipeline
1. `probe_dataset.py` — one-shot schema / language check (run first if you're exploring).
2. `dataset.py` — LeRobot loader (lazy per-episode parquet download, LRU cache).
3. `policy.py` — `HugeBCPolicy`: two PaliGemma forwards (`first_image`, `image`) with the
   same tokenized instruction, concatenated with the 4-dim normalized state, fed to a
   `4100 → 512 → 512 → 4` head.
4. `train_bc.py` — offline MSE-on-actions training loop with two AdamW optimizers
   (head at `3e-4`, LoRA at `1e-6`) and gradient accumulation.
5. `eval_bc.py` — reports action MSE on a split, broken down by `env_id` and
   `task_index`.

## Install deps (one-time, into the `isaac` conda env)
```
source /home/ubuntu/miniconda3/bin/activate isaac
pip install datasets pyarrow
```
(`transformers`, `torch`, `huggingface_hub`, `pandas`, `PIL` are already present.)

## Smoke test
Already verified on 2026-04-22:
- **Dataset path**: loader returns 171,252 train frames across 1733 episodes,
  SigLIP-normalized images, normalized state/action, correct language strings.
- **Tokenization path**: 288-token sequence (256 image + 32 text) with right-
  padding so `attention_mask.sum()-1` indexes the actual last text token.
- **Head path**: 4100→512→512→4 MLP accepts the concatenated features and
  backpropagates through MSE loss.
- **PaliGemma forward on GPU**: *not run in-session* because the GPU was fully
  occupied by another user's vLLM engine. Run the quick GPU smoke below once
  the GPU is free.

```bash
source /home/ubuntu/miniconda3/bin/activate isaac
cd /home/ubuntu/drone_project

# (1) schema / repo check
python -m huge_bench.probe_dataset --skip_download

# (2) tiny GPU forward + backward (one batch, batch_size=2)
python - <<'PY'
import torch
from huge_bench.dataset import HugeTask0, collate_bc
from huge_bench.policy import HugeBCPolicy
from torch.utils.data import DataLoader

ds = HugeTask0(split="train")
loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate_bc)
batch = next(iter(loader))
m = HugeBCPolicy(max_text_length=32, lora_rank=8).cuda()
for p in m.head.parameters(): p.requires_grad = True
pred = m(batch, with_grad_through_lora=True)
loss = ((pred - batch["action"].cuda()) ** 2).mean()
loss.backward()
print("pred shape:", pred.shape, "loss:", float(loss))
print("GPU MB:", torch.cuda.memory_allocated() // 1024**2)
PY
```

Then a short training run (~10 steps, still tiny):
```bash
python -m huge_bench.train_bc \
    --batch_size 2 --grad_accum 1 --max_steps 10 \
    --val_every 5 --val_batches 2 --save_every 10 --num_workers 0
```

## Full training
```
bash /home/ubuntu/drone_project/huge_bench/run_train_bc.sh
# Overrides via env vars: BATCH_SIZE=32 MAX_STEPS=40000 bash run_train_bc.sh
```
Logs go to `drone_project/logs/huge_bench/<timestamp>/` (TensorBoard events + per-step
`args.json` + checkpoints `model_<step>.pt`).

## Evaluation
```
python -m huge_bench.eval_bc \
    --checkpoint /home/ubuntu/drone_project/logs/huge_bench/<run>/model_20000.pt \
    --split test_seen --out_json /tmp/test_seen.json
python -m huge_bench.eval_bc \
    --checkpoint <path> --split test_unseen --out_json /tmp/test_unseen.json
```

## What this does NOT do
- **No Isaac Sim rollouts.** The HUGE-Bench Isaac Sim scenes are not publicly released;
  this is pure offline BC on trajectories.
- **No sim-to-real on hardware.** The paper itself doesn't do hardware transfer either.
- **Task0 only.** If the authors publish other tasks under the same org, swap `REPO_ID`
  in `dataset.py`.

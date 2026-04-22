"""Offline behavior cloning on HUGE_Dataset_task0.

Run from anywhere; does not need IsaacLab.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from huge_bench.dataset import HugeTask0, collate_bc
from huge_bench.policy import HugeBCPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--head_lr", type=float, default=3e-4)
    p.add_argument("--lora_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_split", type=str, default="test_seen", choices=["test_seen", "test_unseen"])
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_dir", type=str, default="/home/ubuntu/drone_project/logs/huge_bench")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max_text_length", type=int, default=64)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data ----------------------------------------------------------
    print("[BC] Building train dataset...")
    train_ds = HugeTask0(split="train")
    print(f"[BC] train: {len(train_ds)} frames across {train_ds.num_episodes()} episodes")

    print(f"[BC] Building val dataset ({args.val_split})...")
    val_ds = HugeTask0(split=args.val_split)
    print(f"[BC] val: {len(val_ds)} frames across {val_ds.num_episodes()} episodes")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_bc,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_bc,
        pin_memory=True, drop_last=False, persistent_workers=args.num_workers > 0,
    )

    # --- model ---------------------------------------------------------
    print("[BC] Building policy...")
    model = HugeBCPolicy(max_text_length=args.max_text_length, lora_rank=args.lora_rank)
    model = model.to(device)
    # Head needs gradients, LoRA already gradient-enabled inside extractor.
    for p in model.head.parameters():
        p.requires_grad = True

    head_params = list(model.head_parameters())
    lora_params = list(model.lora_parameters())
    n_head = sum(p.numel() for p in head_params)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"[BC] head trainable: {n_head:,}  | lora trainable: {n_lora:,}")

    head_opt = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
    lora_opt = torch.optim.AdamW(lora_params, lr=args.lora_lr, weight_decay=0.0)

    # --- logging -------------------------------------------------------
    run_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[BC] Logging to {run_dir}")

    # --- resume --------------------------------------------------------
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        head_opt.load_state_dict(ckpt["head_opt_state_dict"])
        lora_opt.load_state_dict(ckpt["lora_opt_state_dict"])
        start_step = int(ckpt.get("step", 0))
        print(f"[BC] Resumed from {args.resume} at step {start_step}")

    # --- train loop ----------------------------------------------------
    model.train()
    loss_fn = torch.nn.MSELoss()
    step = start_step
    accum = 0
    head_opt.zero_grad(set_to_none=True)
    lora_opt.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_n = 0

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        target = batch["action"].to(device)
        pred = model(batch, with_grad_through_lora=True)
        loss = loss_fn(pred, target) / args.grad_accum
        loss.backward()
        accum += 1

        running_loss += float(loss.item()) * args.grad_accum
        running_n += 1

        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(head_params + lora_params, max_norm=1.0)
            head_opt.step()
            lora_opt.step()
            head_opt.zero_grad(set_to_none=True)
            lora_opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            if step % 10 == 0:
                avg = running_loss / max(1, running_n)
                rate = (step - start_step) / max(1e-6, time.time() - t0)
                writer.add_scalar("train/action_mse", avg, step)
                writer.add_scalar("train/steps_per_sec", rate, step)
                print(f"[BC] step {step:6d} | mse={avg:.4f} | {rate:.2f} it/s")
                running_loss = 0.0
                running_n = 0

            if step % args.val_every == 0:
                model.eval()
                val_mse = _validate(model, val_loader, device, args.val_batches, loss_fn)
                writer.add_scalar(f"val/{args.val_split}_action_mse", val_mse, step)
                print(f"[BC] step {step:6d} | val_mse({args.val_split})={val_mse:.4f}")
                model.train()

            if step % args.save_every == 0 or step == args.max_steps:
                ckpt_path = run_dir / f"model_{step}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "head_opt_state_dict": head_opt.state_dict(),
                    "lora_opt_state_dict": lora_opt.state_dict(),
                    "step": step,
                    "args": vars(args),
                }, ckpt_path)
                print(f"[BC] saved {ckpt_path}")

    writer.close()


@torch.no_grad()
def _validate(model, loader, device, max_batches, loss_fn) -> float:
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        pred = model(batch, with_grad_through_lora=False)
        target = batch["action"].to(device)
        total += float(loss_fn(pred, target).item())
        n += 1
    return total / max(1, n)


if __name__ == "__main__":
    main()

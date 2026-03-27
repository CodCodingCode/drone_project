"""Transfer hover checkpoint to waypoint nav policy.

Since hover and waypoint_nav have identical architectures (256x256 MLP,
15-dim obs, 4-dim action), this is a direct copy with optimizer/iteration
reset so training restarts fresh while keeping learned weights.

Usage:
    conda activate isaac
    cd /home/ubuntu/drone_project
    python transfer_hover_to_waypoint.py \
        --hover_checkpoint logs/rsl_rl/hover_pretrain/<timestamp>/model_299.pt \
        --output_path logs/rsl_rl/waypoint_nav/pretrained_init.pt
"""

import argparse
import os

import torch


def transfer(hover_path: str, output_path: str):
    hover_ckpt = torch.load(hover_path, map_location="cpu", weights_only=False)

    output_ckpt = {
        "actor_state_dict": hover_ckpt["actor_state_dict"],
        "critic_state_dict": hover_ckpt["critic_state_dict"],
        "optimizer_state_dict": {},  # reset optimizer momentum
        "iter": 0,
        "infos": None,
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save(output_ckpt, output_path)
    print(f"[INFO] Saved transferred checkpoint to: {output_path}")
    print(f"       Architecture: 256x256 MLP, 15-dim obs (identical, direct copy)")
    print(f"       Optimizer and iteration counter reset to 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer hover weights to waypoint nav policy.")
    parser.add_argument("--hover_checkpoint", type=str, required=True, help="Path to hover model .pt file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="logs/rsl_rl/waypoint_nav/pretrained_init.pt",
        help="Where to save the transferred checkpoint",
    )
    args = parser.parse_args()
    transfer(args.hover_checkpoint, args.output_path)

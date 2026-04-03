"""Transfer waypoint nav action head weights into VLA architecture.

Maps the waypoint MLP (256×15 input) into the VLA action head (256×2057 input).
Flight state dims (first 9) are placed at the end of the fused vector (after
PaliGemma's 2048-dim features). Hidden and output layers transfer directly.

Usage:
    cd ~/drone_project
    python vla/transfer_waypoint_to_vla.py \
        --waypoint_checkpoint logs/rsl_rl/waypoint_nav/.../model_2998.pt \
        --output_path logs/rsl_rl/vla_drone_direct/vla_init.pt
"""

import argparse
import math
import os

import torch

WAYPOINT_OBS_DIM = 15
PALIGEMMA_FEAT_DIM = 2048
FLIGHT_STATE_DIM = 9
VLA_INPUT_DIM = PALIGEMMA_FEAT_DIM + FLIGHT_STATE_DIM  # 2057
HIDDEN_DIM = 256


def transfer(waypoint_path: str, output_path: str):
    ckpt = torch.load(waypoint_path, map_location="cpu", weights_only=False)

    new_actor = {}
    for key, param in ckpt["actor_state_dict"].items():
        if key == "mlp.0.weight":
            # Waypoint: (256, 15) → VLA: (256, 2057)
            # Flight state at positions [2048:2057], PaliGemma features at [0:2048]
            new = torch.zeros(HIDDEN_DIM, VLA_INPUT_DIM)
            fan_in = VLA_INPUT_DIM
            bound = math.sqrt(1.0 / fan_in)
            torch.nn.init.uniform_(new, -bound, bound)
            # Copy flight state columns (first 9 of waypoint → last 9 of VLA)
            new[:, PALIGEMMA_FEAT_DIM:PALIGEMMA_FEAT_DIM + FLIGHT_STATE_DIM] = param[:, :FLIGHT_STATE_DIM]
            new_actor[key] = new
        elif key == "mlp.0.bias":
            new_actor[key] = param.clone()
        elif key.startswith("mlp."):
            # Hidden and output layers: direct copy
            new_actor[key] = param.clone()
        elif key == "distribution.std_param" or key == "_std_param":
            new_actor[key] = param.clone()
        elif "obs_normalizer" in key or "_obs_mean" in key or "_obs_var" in key:
            # Reshape normalizer for 9-dim flight state
            if param.shape[-1] == WAYPOINT_OBS_DIM:
                new_actor[key] = param[:, :FLIGHT_STATE_DIM] if param.dim() > 1 else param[:FLIGHT_STATE_DIM]
            else:
                new_actor[key] = param.clone()
        elif "_obs_count" in key:
            new_actor[key] = torch.tensor(100.0)
        else:
            new_actor[key] = param.clone()

    # Critic gets same treatment (separate value head)
    new_critic = {}
    for key, param in ckpt.get("critic_state_dict", {}).items():
        if key == "mlp.0.weight":
            new = torch.zeros(HIDDEN_DIM, VLA_INPUT_DIM)
            bound = math.sqrt(1.0 / VLA_INPUT_DIM)
            torch.nn.init.uniform_(new, -bound, bound)
            new[:, PALIGEMMA_FEAT_DIM:PALIGEMMA_FEAT_DIM + FLIGHT_STATE_DIM] = param[:, :FLIGHT_STATE_DIM]
            new_critic[key] = new
        elif key == "mlp.0.bias":
            new_critic[key] = param.clone()
        elif key.startswith("mlp."):
            # Output layer changes from 4-dim (action) to 1-dim (value)
            if key == "mlp.4.weight" and param.shape[0] == 4:
                # Waypoint actor output → critic needs 1-dim output
                new_critic[key] = torch.randn(1, HIDDEN_DIM) * 0.01
            elif key == "mlp.4.bias" and param.shape[0] == 4:
                new_critic[key] = torch.zeros(1)
            else:
                new_critic[key] = param.clone()
        elif "obs_normalizer" in key or "_obs_mean" in key or "_obs_var" in key:
            if param.shape[-1] == WAYPOINT_OBS_DIM:
                new_critic[key] = param[:, :FLIGHT_STATE_DIM] if param.dim() > 1 else param[:FLIGHT_STATE_DIM]
            else:
                new_critic[key] = param.clone()
        elif "_obs_count" in key:
            new_critic[key] = torch.tensor(100.0)
        else:
            new_critic[key] = param.clone()

    output_ckpt = {
        "actor_state_dict": new_actor,
        "critic_state_dict": new_critic,
        "optimizer_state_dict": {},
        "iter": 0,
        "infos": None,
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save(output_ckpt, output_path)
    print(f"[INFO] Saved VLA checkpoint to: {output_path}")
    print(f"       Action head input: ({HIDDEN_DIM}, {WAYPOINT_OBS_DIM}) → ({HIDDEN_DIM}, {VLA_INPUT_DIM})")
    print(f"       Flight state dims [0:{FLIGHT_STATE_DIM}] → VLA dims [{PALIGEMMA_FEAT_DIM}:{PALIGEMMA_FEAT_DIM + FLIGHT_STATE_DIM}]")
    print(f"       PaliGemma feature dims [0:{PALIGEMMA_FEAT_DIM}]: Kaiming-initialized")
    print(f"       Hidden layers: direct copy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer waypoint nav weights to VLA action head.")
    parser.add_argument("--waypoint_checkpoint", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="logs/rsl_rl/vla_drone_direct/vla_init.pt")
    args = parser.parse_args()
    transfer(args.waypoint_checkpoint, args.output_path)

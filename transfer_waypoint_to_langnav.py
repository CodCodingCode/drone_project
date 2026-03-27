"""Transfer learned waypoint nav weights into the lang_nav policy.

Both policies use 256x256 hidden layers, so only the input layer needs
resizing (15-dim obs -> 533-dim obs). All hidden and output layers
transfer directly.

The first 12 obs dims are shared (lin_vel, ang_vel, projected_gravity,
target_pos_b). New input columns for CLIP embeddings and object positions
are initialized to small random values (Kaiming uniform) to break symmetry.

Usage:
    conda activate isaac
    cd /home/ubuntu/drone_project
    python transfer_waypoint_to_langnav.py \
        --waypoint_checkpoint logs/rsl_rl/waypoint_nav/<timestamp>/model_399.pt \
        --output_path logs/rsl_rl/lang_drone_direct/pretrained_init.pt
"""

import argparse
import math
import os

import torch

WAYPOINT_OBS_DIM = 15
LANGNAV_OBS_DIM = 533
HIDDEN_DIM = 256
SHARED_OBS_DIMS = 12


def _expand_input_weight(param, new_cols, shared_cols):
    """Expand input weight matrix from (256, 15) to (256, 533).

    Shared columns are copied. New columns get Kaiming uniform init
    to break symmetry (better than zeros for learning).
    """
    rows = param.shape[0]
    new = torch.zeros(rows, new_cols)
    # Kaiming uniform for new columns
    fan_in = new_cols
    bound = math.sqrt(1.0 / fan_in)
    torch.nn.init.uniform_(new, -bound, bound)
    # Overwrite shared columns with trained weights
    new[:, :shared_cols] = param[:, :shared_cols]
    return new


def _expand_obs_normalizer(param, new_dim, shared_dims, fill_value):
    """Expand obs normalizer stat from (1, 15) to (1, 533)."""
    new = torch.full((1, new_dim), fill_value)
    new[:, :shared_dims] = param[:, :shared_dims]
    return new


def transfer(waypoint_path: str, output_path: str):
    ckpt = torch.load(waypoint_path, map_location="cpu", weights_only=False)

    new_actor = {}
    new_critic = {}

    for src_state, new_state in [
        (ckpt["actor_state_dict"], new_actor),
        (ckpt["critic_state_dict"], new_critic),
    ]:
        for key, param in src_state.items():
            if key == "mlp.0.weight":
                # Input layer: (256, 15) -> (256, 533)
                new_state[key] = _expand_input_weight(param, LANGNAV_OBS_DIM, SHARED_OBS_DIMS)

            elif key == "obs_normalizer._mean":
                new_state[key] = _expand_obs_normalizer(param, LANGNAV_OBS_DIM, SHARED_OBS_DIMS, 0.0)

            elif key in ("obs_normalizer._var", "obs_normalizer._std"):
                new_state[key] = _expand_obs_normalizer(param, LANGNAV_OBS_DIM, SHARED_OBS_DIMS, 1.0)

            else:
                # Everything else transfers directly (hidden layers, output, bias, etc.)
                new_state[key] = param.clone()

    output_ckpt = {
        "actor_state_dict": new_actor,
        "critic_state_dict": new_critic,
        "optimizer_state_dict": {},
        "iter": 0,
        "infos": None,
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save(output_ckpt, output_path)
    print(f"[INFO] Saved transferred checkpoint to: {output_path}")
    print(f"       Input layer: ({HIDDEN_DIM}, {WAYPOINT_OBS_DIM}) -> ({HIDDEN_DIM}, {LANGNAV_OBS_DIM})")
    print(f"       Shared obs dims 0-{SHARED_OBS_DIMS - 1} copied, new dims Kaiming-initialized")
    print(f"       Hidden + output layers: direct copy (both 256x256)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer waypoint nav weights to lang_nav policy.")
    parser.add_argument("--waypoint_checkpoint", type=str, required=True, help="Path to waypoint model .pt file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="logs/rsl_rl/lang_drone_direct/pretrained_init.pt",
        help="Where to save the transferred checkpoint",
    )
    args = parser.parse_args()
    transfer(args.waypoint_checkpoint, args.output_path)

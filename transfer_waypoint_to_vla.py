"""Transfer learned waypoint nav weights into the VLA policy.

Both policies use 256x256 hidden layers, so only the input layer needs
resizing (15-dim obs -> 1033-dim obs). All hidden and output layers
transfer directly.

Only the first 9 obs dims are shared (lin_vel, ang_vel, projected_gravity).
The waypoint's target_pos_b (dims 9-12) is dropped since the VLA doesn't
use privileged position info. New input columns for CLIP text/image
embeddings are Kaiming-initialized.

Usage:
    conda activate isaac
    cd /home/ubuntu/drone_project
    python transfer_waypoint_to_vla.py \
        --waypoint_checkpoint logs/rsl_rl/waypoint_nav/<timestamp>/model_<iter>.pt \
        --output_path logs/rsl_rl/lang_drone_direct/vla_init.pt
"""

import argparse
import os

import torch

WAYPOINT_OBS_DIM = 15
VLA_OBS_DIM = 1033
HIDDEN_DIM = 256
SHARED_OBS_DIMS = 9  # lin_vel(3) + ang_vel(3) + projected_gravity(3)


def _expand_input_weight(param, new_cols, shared_cols):
    """Expand input weight matrix from (256, 15) to (256, 1033).

    First 9 columns are copied (flight state). CLIP columns are
    zero-initialized so the first hidden layer initially ignores them,
    preserving the pretrained hover/nav behaviour on day one.
    """
    rows = param.shape[0]
    new = torch.zeros(rows, new_cols)
    # Copy shared flight state columns
    new[:, :shared_cols] = param[:, :shared_cols]
    return new


def _expand_obs_normalizer(param, new_dim, shared_dims, fill_value):
    """Expand obs normalizer stat from (1, 15) to (1, 1033)."""
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
                new_state[key] = _expand_input_weight(param, VLA_OBS_DIM, SHARED_OBS_DIMS)

            elif key == "obs_normalizer._mean":
                new_state[key] = _expand_obs_normalizer(param, VLA_OBS_DIM, SHARED_OBS_DIMS, 0.0)

            elif key in ("obs_normalizer._var", "obs_normalizer._std"):
                new_state[key] = _expand_obs_normalizer(param, VLA_OBS_DIM, SHARED_OBS_DIMS, 1.0)

            elif key == "obs_normalizer._count":
                # Reset count so the normalizer adapts quickly to CLIP
                # embedding statistics (flight dims re-converge fast since
                # the physics are identical).
                new_state[key] = torch.tensor(100.0)

            else:
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
    print(f"[INFO] Saved VLA checkpoint to: {output_path}")
    print(f"       Input layer: ({HIDDEN_DIM}, {WAYPOINT_OBS_DIM}) -> ({HIDDEN_DIM}, {VLA_OBS_DIM})")
    print(f"       Flight state dims 0-{SHARED_OBS_DIMS - 1} copied, all else Kaiming-initialized")
    print(f"       Hidden + output layers: direct copy (both 256x256)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer waypoint nav weights to VLA policy.")
    parser.add_argument("--waypoint_checkpoint", type=str, required=True, help="Path to waypoint model .pt file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="logs/rsl_rl/lang_drone_direct/vla_init.pt",
        help="Where to save the transferred checkpoint",
    )
    args = parser.parse_args()
    transfer(args.waypoint_checkpoint, args.output_path)

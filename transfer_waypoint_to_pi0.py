"""Transfer learned waypoint nav weights into the Pi0 policy.

Both policies use 256x256 hidden layers, so only the input layer needs
resizing (15-dim obs -> 2057-dim obs). All hidden and output layers
transfer directly.

Only the first 9 obs dims are shared (lin_vel, ang_vel, projected_gravity).
The waypoint's target_pos_b (dims 9-12) is dropped since the Pi0 policy
uses vision features instead of privileged position info. New input columns
for Pi0 features are zero-initialized so the MLP initially ignores them,
preserving pretrained hover/nav behavior.

Usage:
    conda activate isaac
    cd /home/ubuntu/drone_project
    python transfer_waypoint_to_pi0.py \
        --waypoint_checkpoint model_2998_waypoint.pt \
        --output_path logs/rsl_rl/pi0_drone_direct/pi0_init.pt
"""

import argparse
import os

import torch

WAYPOINT_OBS_DIM = 15
PI0_OBS_DIM = 2057  # Pi0 features 2048 + flight state 9
HIDDEN_DIM = 256
SHARED_OBS_DIMS = 9  # lin_vel(3) + ang_vel(3) + projected_gravity(3)


def _expand_input_weight(param, new_cols, shared_cols):
    """Expand input weight matrix from (256, 15) to (256, 2057).

    The Pi0 model concatenates as [pi0_features_2048 | flight_state_9],
    so flight state columns go at the END (cols 2048:2057), not the start.
    Pi0 feature columns (0:2048) are zero-initialized so the first
    hidden layer initially ignores them, preserving pretrained hover/nav.
    """
    rows = param.shape[0]
    pi0_dim = new_cols - shared_cols  # 2048
    new = torch.zeros(rows, new_cols)
    # Copy shared flight state columns to the END (after Pi0 features)
    new[:, pi0_dim:pi0_dim + shared_cols] = param[:, :shared_cols]
    return new


def _expand_obs_normalizer(param, new_dim, shared_dims, fill_value):
    """Expand obs normalizer stat from (1, 15) to (1, 2057).

    Flight state dims go at the END to match [pi0_features | flight_state] layout.
    """
    pi0_dim = new_dim - shared_dims  # 2048
    new = torch.full((1, new_dim), fill_value)
    new[:, pi0_dim:pi0_dim + shared_dims] = param[:, :shared_dims]
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
                new_state[key] = _expand_input_weight(param, PI0_OBS_DIM, SHARED_OBS_DIMS)

            elif key == "obs_normalizer._mean":
                new_state[key] = _expand_obs_normalizer(param, PI0_OBS_DIM, SHARED_OBS_DIMS, 0.0)

            elif key in ("obs_normalizer._var", "obs_normalizer._std"):
                new_state[key] = _expand_obs_normalizer(param, PI0_OBS_DIM, SHARED_OBS_DIMS, 1.0)

            elif key == "obs_normalizer._count":
                # Reset count so the normalizer adapts quickly to Pi0
                # feature statistics (flight dims re-converge fast since
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
    print(f"[INFO] Saved Pi0 checkpoint to: {output_path}")
    print(f"       Input layer: ({HIDDEN_DIM}, {WAYPOINT_OBS_DIM}) -> ({HIDDEN_DIM}, {PI0_OBS_DIM})")
    print(f"       Flight state dims 0-{SHARED_OBS_DIMS - 1} copied to positions 2048-2056")
    print(f"       Pi0 feature columns (0-2047): zero-initialized")
    print(f"       Hidden + output layers: direct copy (both 256x256)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer waypoint nav weights to Pi0 policy.")
    parser.add_argument("--waypoint_checkpoint", type=str, required=True, help="Path to waypoint model .pt file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="logs/rsl_rl/pi0_drone_direct/pi0_init.pt",
        help="Where to save the transferred checkpoint",
    )
    args = parser.parse_args()
    transfer(args.waypoint_checkpoint, args.output_path)

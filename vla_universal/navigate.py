"""Navigate entry point.

Loads a semantic_map.json for the requested scene, matches the user prompt
against its POIs, picks a target, and flies the drone there using the frozen
waypoint policy. Records video to `videos/vla_universal/navigate_<ts>.mp4`.

Launch via run_navigate.sh.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Navigate to a prompt target.")
parser.add_argument("--scene", type=str, default="warehouse_full")
parser.add_argument("--prompt", type=str, required=True,
                    help="Natural-language command, e.g. 'fly to the forklift'")
parser.add_argument("--map", type=str, default=None,
                    help="Path to semantic_map.json (default logs/maps/<scene>.json)")
parser.add_argument("--pick", choices=["closest", "highest_conf", "index", "ask"],
                    default="closest",
                    help="Disambiguation policy when prompt matches multiple POIs")
parser.add_argument("--index", type=int, default=0,
                    help="When --pick=index, which candidate (sorted by id)")
parser.add_argument("--target_id", type=str, default=None,
                    help="Explicit POI id override (skips pick logic)")
parser.add_argument("--timeout_s", type=float, default=60.0)
parser.add_argument("--record_video", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------
import datetime
import numpy as np
import torch
import gymnasium as gym

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla  # noqa: F401
import vla_warehouse  # noqa: F401
import vla_universal  # noqa: F401

from vla_universal.universal_env import UniversalDroneEnvCfg
from vla_universal.waypoint_controller import WaypointController
from vla_universal.semantic_map import SemanticMap, POI


def pick_target(pois: list[POI], drone_xy: np.ndarray, policy: str,
                index: int, target_id: str | None) -> POI:
    if target_id is not None:
        matches = [p for p in pois if p.id == target_id]
        if not matches:
            raise SystemExit(f"no POI with id='{target_id}'; "
                             f"available: {[p.id for p in pois]}")
        return matches[0]

    if policy == "closest":
        def d2(p):
            dx = p.xyz_world[0] - drone_xy[0]
            dy = p.xyz_world[1] - drone_xy[1]
            return dx * dx + dy * dy
        return min(pois, key=d2)
    elif policy == "highest_conf":
        return max(pois, key=lambda p: p.confidence)
    elif policy == "index":
        pois_sorted = sorted(pois, key=lambda p: p.id)
        return pois_sorted[min(index, len(pois_sorted) - 1)]
    elif policy == "ask":
        if len(pois) == 1:
            return pois[0]
        print("Multiple matches — re-run with --target_id <id>:")
        for p in sorted(pois, key=lambda p: p.id):
            print(f"  {p.id:24s}  cls={p.cls:16s}  xyz={p.xyz_world}  "
                  f"conf={p.confidence:.2f}")
        raise SystemExit(1)
    else:
        raise ValueError(f"unknown --pick policy: {policy}")


def quat_rotate_inverse_np(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a (w,x,y,z) quaternion."""
    w, x, y, z = q_wxyz
    # inverse of a unit quat = conjugate
    q_inv = np.array([w, -x, -y, -z], dtype=np.float32)
    # apply: use the same formula as _quat_apply_np in projection.py
    qxyz = q_inv[1:]
    t = 2.0 * np.cross(qxyz, v)
    return v + q_inv[0] * t + np.cross(qxyz, t)


def main() -> int:
    map_path = args.map or os.path.join(
        _DRONE_PROJECT, "logs", "maps", f"{args.scene}.json"
    )
    if not os.path.exists(map_path):
        print(f"[navigate] no map at {map_path}. "
              f"Run scan.py first: bash run_scan.sh --scene {args.scene}")
        return 2

    smap = SemanticMap.load(map_path)
    print(f"[navigate] loaded {len(smap.pois)} POIs from {map_path} "
          f"(classes: {', '.join(smap.classes())})")

    matches = smap.query(args.prompt)
    if not matches:
        print(f"[navigate] prompt '{args.prompt}' matched no POIs.")
        print(f"           available classes: {', '.join(smap.classes())}")
        return 2
    print(f"[navigate] prompt '{args.prompt}' -> {len(matches)} candidate(s)")

    # -----------------------------------------------------------------
    # Env
    # -----------------------------------------------------------------
    env_cfg = UniversalDroneEnvCfg()
    env_cfg.scene_name = args.scene
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = "cuda:0"

    env = gym.make("Isaac-VLADrone-Universal-v0", cfg=env_cfg)
    env_impl = env.unwrapped
    zero = torch.zeros((1, 4), device="cuda:0")
    for _ in range(10):
        env.step(zero)

    drone_pos, _drone_quat = env_impl.get_drone_pose()
    target = pick_target(matches, drone_pos[:2], args.pick, args.index, args.target_id)
    print(f"[navigate] selected target: {target.id} ({target.cls}) "
          f"at {target.xyz_world}  conf={target.confidence:.2f}")

    # -----------------------------------------------------------------
    # Controller + main loop
    # -----------------------------------------------------------------
    controller = WaypointController(device="cuda:0")
    target_xyz = np.array(target.xyz_world, dtype=np.float32)
    target_range = 3.0  # clamp body-frame target per-dim (waypoint policy training bound)

    # Video recording — write raw cam frames to an ffmpeg pipe
    video_dir = os.path.join(_DRONE_PROJECT, "videos", "vla_universal")
    os.makedirs(video_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(video_dir, f"navigate_{args.scene}_{stamp}.mp4")
    writer = None
    if args.record_video:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 50.0, (224 * 2, 224))
        print(f"[navigate] recording to {video_path}")

    max_steps = int(args.timeout_s * 50)  # env step_dt ≈ 0.02s
    converged_ticks = 0
    for step in range(max_steps):
        # Assemble 15-dim observation for the waypoint MLP
        fs = env_impl.get_flight_state()
        drone_pos, drone_quat = env_impl.get_drone_pose()
        pos_err_w = (target_xyz - drone_pos).astype(np.float32)
        target_body = quat_rotate_inverse_np(drone_quat, pos_err_w)
        # Clamp body-frame target to policy's training range
        target_body = np.clip(target_body, -target_range, target_range)

        action = controller.act(fs, target_body, pos_err_w)  # (4,)
        action_t = action.unsqueeze(0).float().clamp(-1.0, 1.0)
        _obs, _rew, _done, _info = env.step(action_t)

        # Video frame: front + observer side-by-side at 224×224 each
        if writer is not None and step % 2 == 0:  # 25fps
            batch = env_impl.get_camera_batch()
            import cv2
            front = (batch["rgb"][0] * 255).clip(0, 255).astype("uint8")
            front = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            front = cv2.resize(front, (224, 224))
            dist = float(np.linalg.norm(pos_err_w))
            cv2.putText(front, f"{target.cls} {dist:.1f}m", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            right = (batch["rgb"][1] * 255).clip(0, 255).astype("uint8")
            right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
            right = cv2.resize(right, (224, 224))
            combo = np.concatenate([front, right], axis=1)
            writer.write(combo)

        dist = float(np.linalg.norm(pos_err_w))
        speed = float(np.linalg.norm(fs[:3]))
        if dist < 0.5 and speed < 0.3:
            converged_ticks += 1
            if converged_ticks >= 50:
                print(f"[navigate] converged at step {step}: "
                      f"dist={dist:.2f}m, speed={speed:.2f}m/s")
                break
        else:
            converged_ticks = 0

        if step % 100 == 0:
            print(f"[navigate] step {step:5d}  "
                  f"drone=({drone_pos[0]:+.1f},{drone_pos[1]:+.1f},{drone_pos[2]:+.1f})  "
                  f"dist={dist:.2f}m  speed={speed:.2f}m/s")
    else:
        print(f"[navigate] timeout after {max_steps} steps. "
              f"Final dist={dist:.2f}m")

    if writer is not None:
        writer.release()
        print(f"[navigate] video saved: {video_path}")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        simulation_app.close()
    sys.exit(rc)

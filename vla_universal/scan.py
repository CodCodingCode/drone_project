"""Scan entry point.

Boots a UniversalDroneEnv in the chosen scene, teleports the drone through
a scripted waypoint pattern, captures RGB+depth from all 4 cameras at each
stop, then runs PaliGemma detection + 3D projection + dedup to produce
`logs/maps/<scene_name>.json`.

Launch via vla_universal/run_scan.sh — this file relies on AppLauncher
being initialized first.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scan a scene into a semantic map.")
parser.add_argument("--scene", type=str, default="warehouse_full",
                    help="Key in vla_warehouse.pois.SCENES")
parser.add_argument("--quick", action="store_true",
                    help="Halve altitude count + double xy spacing")
parser.add_argument("--extra_classes", type=str, default="",
                    help="Comma-separated extra detection classes")
parser.add_argument("--output", type=str, default=None,
                    help="Output map path (default logs/maps/<scene>.json)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle_ticks", type=int, default=8,
                    help="Physics ticks to run between teleport and capture")
parser.add_argument("--xy_spacing", type=float, default=6.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------
# After AppLauncher: torch / isaaclab / pxr are importable
# -----------------------------------------------------------------------
import numpy as np
import torch
import gymnasium as gym

# Ensure vla_warehouse + vla_universal are on the path
_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla  # noqa: F401 (imports shared env stuff)
import vla_warehouse  # noqa: F401
import vla_universal  # noqa: F401 (registers gym id)

from vla_universal.universal_env import UniversalDroneEnvCfg
from vla_universal.detector import PaliGemmaDetector, DEFAULT_CLASSES, Detection
from vla_universal.projection import bbox_to_world
from vla_universal.semantic_map import SemanticMap, cluster_detections
from vla_universal.flight_path import generate_scan_waypoints
from vla_warehouse import pois as _pois_mod


def main() -> int:
    # -----------------------------------------------------------------
    # Build env
    # -----------------------------------------------------------------
    env_cfg = UniversalDroneEnvCfg()
    env_cfg.scene_name = args.scene
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.sim.device = "cuda:0"

    print(f"[scan] scene = {args.scene}")
    env = gym.make("Isaac-VLADrone-Universal-v0", cfg=env_cfg)
    env_impl = env.unwrapped  # UniversalDroneEnv

    # -----------------------------------------------------------------
    # Derive scan bbox from scene registry (fallback: bbox discovered at load)
    # -----------------------------------------------------------------
    scene_entry = _pois_mod.get_scene(args.scene)
    ceiling_z = float(scene_entry.get("ceiling_z", 6.0))

    # Use known warehouse bbox by default; override from printed bbox if we
    # want to get fancier. For a first cut this is good enough because the
    # parent env's _force_load_and_inspect already logs the real bbox.
    # (We use a conservative bbox here so waypoints stay inside.)
    default_bboxes = {
        "warehouse_full":      ((-25.0, -38.0, 0.0), ( 6.0, 30.0, 9.3)),
        "warehouse":           ((-10.0, -15.0, 0.0), (10.0, 15.0, 6.0)),
        "warehouse_shelves":   ((-10.0, -15.0, 0.0), (10.0, 15.0, 6.0)),
        "hospital":            ((-15.0, -15.0, 0.0), (15.0, 15.0, 4.0)),
        "office":              ((-12.0, -12.0, 0.0), (12.0, 12.0, 3.5)),
    }
    bmin, bmax = default_bboxes.get(
        args.scene,
        ((-10.0, -10.0, 0.0), (10.0, 10.0, ceiling_z)),
    )
    scene_bbox = (np.array(bmin, dtype=np.float32), np.array(bmax, dtype=np.float32))

    waypoints = generate_scan_waypoints(
        scene_bbox_world=scene_bbox,
        ceiling_z=ceiling_z,
        xy_spacing=args.xy_spacing,
        quick=args.quick,
    )
    print(f"[scan] {len(waypoints)} waypoints over "
          f"{len({round(w[0][2], 2) for w in waypoints})} altitudes; "
          f"ceiling_z={ceiling_z}")

    # -----------------------------------------------------------------
    # Warmup: let the env settle so first camera capture works
    # -----------------------------------------------------------------
    zero_action = torch.zeros((1, 4), device="cuda:0")
    for _ in range(args.settle_ticks * 2):
        env.step(zero_action)

    # -----------------------------------------------------------------
    # Flight: teleport + capture at each waypoint
    # -----------------------------------------------------------------
    frames = []  # list of dicts with rgb/depth/cam_pos_w/cam_quat_w per stop
    t0 = time.time()
    for i, (xyz, yaw) in enumerate(waypoints):
        env_impl.teleport_drone(xyz, yaw)
        # run a few physics ticks so cameras stabilize
        for _ in range(args.settle_ticks):
            env.step(zero_action)
        batch = env_impl.get_camera_batch()
        batch["waypoint_idx"] = i
        frames.append(batch)
        if (i + 1) % max(1, len(waypoints) // 10) == 0 or i == len(waypoints) - 1:
            print(f"[scan] flight {i + 1}/{len(waypoints)} "
                  f"@ ({xyz[0]:+.1f},{xyz[1]:+.1f},{xyz[2]:+.1f}m)  "
                  f"elapsed={time.time() - t0:.1f}s")

    flight_elapsed = time.time() - t0
    print(f"[scan] flight complete in {flight_elapsed:.1f}s — running detection ...")

    # -----------------------------------------------------------------
    # Detection — batch 4 cams per stop
    # -----------------------------------------------------------------
    classes = list(DEFAULT_CLASSES)
    if args.extra_classes:
        extras = [c.strip().lower() for c in args.extra_classes.split(",") if c.strip()]
        for c in extras:
            if c not in classes:
                classes.append(c)
    print(f"[scan] detecting {len(classes)} classes: {', '.join(classes)}")

    detector = PaliGemmaDetector(device="cuda:0")

    detections_with_xyz: list[tuple[Detection, np.ndarray]] = []
    t1 = time.time()
    for fidx, frame in enumerate(frames):
        rgb_batch = frame["rgb"]       # (4, 224, 224, 3) in [0,1]
        per_cam_dets = detector.detect_batch(
            rgbs=rgb_batch, classes=classes,
            cam_idx_offset=0, frame_idx=fidx,
        )
        # per_cam_dets[i] is list[Detection] for cam i
        for cam_idx, dets in enumerate(per_cam_dets):
            # detector assigned cam_idx = cam_idx_offset + i; we zero-offset,
            # so dets[j].cam_idx equals the image index within the batch.
            for d in dets:
                result = bbox_to_world(
                    bbox_xyxy=d.bbox_xyxy,
                    depth_img=frame["depth_m"][d.cam_idx],
                    cam_pos_w=frame["cam_pos_w"][d.cam_idx],
                    cam_quat_w=frame["cam_quat_w"][d.cam_idx],
                )
                if result is None:
                    continue
                xyz, _depth = result
                detections_with_xyz.append((d, xyz))
        if (fidx + 1) % max(1, len(frames) // 8) == 0:
            print(f"[scan] detected {fidx + 1}/{len(frames)} frames, "
                  f"{len(detections_with_xyz)} raw detections so far")
    print(f"[scan] detection complete in {time.time() - t1:.1f}s — "
          f"{len(detections_with_xyz)} raw detections")

    # -----------------------------------------------------------------
    # Cluster + save map
    # -----------------------------------------------------------------
    pois = cluster_detections(
        detections_with_xyz,
        scene_bbox=scene_bbox,
        min_views=2,
    )
    print(f"[scan] clustered into {len(pois)} POIs")

    smap = SemanticMap(
        scene_name=args.scene,
        scene_usd_path=scene_entry["usd_path"],
        scene_bbox_world=[list(map(float, bmin)), list(map(float, bmax))],
        pois=pois,
        scan_params={
            "altitudes": sorted({round(w[0][2], 2) for w in waypoints}),
            "xy_spacing": args.xy_spacing,
            "stops": len(waypoints),
            "quick": args.quick,
        },
    )
    out = args.output or os.path.join(
        _DRONE_PROJECT, "logs", "maps", f"{args.scene}.json"
    )
    smap.save(out)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    by_cls: dict[str, int] = {}
    for p in pois:
        by_cls[p.cls] = by_cls.get(p.cls, 0) + 1
    if by_cls:
        print("[scan] POIs by class:")
        for c, n in sorted(by_cls.items(), key=lambda kv: -kv[1]):
            print(f"    {c:24s} {n}")
    else:
        print("[scan] no POIs found — check detector output or increase scan density")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        simulation_app.close()
    sys.exit(rc)

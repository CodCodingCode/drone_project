"""Scripted scan waypoints — boustrophedon xy grid × several altitudes.

Input:  scene bounding box (world coords) and ceiling height.
Output: list of (xyz_world, yaw_rad) waypoints.

The scan loop calls `env.teleport_drone(xyz, yaw)` at each waypoint,
waits a few physics ticks for the cameras to settle, captures RGB+depth,
and moves to the next. No policy inference during scan.
"""

from __future__ import annotations

import math

import numpy as np


def generate_scan_waypoints(
    scene_bbox_world: tuple[np.ndarray, np.ndarray],  # (min_xyz, max_xyz)
    ceiling_z: float,
    xy_spacing: float = 6.0,
    quick: bool = False,
    inset: float = 3.0,
) -> list[tuple[np.ndarray, float]]:
    """Generate a coarse survey path over the scene.

    * 3 altitudes at 30% / 55% / 80% of ceiling height (clamped ≥ 0.8 m).
    * xy grid inset from the scene bbox by `inset` meters on each side
      to avoid spawning inside walls at the perimeter.
    * Boustrophedon (serpentine) ordering minimizes travel distance and
      induces forward motion between stops — the waypoint policy likes
      motion over pure-yaw.
    * `quick=True` halves both altitude count and xy density for a
      ~2-minute scan vs ~4-minute full scan.

    Returns: list of (xyz_world float32, yaw_rad float).
    Yaw follows the direction of motion between consecutive stops so
    the drone's heading changes smoothly instead of spinning in place.
    """
    bmin, bmax = np.asarray(scene_bbox_world[0]), np.asarray(scene_bbox_world[1])
    xmin, xmax = bmin[0] + inset, bmax[0] - inset
    ymin, ymax = bmin[1] + inset, bmax[1] - inset

    # Clamp altitudes sensibly
    alt_mults = [0.3, 0.55, 0.8] if not quick else [0.4, 0.7]
    altitudes = [max(0.8, ceiling_z * m) for m in alt_mults]

    if quick:
        xy_spacing = xy_spacing * 1.8

    # Generate xy grid points — serpentine
    xs = _linspace_inclusive(xmin, xmax, xy_spacing)
    ys = _linspace_inclusive(ymin, ymax, xy_spacing)

    xy_path: list[tuple[float, float]] = []
    for row_idx, y in enumerate(ys):
        row = [(x, y) for x in (xs if row_idx % 2 == 0 else list(reversed(xs)))]
        xy_path.extend(row)

    # Build 3D waypoint list with per-leg yaw toward the next waypoint.
    waypoints: list[tuple[np.ndarray, float]] = []
    for z in altitudes:
        for i, (x, y) in enumerate(xy_path):
            # Yaw: point toward next waypoint (wraps at end of altitude)
            if i < len(xy_path) - 1:
                nx, ny = xy_path[i + 1]
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = waypoints[-1][1] if waypoints else 0.0
            waypoints.append((np.array([x, y, z], dtype=np.float32), float(yaw)))

    return waypoints


def _linspace_inclusive(lo: float, hi: float, step: float) -> list[float]:
    """Points from lo to hi (inclusive) spaced by ~step. Guarantees ≥ 2 points."""
    if hi <= lo:
        return [(lo + hi) / 2.0]
    n = max(2, int(round((hi - lo) / step)) + 1)
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

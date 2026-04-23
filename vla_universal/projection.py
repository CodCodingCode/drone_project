"""2D pixel + depth + camera pose → 3D world point.

For the 4 onboard cameras, each is a PinholeCameraCfg with:
  focal_length=10 mm, horizontal_aperture=20 mm, resolution=224×224 → ~90° FOV

Isaac Sim's camera convention (same as OpenGL/USD): camera looks down -Z,
with +Y up and +X right in camera-local coords. A pixel at (cx, cy) where
top-left is (0, 0) maps to:
    u = cx / (W-1) - 0.5        # [-0.5, 0.5]
    v = cy / (H-1) - 0.5
    ray_cam = normalize([ u*(aperture/focal),  -v*(aperture/focal),  -1 ])
Multiply by depth to get the 3D point in camera frame; then apply the
camera's world quaternion and add its world position.

`depth_patch_median` samples a 7×7 neighborhood around (cx, cy) and
returns the median — robust to specular highlights, edge pixels, and
thin-object aliasing (e.g. pallet slats).
"""

from __future__ import annotations

import numpy as np


# Default intrinsics for the 4 cams in vla/vla_drone_env.py. If that config
# changes, override these via kwargs on pixel_to_world().
FOCAL_MM = 10.0
APERTURE_MM = 20.0
IMG_SIZE = 224


def depth_patch_median(
    depth_img: np.ndarray,      # (H, W) float meters
    cx: int, cy: int,
    patch: int = 3,             # half-width — 7×7 when patch=3
    valid_min: float = 0.3,
    valid_max: float = 18.0,
) -> float:
    """Median of non-bad depth values in a (2*patch+1)² neighborhood around
    (cx, cy). Returns nan if no valid pixels.
    """
    h, w = depth_img.shape
    x0 = max(0, cx - patch); x1 = min(w, cx + patch + 1)
    y0 = max(0, cy - patch); y1 = min(h, cy + patch + 1)
    patch_vals = depth_img[y0:y1, x0:x1].ravel()
    # Drop NaN / inf / out-of-range
    valid = patch_vals[
        np.isfinite(patch_vals) & (patch_vals >= valid_min) & (patch_vals <= valid_max)
    ]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def _quat_apply_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply quaternion (w,x,y,z) to 3-vector — numpy implementation matching
    isaaclab.utils.math.quat_apply. Returns (3,) or (N, 3)."""
    q = np.asarray(quat_wxyz, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # v_rot = v + 2*w*(q_xyz × v) + 2*(q_xyz × (q_xyz × v))
    qxyz = q[..., 1:]
    t = 2.0 * np.cross(qxyz, v)
    return v + w[..., None] * t + np.cross(qxyz, t)


def pixel_to_world(
    cx: int, cy: int,
    depth_m: float,
    cam_pos_w: np.ndarray,      # (3,)
    cam_quat_w: np.ndarray,     # (4,) wxyz
    focal_mm: float = FOCAL_MM,
    aperture_mm: float = APERTURE_MM,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Unproject a pixel to a 3D world coordinate given its depth and the
    camera's world-space pose. Returns (3,) float32.
    """
    u = cx / (img_size - 1) - 0.5
    v = cy / (img_size - 1) - 0.5
    scale = aperture_mm / focal_mm   # 2.0 for default cams (90° FOV)

    # Camera-frame ray (unnormalized) — looks along -Z, +Y up, +X right.
    # Row 0 = top of image corresponds to +Y in camera frame, so v is negated.
    ray_cam = np.array([u * scale, -v * scale, -1.0], dtype=np.float32)
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    ray_world = _quat_apply_np(cam_quat_w, ray_cam)
    return cam_pos_w.astype(np.float32) + depth_m * ray_world.astype(np.float32)


def bbox_to_world(
    bbox_xyxy: tuple[int, int, int, int],
    depth_img: np.ndarray,      # (H, W) float meters
    cam_pos_w: np.ndarray,
    cam_quat_w: np.ndarray,
    img_size: int = IMG_SIZE,
) -> tuple[np.ndarray, float] | None:
    """Convenience: bbox center + depth patch → world xyz + depth used.
    Returns None if depth patch has no valid pixels.
    """
    x0, y0, x1, y1 = bbox_xyxy
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    depth_m = depth_patch_median(depth_img, cx, cy)
    if not np.isfinite(depth_m):
        return None
    xyz = pixel_to_world(cx, cy, depth_m, cam_pos_w, cam_quat_w, img_size=img_size)
    return xyz, depth_m

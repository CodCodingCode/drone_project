"""Single-env drone environment for scanning + open-vocabulary navigation.

Inherits VLAWarehouseDroneEnv for its scene-loading machinery (USD reference,
payload forcing, bbox reporting) but:
  * does NOT spawn cube/sphere/cylinder marker objects (set num_active_pois=0)
  * does NOT reset episodes on success/wrong-object (episode_length_s huge)
  * always num_envs=1
  * exposes get_camera_batch() for post-flight detection and set_waypoint_target()
    for the scan + navigate control loops

The env's built-in PPO reward/termination logic is irrelevant here — we drive
the drone with external scripted waypoints, not a policy rollout, so most of
the parent's reward code runs but its outputs are unused.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from vla_warehouse.vla_warehouse_env import (
    VLAWarehouseDroneEnv as _BaseEnv,
    VLAWarehouseDroneEnvCfg as _BaseCfg,
)


@configclass
class UniversalDroneEnvCfg(_BaseCfg):
    # Keep marker-spawn logic in the parent (so the 3 markers still appear
    # and get placed by _reset_idx), but we use them for visualization only.
    # The scan/navigate scripts drive the drone directly via target_body;
    # the parent's reward machinery is idle since we bypass env.step().
    num_active_pois: int = 3

    # Very long episode — we don't want auto-reset during scan or navigate
    episode_length_s = 600.0

    # Single env
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=40.0, replicate_physics=True,
    )

    # Deterministic spawn for reproducible scans
    spawn_xy_radius: float = 0.0


class UniversalDroneEnv(_BaseEnv):
    cfg: UniversalDroneEnvCfg

    # -----------------------------------------------------------------
    # Camera batch — returns all 4 onboard cameras + pose at shutter time
    # -----------------------------------------------------------------
    def get_camera_batch(self) -> dict:
        """Grab the current frame from all 4 cameras, plus each camera's
        world-frame pose. Call after _maybe_capture_camera has fired.

        Returns a dict with numpy arrays:
          rgb       (4, 224, 224, 3) float32 in [0, 1]
          depth_m   (4, 224, 224)    float32 meters (un-normalized)
          cam_pos_w (4, 3)           float32 world position of each cam
          cam_quat_w(4, 4)           float32 (w,x,y,z) world orientation
        """
        rgb = self._cached_rgb[0].detach().cpu().numpy()            # (4, 224, 224, 3)
        # _cached_depth stored normalized to [0, 1] by /20 — undo it
        depth_m = self._cached_depth[0].detach().cpu().numpy() * 20.0  # (4, 224, 224)

        drone_pos = self._robot.data.root_pos_w[0]     # (3,)
        drone_quat = self._robot.data.root_quat_w[0]   # (4,) wxyz
        cam_offset = self._cam_offset                  # (3,)
        cam_pos = drone_pos + quat_apply(drone_quat, cam_offset)  # (3,)

        cam_pos_w = np.zeros((4, 3), dtype=np.float32)
        cam_quat_w = np.zeros((4, 4), dtype=np.float32)
        for i, cam in enumerate(self._cameras):
            rot_offset = torch.tensor(cam.cfg.offset.rot,
                                       dtype=torch.float32, device=self.device)
            cam_quat = quat_mul(drone_quat, rot_offset)
            cam_pos_w[i] = cam_pos.detach().cpu().numpy()
            cam_quat_w[i] = cam_quat.detach().cpu().numpy()

        return {
            "rgb": rgb,
            "depth_m": depth_m,
            "cam_pos_w": cam_pos_w,
            "cam_quat_w": cam_quat_w,
        }

    # -----------------------------------------------------------------
    # Direct teleport (for scan flight paths) — bypasses RL, uses write-pose
    # -----------------------------------------------------------------
    def teleport_drone(self, xyz_world: np.ndarray, yaw_rad: float = 0.0) -> None:
        """Set drone pose directly (no physics transient). Used by the scan
        flight path so we get deterministic captures at each waypoint without
        waiting for the waypoint policy to converge.
        """
        pos = torch.tensor(xyz_world, dtype=torch.float32, device=self.device).unsqueeze(0)
        half = yaw_rad * 0.5
        quat = torch.tensor(
            [np.cos(half), 0.0, 0.0, np.sin(half)],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        pose = torch.cat([pos, quat], dim=-1)  # (1, 7)
        zeros = torch.zeros(1, 6, device=self.device)

        env_ids = torch.tensor([0], device=self.device)
        self._robot.write_root_pose_to_sim(pose, env_ids)
        self._robot.write_root_velocity_to_sim(zeros, env_ids)

    # -----------------------------------------------------------------
    # Flight state for waypoint controller
    # -----------------------------------------------------------------
    def get_flight_state(self) -> np.ndarray:
        """Return the 9-dim body-frame flight state the waypoint policy expects."""
        lin_v = self._robot.data.root_lin_vel_b[0].detach().cpu().numpy()
        ang_v = self._robot.data.root_ang_vel_b[0].detach().cpu().numpy()
        grav  = self._robot.data.projected_gravity_b[0].detach().cpu().numpy()
        return np.concatenate([lin_v, ang_v, grav]).astype(np.float32)

    def get_drone_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(pos_w, quat_w_wxyz) for the drone."""
        pos = self._robot.data.root_pos_w[0].detach().cpu().numpy()
        quat = self._robot.data.root_quat_w[0].detach().cpu().numpy()
        return pos.astype(np.float32), quat.astype(np.float32)

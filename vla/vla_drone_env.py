"""VLA drone navigation environment — PaliGemma end-to-end.

Same scene, rewards, and physics as lang_nav, but instead of frozen CLIP
embeddings, the env provides raw RGB images and tokenized text commands.
PaliGemma processes these inside the policy network (vla_policy.py).

Observation (multi-group dict):
  "policy"      (N, 9)         flight state
  "rgb"         (N, 4, 224, 224, 3) 4-camera RGB [0, 1] float (front/right/back/left)
  "text_tokens" (N, 32)        tokenized text command IDs
  "text_mask"   (N, 32)        attention mask
"""

from __future__ import annotations

import json
import os
import random

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip

# Reuse command bank from lang_nav
import sys
_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)
from lang_nav.commands import COMMANDS, OBJECT_TYPES

# Fixed object offsets (same as lang_nav)
_OBJ_OFFSETS = [(-1.5, 0.0, 0.2), (1.5, 0.0, 0.2), (0.0, 1.5, 0.2)]

# Max tokenized text length for PaliGemma (256 image tokens + text tokens + padding)
_NUM_IMAGE_TOKENS = 256
_MAX_TEXT_LEN = 280  # 256 image + ~20 text + margin


@configclass
class VLADroneEnvCfg(DirectRLEnvCfg):
    # Episode / stepping
    episode_length_s = 15.0
    decimation = 2
    action_space = 4
    observation_space = 9  # flight state only (base class needs an int; actual obs is a dict)
    state_space = 0
    debug_vis = False

    # Simulation (identical to lang_nav)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.18, 0.15)),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=6.0, replicate_physics=True,
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # 4 onboard cameras (front/right/back/left) for 360° coverage
    # Each 90° FOV, 224x224 RGB + depth
    _cam_spawn = sim_utils.PinholeCameraCfg(
        focal_length=10.0,       # shorter focal length → wider FOV
        focus_distance=100.0,
        horizontal_aperture=20.0,  # ~90° FOV: 2*atan(20/(2*10)) ≈ 90°
        clipping_range=(0.01, 20.0),
    )
    # Front camera: looks along +X body axis (default drone forward)
    # Isaac camera convention: -Z is view direction, +Y is up
    # rot (0.5, 0.5, -0.5, -0.5) maps camera -Z → body +X (forward)
    cam_front: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CamFront",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, 0.5, -0.5, -0.5), convention="world"),
        data_types=["rgb", "distance_to_camera"], spawn=_cam_spawn, width=224, height=224,
    )
    # Right camera: looks along +Y body axis (90° CW from front)
    # rot that maps camera -Z → body +Y
    cam_right: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CamRight",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.707107, 0.0, -0.707107), convention="world"),
        data_types=["rgb", "distance_to_camera"], spawn=_cam_spawn, width=224, height=224,
    )
    # Back camera: looks along -X body axis (180° from front)
    # rot that maps camera -Z → body -X
    cam_back: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CamBack",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, 0.5), convention="world"),
        data_types=["rgb", "distance_to_camera"], spawn=_cam_spawn, width=224, height=224,
    )
    # Left camera: looks along -Y body axis (270° from front)
    # rot that maps camera -Z → body -Y
    cam_left: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CamLeft",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.707107, 0.0, 0.707107), convention="world"),
        data_types=["rgb", "distance_to_camera"], spawn=_cam_spawn, width=224, height=224,
    )
    camera_body_offset_pos = (0.05, 0.0, 0.01)

    # How often to grab camera frames (every N physics steps)
    camera_every_n = 4  # every 4 steps — 4 cameras so rendering is heavier

    # PaliGemma tokenizer name (lightweight, no model weights loaded in env)
    tokenizer_name = "google/paligemma-3b-pt-224"
    max_text_length = _MAX_TEXT_LEN

    # Reward scales — stability (always active)
    lin_vel_reward_scale = 0.0   # don't penalize movement — we want navigation
    ang_vel_reward_scale = -0.005
    alive_reward = 0.0  # waypoint policy handles flight stability, no free reward for existing
    uprightness_reward_scale = 0.2
    altitude_warning_low = 0.3
    altitude_warning_high = 2.8
    crash_penalty_scale = -10.0

    # Reward scales — navigation (gated by nav_multiplier)
    distance_to_goal_reward_scale = 35.0
    velocity_toward_goal_scale = 4.0
    proximity_scale = 8.0
    proximity_radius = 1.5
    success_reward = 25.0
    hover_at_target_reward = 30.0  # base reward when conditions met; scales with dwell time
    hover_at_target_radius = 0.5   # moderate radius — small enough to be "at target"
    hover_max_speed = 1.0          # relaxed — waypoint policy doesn't naturally brake hard
    hover_dwell_bonus = 3.0        # big bonus for sustained hovering (up to 4x base at dwell cap)
    hover_max_dwell_steps = 50     # dwell counter caps here (1.0s at 50Hz env step)
    wrong_object_penalty = -3.0

    success_threshold = 0.35

    # No curriculum needed — waypoint policy handles flight from iter 0
    survival_only_steps = 0
    nav_fadein_steps = 1  # instant full-scale navigation rewards
    alive_fadeout_steps = 1  # alive is minimal from the start
    alive_min_scale = 0.2

    # Precision curriculum: after ~200 iters of learning "get close," fade distance/proximity
    # rewards down and boost hover/success rewards so the drone must stop at targets
    precision_curriculum_start = 409600     # ~200 iters (8 steps × 256 envs × 200)
    precision_curriculum_steps = 614400    # fade over ~300 more iters (done by ~500 iters)
    hover_radius_start = 0.5
    hover_radius_end = 0.5


class VLADroneEnv(DirectRLEnv):
    cfg: VLADroneEnvCfg

    def __init__(self, cfg: VLADroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action / thrust buffers
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Per-env language state
        self._target_obj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._current_commands: list[str] = [""] * self.num_envs

        # Tokenized text (pre-computed at reset, reused every step)
        self._text_tokens = torch.zeros(self.num_envs, self.cfg.max_text_length, dtype=torch.long, device=self.device)
        self._text_mask = torch.zeros(self.num_envs, self.cfg.max_text_length, dtype=torch.long, device=self.device)

        # Cached camera images — 4 views (updated every N steps)
        self._num_cameras = 4
        self._cached_rgb = torch.zeros(self.num_envs, 4, 224, 224, 3, dtype=torch.float32, device=self.device)
        self._cached_depth = torch.zeros(self.num_envs, 4, 224, 224, dtype=torch.float32, device=self.device)
        self._steps_since_capture = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Dwell counter: consecutive steps inside hover zone with low speed
        self._hover_dwell = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Camera body-frame offset
        self._cam_offset = torch.tensor(self.cfg.camera_body_offset_pos, dtype=torch.float32, device=self.device)
        # Camera orientation offset (forward-facing, upright) in drone body frame
        # Quaternion (w,x,y,z) = (0.5, 0.5, -0.5, -0.5) rotates Isaac default camera (-Z view, +Y up)
        # to match drone body frame (camera looks at +X, up is +Z)
        # Verified by quaternion math: roll=90°, pitch=0°, yaw=-90°
        self._cam_rot_offset = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=torch.float32, device=self.device)

        # World positions of all 3 objects: (num_envs, 3, 3) — randomized at each reset
        self._obj_pos_w = torch.zeros(self.num_envs, 3, 3, dtype=torch.float32, device=self.device)

        # Robot dynamics
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Logging
        self._episode_sums = {
            k: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for k in [
                "lin_vel", "ang_vel", "alive", "uprightness", "altitude_penalty",
                "distance_to_goal", "velocity_toward_goal", "proximity",
                "hover_at_target", "success", "wrong_object",
            ]
        }

        # Metrics file
        self._metrics_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "vla_metrics.jsonl",
        )
        os.makedirs(os.path.dirname(self._metrics_path), exist_ok=True)
        self._metrics_file = open(self._metrics_path, "w")
        self._log_step = 0

        # Load PaliGemma processor (tokenizer + image token handling)
        from transformers import AutoProcessor
        print("[VLA Env] Loading PaliGemma processor...")
        self._processor = AutoProcessor.from_pretrained(self.cfg.tokenizer_name)
        self._image_token_id = self._processor.tokenizer.convert_tokens_to_ids("<image>")
        print(f"[VLA Env] Processor loaded. Image token id: {self._image_token_id}")

        # Initial reset
        self._reset_idx(None)

    # ------------------------------------------------------------------
    # Scene setup (identical to lang_nav)
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        # 4 cameras for 360° coverage
        self._cameras = [
            TiledCamera(self.cfg.cam_front),
            TiledCamera(self.cfg.cam_right),
            TiledCamera(self.cfg.cam_back),
            TiledCamera(self.cfg.cam_left),
        ]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Spawn objects (same as lang_nav)
        cube_cfg = sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        )
        cube_cfg.func("/World/envs/env_.*/cube", cube_cfg, translation=_OBJ_OFFSETS[0])

        sphere_cfg = sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
        )
        sphere_cfg.func("/World/envs/env_.*/sphere", sphere_cfg, translation=_OBJ_OFFSETS[1])

        cylinder_cfg = sim_utils.CylinderCfg(
            radius=0.2, height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        )
        cylinder_cfg.func("/World/envs/env_.*/cylinder", cylinder_cfg, translation=_OBJ_OFFSETS[2])

        self.scene.clone_environments(copy_from_source=False)

        # Create XformPrimViews for repositioning objects at reset
        from isaaclab.sim.views import XformPrimView
        self._cube_view = XformPrimView("/World/envs/env_.*/cube", device=self.device)
        self._sphere_view = XformPrimView("/World/envs/env_.*/sphere", device=self.device)
        self._cylinder_view = XformPrimView("/World/envs/env_.*/cylinder", device=self.device)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self._robot
        for i, cam in enumerate(self._cameras):
            self.scene.sensors[f"tiled_camera_{i}"] = cam

        light_cfg = sim_utils.DomeLightCfg(
            intensity=1500.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg)
        dist_light = sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.95, 0.85))
        dist_light.func("/World/SunLight", dist_light)

        self._target_marker = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/World/Visuals/target_marker",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.15,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.8, 0.0),
                        emissive_color=(1.0, 0.8, 0.0),
                        opacity=0.85,
                    ),
                ),
            },
        ))

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _update_camera_pose(self):
        drone_pos = self._robot.data.root_pos_w
        drone_quat = self._robot.data.root_quat_w
        cam_pos = drone_pos + quat_apply(drone_quat, self._cam_offset.expand(self.num_envs, -1))
        # Each camera has its own rotation offset baked into the TiledCameraCfg.
        # We compose drone orientation with each camera's offset.
        for cam in self._cameras:
            cam_rot_offset = torch.tensor(cam.cfg.offset.rot, dtype=torch.float32, device=self.device)
            cam_offset_expanded = cam_rot_offset.unsqueeze(0).expand(self.num_envs, -1)
            cam_quat = quat_mul(drone_quat, cam_offset_expanded)
            cam._view.set_world_poses(cam_pos, cam_quat)

    def _maybe_capture_camera(self):
        self._steps_since_capture += 1
        if (self._steps_since_capture >= self.cfg.camera_every_n).any():
            for i, cam in enumerate(self._cameras):
                rgb = cam.data.output["rgb"][:, :, :, :3]  # (N, 224, 224, 3)
                self._cached_rgb[:, i] = rgb.float() / 255.0
                depth = cam.data.output["distance_to_camera"]  # (N, 224, 224, 1)
                self._cached_depth[:, i] = depth.squeeze(-1).clamp(0.0, 20.0) / 20.0
            self._steps_since_capture.zero_()

    # ------------------------------------------------------------------
    # Two-phase learning
    # ------------------------------------------------------------------

    def _get_nav_multiplier(self) -> float:
        steps = self.common_step_counter
        if steps < self.cfg.survival_only_steps:
            return 0.0
        return min((steps - self.cfg.survival_only_steps) / self.cfg.nav_fadein_steps, 1.0)

    def _get_alive_scale(self) -> float:
        """Alive reward fades from 1.0 to alive_min_scale over training."""
        steps = self.common_step_counter
        fade = min(steps / self.cfg.alive_fadeout_steps, 1.0)
        return 1.0 - fade * (1.0 - self.cfg.alive_min_scale)

    def _get_precision_scale(self) -> float:
        """Curriculum: interpolates 0 → 1 over precision_curriculum_steps.

        At 0: loose targeting (full distance/proximity rewards, wide hover radius)
        At 1: precise targeting (shrunken rewards for approach, wide hover bonus for stopping)
        """
        steps = self.common_step_counter
        start = self.cfg.precision_curriculum_start
        length = self.cfg.precision_curriculum_steps
        if steps < start:
            return 0.0
        return min((steps - start) / length, 1.0)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        self._update_camera_pose()
        self._maybe_capture_camera()

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    # ------------------------------------------------------------------
    # Observations (multi-group dict for VLA)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Update target marker
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos_w = self._obj_pos_w[env_ids, self._target_obj_idx]  # (N, 3)
        marker_pos = target_pos_w.clone()
        marker_pos[:, 2] += 0.7
        self._target_marker.visualize(translations=marker_pos)

        flight_state = torch.cat([
            self._robot.data.root_lin_vel_b,       # (N, 3)
            self._robot.data.root_ang_vel_b,       # (N, 3)
            self._robot.data.projected_gravity_b,   # (N, 3)
        ], dim=-1)  # (N, 9)

        # Ground-truth target in body frame (for auxiliary supervision during training only)
        target_gt_body, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            target_pos_w,
        )  # (N, 3)

        # World-frame position error (needed by frozen waypoint policy for slots 12-15)
        pos_error_w = target_pos_w - self._robot.data.root_pos_w  # (N, 3)

        return {
            "policy": flight_state,              # (N, 9)
            "rgb": self._cached_rgb,             # (N, 4, 224, 224, 3) float [0,1] — 4 views
            "text_tokens": self._text_tokens,    # (N, 280) long
            "text_mask": self._text_mask,        # (N, 280) long
            "vla_token_features": torch.zeros(self.num_envs, 1048, 2048, dtype=torch.float16, device=self.device),  # placeholder (1024 image + 24 text tokens), filled by train loop
            "target_gt_body": target_gt_body,    # (N, 3) ground-truth target in body frame for aux supervision
            "pos_error_w": pos_error_w,          # (N, 3) world-frame position error for frozen waypoint policy
            "target_obj_idx": self._target_obj_idx.float(),  # (N,) object class — cast to long for CE loss
            "depth": self._cached_depth,         # (N, 4, 224, 224) normalized depth [0, 1] — 4 views
        }

    # ------------------------------------------------------------------
    # Rewards (identical to lang_nav)
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        drone_pos = self._robot.data.root_pos_w

        target_pos = self._obj_pos_w[env_ids, self._target_obj_idx]
        to_goal = target_pos - drone_pos
        dist_to_target = torch.linalg.norm(to_goal, dim=1)
        dist_mapped = 1.0 - torch.tanh(dist_to_target / 0.8)

        to_goal_dir = to_goal / dist_to_target.unsqueeze(1).clamp(min=0.01)
        vel_toward = torch.sum(self._robot.data.root_lin_vel_w * to_goal_dir, dim=1)
        # Only reward velocity toward goal when far away (>1.0m), not when close (prevents circling)
        far_from_target = (dist_to_target > 1.0).float()
        vel_toward_clamp = torch.clamp(vel_toward, 0.0, 2.0) * far_from_target

        inside_radius = (dist_to_target < self.cfg.proximity_radius).float()
        proximity = inside_radius * (1.0 - dist_to_target / self.cfg.proximity_radius)

        # Real hover reward: inside radius AND low speed, scales with consecutive dwell time
        precision = self._get_precision_scale()  # 0.0 → 1.0
        hover_radius = self.cfg.hover_radius_start + precision * (self.cfg.hover_radius_end - self.cfg.hover_radius_start)
        near_target = (dist_to_target < hover_radius)
        speed = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1)
        slow = (speed < self.cfg.hover_max_speed)
        hovering = near_target & slow  # bool

        # Update dwell counter: +1 when hovering, reset to 0 when not
        self._hover_dwell = torch.where(
            hovering,
            (self._hover_dwell + 1.0).clamp(max=float(self.cfg.hover_max_dwell_steps)),
            torch.zeros_like(self._hover_dwell),
        )
        # Reward scales linearly from base to base + bonus*max_dwell as dwell time grows
        # At step 1 hovering: reward = 1.0x base
        # At step 50 hovering: reward = (1 + dwell_bonus) * base = 3x base (if bonus=2.0)
        dwell_ratio = self._hover_dwell / float(self.cfg.hover_max_dwell_steps)  # 0.0 → 1.0
        dwell_multiplier = 1.0 + self.cfg.hover_dwell_bonus * dwell_ratio
        hover_at_target = hovering.float() * dwell_multiplier

        uprightness = -self._robot.data.projected_gravity_b[:, 2]

        too_low = torch.clamp(self.cfg.altitude_warning_low - drone_pos[:, 2], min=0.0)
        too_high = torch.clamp(drone_pos[:, 2] - self.cfg.altitude_warning_high, min=0.0)
        altitude_penalty = (too_low + too_high) * self.cfg.crash_penalty_scale

        wrong_mask = torch.ones(self.num_envs, 3, dtype=torch.bool, device=self.device)
        wrong_mask[env_ids, self._target_obj_idx] = False
        wrong_obj_pos = self._obj_pos_w[wrong_mask].reshape(self.num_envs, 2, 3)
        dist_to_wrong = torch.linalg.norm(
            wrong_obj_pos - drone_pos.unsqueeze(1), dim=-1
        ).min(dim=1).values

        success = dist_to_target < self.cfg.success_threshold
        wrong_object = dist_to_wrong < self.cfg.success_threshold

        nav = self._get_nav_multiplier()
        alive_scale = self._get_alive_scale()

        # Precision curriculum: loose rewards fade, precise rewards amplify
        # loose_scale: 1.0 → 0.2 (distance, proximity, velocity_toward_goal shrink)
        # precise_scale: 1.0 → 3.0 (hover, success grow)
        loose_scale = 1.0 - 0.8 * precision
        precise_scale = 1.0 + 2.0 * precision

        rewards = {
            "lin_vel": torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1) * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1) * self.cfg.ang_vel_reward_scale * self.step_dt,
            "alive": torch.ones(self.num_envs, device=self.device) * self.cfg.alive_reward * alive_scale * self.step_dt,
            "uprightness": uprightness * self.cfg.uprightness_reward_scale * self.step_dt,
            "altitude_penalty": altitude_penalty * self.step_dt,
            "distance_to_goal": dist_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt * nav * loose_scale,
            "velocity_toward_goal": vel_toward_clamp * self.cfg.velocity_toward_goal_scale * self.step_dt * nav * loose_scale,
            "proximity": proximity * self.cfg.proximity_scale * self.step_dt * nav * loose_scale,
            "hover_at_target": hover_at_target * self.cfg.hover_at_target_reward * self.step_dt * nav * precise_scale,
            "success": success.float() * self.cfg.success_reward * nav * precise_scale,
            "wrong_object": wrong_object.float() * self.cfg.wrong_object_penalty * nav * precise_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v
        return reward

    # ------------------------------------------------------------------
    # Termination (identical to lang_nav)
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = torch.arange(self.num_envs, device=self.device)
        drone_pos = self._robot.data.root_pos_w

        target_pos = self._obj_pos_w[env_ids, self._target_obj_idx]
        dist_to_target = torch.linalg.norm(target_pos - drone_pos, dim=1)

        wrong_mask = torch.ones(self.num_envs, 3, dtype=torch.bool, device=self.device)
        wrong_mask[env_ids, self._target_obj_idx] = False
        wrong_obj_pos = self._obj_pos_w[wrong_mask].reshape(self.num_envs, 2, 3)
        dist_to_wrong = torch.linalg.norm(
            wrong_obj_pos - drone_pos.unsqueeze(1), dim=-1
        ).min(dim=1).values

        success = dist_to_target < self.cfg.success_threshold
        wrong_object = dist_to_wrong < self.cfg.success_threshold
        fell = (drone_pos[:, 2] < 0.1) | (drone_pos[:, 2] > 3.0)

        terminated = success | wrong_object | fell
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        extras: dict = {}
        for key in self._episode_sums:
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Curriculum/nav_multiplier"] = self._get_nav_multiplier()
        extras["Curriculum/precision_scale"] = self._get_precision_scale()

        # Termination breakdown: why did episodes end?
        drone_pos = self._robot.data.root_pos_w[env_ids]
        target_pos = self._obj_pos_w[env_ids, self._target_obj_idx[env_ids]]
        dist = torch.linalg.norm(target_pos - drone_pos, dim=1)

        wrong_mask = torch.ones(len(env_ids), 3, dtype=torch.bool, device=self.device)
        wrong_mask[torch.arange(len(env_ids), device=self.device), self._target_obj_idx[env_ids]] = False
        wrong_obj_pos = self._obj_pos_w[env_ids][wrong_mask].reshape(len(env_ids), 2, 3)
        dist_to_wrong = torch.linalg.norm(wrong_obj_pos - drone_pos.unsqueeze(1), dim=-1).min(dim=1).values

        success_term = (dist < self.cfg.success_threshold).float().mean()
        wrong_term = (dist_to_wrong < self.cfg.success_threshold).float().mean()
        fell_term = ((drone_pos[:, 2] < 0.1) | (drone_pos[:, 2] > 3.0)).float().mean()
        timeout_term = (self.episode_length_buf[env_ids] >= self.max_episode_length - 1).float().mean()

        extras["Termination/success_rate"] = success_term
        extras["Termination/wrong_object_rate"] = wrong_term
        extras["Termination/fell_rate"] = fell_term
        extras["Termination/timeout_rate"] = timeout_term

        # Distance diagnostics (at episode end)
        extras["Diagnostic/final_dist_to_target"] = dist.mean()
        extras["Diagnostic/final_dist_to_wrong"] = dist_to_wrong.mean()
        extras["Diagnostic/mean_speed"] = torch.linalg.norm(self._robot.data.root_lin_vel_w[env_ids], dim=1).mean()
        extras["Diagnostic/mean_hover_dwell"] = self._hover_dwell[env_ids].mean()

        self.extras["log"] = extras

        self._log_step += 1
        if self._log_step % 50 == 0:
            row = {k: float(v) if isinstance(v, (int, float)) else float(v.item()) for k, v in extras.items()}
            row["step"] = self.common_step_counter
            self._metrics_file.write(json.dumps(row) + "\n")
            self._metrics_file.flush()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0

        # Reset robot kinematics
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Randomize object positions within the arena
        n = len(env_ids)
        env_origins = self._terrain.env_origins[env_ids]  # (n, 3)
        min_sep = 0.8  # minimum separation between objects

        obj_positions = []  # 3 tensors of (n, 3)
        for obj_idx in range(3):
            for _ in range(100):  # rejection sampling with max attempts
                xy = (torch.rand(n, 2, device=self.device) - 0.5) * 3.5  # [-1.75, 1.75]
                pos = torch.cat([xy, torch.full((n, 1), 0.2, device=self.device)], dim=-1)
                if obj_idx == 0:
                    break
                # Check min distance to all previously placed objects
                ok = torch.ones(n, dtype=torch.bool, device=self.device)
                for prev in obj_positions:
                    dist = torch.linalg.norm(pos[:, :2] - prev[:, :2], dim=1)
                    ok &= dist > min_sep
                if ok.all():
                    break
                # For envs that failed, resample only those
                pos[~ok] = obj_positions[0][~ok]  # fallback to avoid infinite loop
            obj_positions.append(pos)

        # World positions = env_origin + local offset
        cube_world = env_origins + obj_positions[0]
        sphere_world = env_origins + obj_positions[1]
        cylinder_world = env_origins + obj_positions[2]

        # Reposition the visual prims
        self._cube_view.set_world_poses(positions=cube_world, indices=env_ids)
        self._sphere_view.set_world_poses(positions=sphere_world, indices=env_ids)
        self._cylinder_view.set_world_poses(positions=cylinder_world, indices=env_ids)

        # Update cached positions for reward computation
        self._obj_pos_w[env_ids, 0] = cube_world
        self._obj_pos_w[env_ids, 1] = sphere_world
        self._obj_pos_w[env_ids, 2] = cylinder_world

        # Sample new commands and tokenize for PaliGemma
        n = len(env_ids)
        obj_type_indices = torch.randint(0, len(OBJECT_TYPES), (n,))

        commands = [
            random.choice(COMMANDS[OBJECT_TYPES[i.item()]])
            for i in obj_type_indices
        ]

        # Tokenize with PaliGemma processor (prepends 256 image placeholder tokens)
        # Add <image> prefix so PaliGemma knows where to inject visual features
        prefixed_commands = ["\n" + cmd for cmd in commands]
        tokenized = self._processor.tokenizer(
            prefixed_commands,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_text_length - _NUM_IMAGE_TOKENS,
        )
        # Prepend 256 image tokens to input_ids and attention_mask
        batch_size = len(commands)
        img_tokens = torch.full((batch_size, _NUM_IMAGE_TOKENS), self._image_token_id, dtype=torch.long)
        img_mask = torch.ones(batch_size, _NUM_IMAGE_TOKENS, dtype=torch.long)

        full_ids = torch.cat([img_tokens, tokenized["input_ids"]], dim=1)
        full_mask = torch.cat([img_mask, tokenized["attention_mask"]], dim=1)

        self._text_tokens[env_ids] = full_ids.to(self.device)
        self._text_mask[env_ids] = full_mask.to(self.device)
        self._target_obj_idx[env_ids] = obj_type_indices.to(self.device)
        for i, eid in enumerate(env_ids):
            self._current_commands[int(eid)] = commands[i]

        # Force camera capture on next step
        self._steps_since_capture[env_ids] = self.cfg.camera_every_n

        # Reset hover dwell counter
        self._hover_dwell[env_ids] = 0.0

"""Cesium-world VLA drone navigation environment.

Subclasses vla.vla_drone_env.VLADroneEnv to stream real-world 3D Tiles via
Cesium for Omniverse. The drone, cameras, observation pipeline, policy
interface, and PPO training loop are unchanged — only scene construction
and target/reward geometry change:

  * Scene: replaces the flat ground + 3 hand-placed shapes with a Google
    Photorealistic 3D Tiles tileset of a real city, centered at a chosen
    lat/lon. The 3 cube/sphere/cylinder markers remain (they're what the
    reward machinery tracks) and are repositioned at reset to real-world
    POI locations from pois.CITY_BANKS.
  * Targets: each episode picks `num_active_pois` POIs. One is the target
    (its name is substituted into the language prompt); the rest are
    distractors (reaching them triggers wrong_object_penalty).
  * Reward: same terms as parent, but with a larger tanh scale in the
    distance-shaping term so the gradient reaches across a city.
  * Spawn: drone starts at ~80m AGL above the georef origin with XY noise.
  * Altitude bounds: expanded to a city-scale airspace (20m–300m).

Aliased exports (`VLADroneEnvCfg`, `VLADroneEnv`) let train.py/play.py
swap this in via sys.modules patching without duplicating their code.
"""

from __future__ import annotations

import os
import random

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from vla.vla_drone_env import (
    VLADroneEnv as _BaseVLADroneEnv,
    VLADroneEnvCfg as _BaseVLADroneEnvCfg,
    _NUM_IMAGE_TOKENS,
)

from vla_cesium import pois as _pois_mod
from vla_cesium.cesium_setup import (
    load_cesium_world,
    pois_to_enu_tensor,
    ASSET_GOOGLE_PHOTOREALISTIC,
)


@configclass
class VLACesiumDroneEnvCfg(_BaseVLADroneEnvCfg):
    # --------------------------- Cesium world ---------------------------
    city: str = "manhattan"                          # key into CITY_BANKS
    cesium_ion_token_env_var: str = "CESIUM_ION_TOKEN"
    cesium_ion_asset_id: int = ASSET_GOOGLE_PHOTOREALISTIC

    # Derived fields (overridden at runtime from pois.CITY_BANKS[city])
    georef_origin_lat: float = 40.7580
    georef_origin_lon: float = -73.9855
    georef_origin_alt: float = 10.0

    # --------------------------- Episode & arena ---------------------------
    # City-scale flight takes longer than the 2.5m arena
    episode_length_s = 60.0

    # Drone spawn pose (local ENU around georef origin)
    spawn_altitude: float = 80.0
    spawn_altitude_jitter: float = 20.0
    spawn_xy_radius: float = 30.0

    # POI selection
    num_active_pois: int = 3          # matches the 3 marker objects (cube/sphere/cyl)
    poi_visible_radius: float = 400.0 # only POIs within this horizontal radius eligible

    # --------------------------- Overridden reward knobs ---------------------------
    # Characteristic distance for tanh shaping — tuned for city-scale (was 0.8m)
    distance_tanh_scale: float = 40.0

    # Success/proximity at city scale
    success_threshold = 8.0           # was 0.35m
    proximity_radius = 25.0           # was 1.5m
    hover_at_target_radius = 10.0     # was 0.5m
    hover_radius_start = 10.0
    hover_radius_end = 10.0
    hover_max_speed = 3.0             # was 1.0 m/s

    # Altitude envelope in city airspace
    altitude_warning_low = 20.0       # was 0.3m
    altitude_warning_high = 300.0     # was 2.8m

    # Termination altitudes (parent uses hardcoded 0.1 / 3.0 — we override in _get_dones)
    terminate_altitude_low: float = 5.0
    terminate_altitude_high: float = 400.0


class VLACesiumDroneEnv(_BaseVLADroneEnv):
    """VLA drone env with Cesium 3D Tiles as the world."""

    cfg: VLACesiumDroneEnvCfg

    # ---------------------------------------------------------------
    # Init — resolve city bank before super().__init__ calls _setup_scene
    # ---------------------------------------------------------------
    def __init__(self, cfg: VLACesiumDroneEnvCfg, render_mode: str | None = None, **kwargs):
        # Resolve the city bank and patch the origin lat/lon into cfg BEFORE
        # _setup_scene runs (super().__init__ calls it).
        city_entry = _pois_mod.get_city(cfg.city)
        cfg.georef_origin_lat, cfg.georef_origin_lon, cfg.georef_origin_alt = city_entry["origin"]
        self._poi_bank = city_entry["pois"]
        # Held as python list; tensorized at end of __init__ once device is known
        super().__init__(cfg, render_mode, **kwargs)

        # Per-env POI selection: (num_envs, num_active_pois) indices into _poi_bank
        self._active_poi_idx = torch.zeros(
            self.num_envs, cfg.num_active_pois, dtype=torch.long, device=self.device
        )

        # World ENU positions of every POI in the bank — recomputed here so the
        # flat-earth approximation uses the same origin as _setup_scene.
        self._poi_enu = pois_to_enu_tensor(
            self._poi_bank,
            cfg.georef_origin_lat, cfg.georef_origin_lon, cfg.georef_origin_alt,
            device=self.device,
        )  # (num_pois, 3)

        # Kick a reset now that POI state exists. Parent already called one,
        # but it hit the legacy shape-placement path. Redo under POI logic.
        self._reset_idx(None)

    # ---------------------------------------------------------------
    # Scene: same cube/sphere/cylinder markers + Cesium tileset
    # ---------------------------------------------------------------
    def _setup_scene(self):
        """Replicate parent's scene but (a) keep ground plane hidden below
        the city, (b) spawn 3 large POI markers instead of tiny shapes, and
        (c) load the Cesium tileset after cloning."""
        from isaaclab.assets import Articulation
        from isaaclab.sim.views import XformPrimView

        self._robot = Articulation(self.cfg.robot)
        self._cameras = [
            TiledCamera(self.cfg.cam_front),
            TiledCamera(self.cfg.cam_right),
            TiledCamera(self.cfg.cam_back),
            TiledCamera(self.cfg.cam_left),
        ]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Bigger markers — cube/sphere/cylinder as 6m objects so they're visible
        # from 80m altitude. These stand in for real POIs; repositioned at reset.
        marker_offset = (0.0, 0.0, 0.0)
        cube_cfg = sim_utils.CuboidCfg(
            size=(6.0, 6.0, 6.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.15, 0.15),
                emissive_color=(0.5, 0.05, 0.05),
            ),
        )
        cube_cfg.func("/World/envs/env_.*/cube", cube_cfg, translation=marker_offset)

        sphere_cfg = sim_utils.SphereCfg(
            radius=3.5,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.25, 0.95),
                emissive_color=(0.05, 0.08, 0.4),
            ),
        )
        sphere_cfg.func("/World/envs/env_.*/sphere", sphere_cfg, translation=marker_offset)

        cylinder_cfg = sim_utils.CylinderCfg(
            radius=3.0, height=8.0,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.9, 0.2),
                emissive_color=(0.05, 0.4, 0.08),
            ),
        )
        cylinder_cfg.func("/World/envs/env_.*/cylinder", cylinder_cfg, translation=marker_offset)

        self.scene.clone_environments(copy_from_source=False)

        self._cube_view = XformPrimView("/World/envs/env_.*/cube", device=self.device)
        self._sphere_view = XformPrimView("/World/envs/env_.*/sphere", device=self.device)
        self._cylinder_view = XformPrimView("/World/envs/env_.*/cylinder", device=self.device)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self._robot
        for i, cam in enumerate(self._cameras):
            self.scene.sensors[f"tiled_camera_{i}"] = cam

        # Lights — keep parent's sky HDR; Cesium tiles have their own textures
        light_cfg = sim_utils.DomeLightCfg(
            intensity=1200.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg)
        dist_light = sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 0.95, 0.85))
        dist_light.func("/World/SunLight", dist_light)

        self._target_marker = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/World/Visuals/target_marker",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=2.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.85, 0.0),
                        emissive_color=(1.0, 0.85, 0.0),
                        opacity=0.85,
                    ),
                ),
            },
        ))

        # --- Cesium 3D Tiles (this is the new part) -----------------------------
        load_cesium_world(
            lat=self.cfg.georef_origin_lat,
            lon=self.cfg.georef_origin_lon,
            alt=self.cfg.georef_origin_alt,
            ion_asset_id=self.cfg.cesium_ion_asset_id,
            ion_token_env=self.cfg.cesium_ion_token_env_var,
        )

    # ---------------------------------------------------------------
    # Reset — pick POIs, place markers at real-world coords, build prompt
    # ---------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Episode-length randomization (copied from parent for determinism)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Logging — same shape as parent
        extras: dict = {}
        for key in self._episode_sums:
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Curriculum/nav_multiplier"] = self._get_nav_multiplier()
        extras["Curriculum/precision_scale"] = self._get_precision_scale()
        self.extras.setdefault("log", {}).update(extras)

        self._actions[env_ids] = 0.0

        # --- Drone spawn: random XY within spawn_xy_radius, altitude with jitter ---
        n = len(env_ids)
        env_origins = self._terrain.env_origins[env_ids]  # (n, 3)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        theta = torch.rand(n, device=self.device) * (2 * torch.pi)
        radius = torch.sqrt(torch.rand(n, device=self.device)) * self.cfg.spawn_xy_radius
        spawn_xy = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
        spawn_z = self.cfg.spawn_altitude + (torch.rand(n, device=self.device) - 0.5) * 2 * self.cfg.spawn_altitude_jitter

        default_root_state[:, 0] = env_origins[:, 0] + spawn_xy[:, 0]
        default_root_state[:, 1] = env_origins[:, 1] + spawn_xy[:, 1]
        default_root_state[:, 2] = env_origins[:, 2] + spawn_z
        # Zero velocities
        default_root_state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- POI selection per env ---
        num_pois = len(self._poi_bank)
        active = self.cfg.num_active_pois
        if num_pois < active:
            raise RuntimeError(
                f"City '{self.cfg.city}' has {num_pois} POIs but num_active_pois={active}. "
                f"Add more entries to vla_cesium/pois.py."
            )
        # Sample `active` distinct POI indices per env
        picks = torch.stack([
            torch.randperm(num_pois, device=self.device)[:active] for _ in range(n)
        ], dim=0)  # (n, active)
        self._active_poi_idx[env_ids] = picks

        # Place the 3 markers at the 3 active POIs' world positions
        # Marker positions are LOCAL to env (set_local_poses on per-env parent),
        # so we pass ENU-relative-to-env-origin. But our POIs are in ENU relative
        # to georef origin, which coincides with env_origin of env 0. For envs at
        # different env_origins, place POIs in stage-global (world) frame, which
        # means local_pos = world_pos - env_origin.
        poi_world = self._poi_enu[picks]  # (n, active, 3)
        poi_local = poi_world - env_origins.unsqueeze(1)  # (n, active, 3)

        env_ids_list = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        self._cube_view.set_local_poses(translations=poi_local[:, 0], indices=env_ids_list)
        self._sphere_view.set_local_poses(translations=poi_local[:, 1], indices=env_ids_list)
        self._cylinder_view.set_local_poses(translations=poi_local[:, 2], indices=env_ids_list)

        # Cache world positions for reward (parent's reward logic reads _obj_pos_w)
        self._obj_pos_w[env_ids, 0] = poi_world[:, 0]
        self._obj_pos_w[env_ids, 1] = poi_world[:, 1]
        self._obj_pos_w[env_ids, 2] = poi_world[:, 2]

        # --- Target POI + language prompt ---
        target_slot = torch.randint(0, active, (n,), device=self.device)  # which of the 3 markers is target
        self._target_obj_idx[env_ids] = target_slot

        # Build commands from target POI's prompt bank
        commands = []
        target_pois_cpu = picks.gather(1, target_slot.unsqueeze(1)).squeeze(1).cpu().tolist()
        for poi_idx in target_pois_cpu:
            poi = self._poi_bank[poi_idx]
            commands.append(random.choice(poi.prompts))

        # Tokenize with PaliGemma processor (identical to parent)
        prefixed_commands = ["\n" + cmd for cmd in commands]
        tokenized = self._processor.tokenizer(
            prefixed_commands,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_text_length - _NUM_IMAGE_TOKENS,
        )
        batch_size = len(commands)
        img_tokens = torch.full((batch_size, _NUM_IMAGE_TOKENS), self._image_token_id, dtype=torch.long)
        img_mask = torch.ones(batch_size, _NUM_IMAGE_TOKENS, dtype=torch.long)
        full_ids = torch.cat([img_tokens, tokenized["input_ids"]], dim=1)
        full_mask = torch.cat([img_mask, tokenized["attention_mask"]], dim=1)
        self._text_tokens[env_ids] = full_ids.to(self.device)
        self._text_mask[env_ids] = full_mask.to(self.device)

        for i, eid in enumerate(env_ids):
            self._current_commands[int(eid)] = commands[i]

        self._steps_since_capture[env_ids] = self.cfg.camera_every_n
        self._hover_dwell[env_ids] = 0.0

    # ---------------------------------------------------------------
    # Reward — override distance tanh scale for city-scale gradient
    # ---------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        drone_pos = self._robot.data.root_pos_w

        target_pos = self._obj_pos_w[env_ids, self._target_obj_idx]
        to_goal = target_pos - drone_pos
        dist_to_target = torch.linalg.norm(to_goal, dim=1)
        # City-scale tanh scale (cfg.distance_tanh_scale, default 40m)
        dist_mapped = 1.0 - torch.tanh(dist_to_target / self.cfg.distance_tanh_scale)

        to_goal_dir = to_goal / dist_to_target.unsqueeze(1).clamp(min=0.01)
        vel_toward = torch.sum(self._robot.data.root_lin_vel_w * to_goal_dir, dim=1)
        # "Far" threshold scales with distance_tanh_scale (was 1.0m for 0.8m scale)
        far_threshold = self.cfg.distance_tanh_scale * 0.25
        far_from_target = (dist_to_target > far_threshold).float()
        # Max approach speed scales too (was 2.0 m/s for 2.5m arena)
        vel_cap = max(self.cfg.distance_tanh_scale * 0.25, 3.0)
        vel_toward_clamp = torch.clamp(vel_toward, 0.0, vel_cap) * far_from_target

        inside_radius = (dist_to_target < self.cfg.proximity_radius).float()
        proximity = inside_radius * (1.0 - dist_to_target / self.cfg.proximity_radius)

        precision = self._get_precision_scale()
        hover_radius = self.cfg.hover_radius_start + precision * (
            self.cfg.hover_radius_end - self.cfg.hover_radius_start
        )
        near_target = (dist_to_target < hover_radius)
        speed = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1)
        slow = (speed < self.cfg.hover_max_speed)
        hovering = near_target & slow

        self._hover_dwell = torch.where(
            hovering,
            (self._hover_dwell + 1.0).clamp(max=float(self.cfg.hover_max_dwell_steps)),
            torch.zeros_like(self._hover_dwell),
        )
        dwell_ratio = self._hover_dwell / float(self.cfg.hover_max_dwell_steps)
        dwell_multiplier = 1.0 + self.cfg.hover_dwell_bonus * dwell_ratio
        hover_at_target = hovering.float() * dwell_multiplier

        uprightness = -self._robot.data.projected_gravity_b[:, 2]

        too_low = torch.clamp(self.cfg.altitude_warning_low - drone_pos[:, 2], min=0.0)
        too_high = torch.clamp(drone_pos[:, 2] - self.cfg.altitude_warning_high, min=0.0)
        altitude_penalty = (too_low + too_high) * self.cfg.crash_penalty_scale

        # Wrong-object: distance to any of the other active POIs
        n_active = self.cfg.num_active_pois
        wrong_mask = torch.ones(self.num_envs, n_active, dtype=torch.bool, device=self.device)
        wrong_mask[env_ids, self._target_obj_idx] = False
        wrong_obj_pos = self._obj_pos_w[wrong_mask].reshape(self.num_envs, n_active - 1, 3)
        dist_to_wrong = torch.linalg.norm(
            wrong_obj_pos - drone_pos.unsqueeze(1), dim=-1
        ).min(dim=1).values

        success = dist_to_target < self.cfg.success_threshold
        wrong_object = dist_to_wrong < self.cfg.success_threshold

        nav = self._get_nav_multiplier()
        alive_scale = self._get_alive_scale()
        loose_scale = 1.0 - 0.8 * precision
        precise_scale = 1.0 + 2.0 * precision

        rewards = {
            "lin_vel": torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
                       * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
                       * self.cfg.ang_vel_reward_scale * self.step_dt,
            "alive": torch.ones(self.num_envs, device=self.device)
                     * self.cfg.alive_reward * alive_scale * self.step_dt,
            "uprightness": uprightness * self.cfg.uprightness_reward_scale * self.step_dt,
            "altitude_penalty": altitude_penalty * self.step_dt,
            "distance_to_goal": dist_mapped * self.cfg.distance_to_goal_reward_scale
                                * self.step_dt * nav * loose_scale,
            "velocity_toward_goal": vel_toward_clamp * self.cfg.velocity_toward_goal_scale
                                    * self.step_dt * nav * loose_scale,
            "proximity": proximity * self.cfg.proximity_scale * self.step_dt * nav * loose_scale,
            "hover_at_target": hover_at_target * self.cfg.hover_at_target_reward
                               * self.step_dt * nav * precise_scale,
            "success": success.float() * self.cfg.success_reward * nav * precise_scale,
            "wrong_object": wrong_object.float() * self.cfg.wrong_object_penalty * nav * precise_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v
        return reward

    # ---------------------------------------------------------------
    # Termination — city-scale altitude bounds
    # ---------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = torch.arange(self.num_envs, device=self.device)
        drone_pos = self._robot.data.root_pos_w

        target_pos = self._obj_pos_w[env_ids, self._target_obj_idx]
        dist_to_target = torch.linalg.norm(target_pos - drone_pos, dim=1)

        n_active = self.cfg.num_active_pois
        wrong_mask = torch.ones(self.num_envs, n_active, dtype=torch.bool, device=self.device)
        wrong_mask[env_ids, self._target_obj_idx] = False
        wrong_obj_pos = self._obj_pos_w[wrong_mask].reshape(self.num_envs, n_active - 1, 3)
        dist_to_wrong = torch.linalg.norm(
            wrong_obj_pos - drone_pos.unsqueeze(1), dim=-1
        ).min(dim=1).values

        success = dist_to_target < self.cfg.success_threshold
        wrong_object = dist_to_wrong < self.cfg.success_threshold
        # Altitude relative to env origin (which == georef origin for env 0)
        drone_altitude = drone_pos[:, 2] - self._terrain.env_origins[:, 2]
        fell = (drone_altitude < self.cfg.terminate_altitude_low) | (drone_altitude > self.cfg.terminate_altitude_high)

        terminated = success | wrong_object | fell
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out


# -------- sys.modules-patch aliases --------------------------------
# Train/play scripts do `from vla.vla_drone_env import VLADroneEnvCfg`; we
# inject this module in place of that name, so they pick up the Cesium variants.
VLADroneEnvCfg = VLACesiumDroneEnvCfg
VLADroneEnv = VLACesiumDroneEnv

"""VLA drone navigation in Isaac Sim's sample environments (warehouse etc.).

Subclasses vla.vla_drone_env.VLADroneEnv. The drone, cameras, observation
pipeline, policy interface, and PPO training loop are unchanged — only
scene construction, reward scale, and target sampling differ:

  * Scene: replaces the flat ground + 3 hand-placed shapes with a rich USD
    scene (Simple_Warehouse / Hospital / Office) cloned per env. The 3
    cube/sphere/cylinder markers stay (they're what the reward machinery
    tracks) and get repositioned at reset onto real POI locations inside
    the scene — forklifts, shelves, pallets, beds, desks, etc.
  * Targets: each episode picks `num_active_pois` POIs from the scene's
    bank in pois.py. One is the target (its prompt becomes the language
    command); the rest are distractors (reaching them triggers
    wrong_object_penalty).
  * Reward: same terms as parent, but with knobs tuned for a ~20m indoor
    arena instead of the parent's 2.5m cube.

The parent's checkpoint loader tolerates shape mismatches, so your existing
vla/ weights warm-start here unchanged.
"""

from __future__ import annotations

import random

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.scene import InteractiveSceneCfg

from vla.vla_drone_env import (
    VLADroneEnv as _BaseVLADroneEnv,
    VLADroneEnvCfg as _BaseVLADroneEnvCfg,
    _NUM_IMAGE_TOKENS,
)

from vla_warehouse import pois as _pois_mod
from vla_warehouse.scene_setup import load_scene


@configclass
class VLAWarehouseDroneEnvCfg(_BaseVLADroneEnvCfg):
    # --------------------------- Scene selection ---------------------------
    # NOTE: "warehouse" (warehouse.usd) is typically just an empty floor/walls
    # shell with no props. "warehouse_full" or "warehouse_shelves" are the
    # populated variants with forklifts / racks / pallets.
    scene_name: str = "warehouse_full"   # key into pois.SCENES

    # --------------------------- Episode ---------------------------
    episode_length_s = 30.0

    # --------------------------- Drone spawn ---------------------------
    spawn_altitude: float = 2.5        # mid-height in warehouse
    spawn_altitude_jitter: float = 0.8
    spawn_xy_radius: float = 3.0       # random horizontal offset from env origin

    # --------------------------- POI selection ---------------------------
    num_active_pois: int = 3           # matches the 3 marker objects

    # --------------------------- Reward knobs (indoor-scale) ---------------------------
    distance_tanh_scale: float = 5.0   # characteristic scale for distance shaping
    success_threshold = 1.0            # within 1m of POI
    proximity_radius = 3.0
    hover_at_target_radius = 1.5
    hover_radius_start = 1.5
    hover_radius_end = 1.5
    hover_max_speed = 0.8

    # Altitude envelope — warehouse ceiling is ~6m
    altitude_warning_low = 0.3
    altitude_warning_high = 5.5
    terminate_altitude_low: float = 0.15
    terminate_altitude_high: float = 6.5

    # Override the InteractiveSceneCfg default env_spacing so cloned
    # warehouses don't overlap. Resolved from pois.SCENES at init time.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=40.0, replicate_physics=True,
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Widen camera clip range from parent's (0.01, 20) — warehouse walls
        # on the far side are up to ~35m away.
        wide_clip = sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=100.0,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 50.0),
        )
        for cam in (self.cam_front, self.cam_right, self.cam_back, self.cam_left):
            cam.spawn = wide_clip


class VLAWarehouseDroneEnv(_BaseVLADroneEnv):
    cfg: VLAWarehouseDroneEnvCfg

    def __init__(self, cfg: VLAWarehouseDroneEnvCfg, render_mode: str | None = None, **kwargs):
        # Resolve scene bank and sync env_spacing BEFORE super().__init__ runs
        scene_entry = _pois_mod.get_scene(cfg.scene_name)
        self._scene_entry = scene_entry
        self._poi_bank = scene_entry["pois"]
        self._scene_usd_relpath = scene_entry["usd_path"]
        # Match env_spacing to scene size so cloned warehouses don't overlap
        cfg.scene.env_spacing = max(cfg.scene.env_spacing, scene_entry["env_spacing"])

        super().__init__(cfg, render_mode, **kwargs)

        # Per-env active POI indices: (num_envs, num_active_pois)
        self._active_poi_idx = torch.zeros(
            self.num_envs, cfg.num_active_pois, dtype=torch.long, device=self.device
        )

        # POI positions in local-env frame (meters)
        self._poi_local = torch.tensor(
            [[p.x, p.y, p.z] for p in self._poi_bank],
            dtype=torch.float32, device=self.device,
        )  # (num_pois, 3)

        # Fresh reset under POI logic
        self._reset_idx(None)

    # ---------------------------------------------------------------
    # Scene setup — per-env warehouse clone + marker objects
    # ---------------------------------------------------------------
    def _setup_scene(self):
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

        # Load the USD scene per env (warehouse/hospital/office)
        load_scene(
            self._scene_usd_relpath,
            prim_regex="/World/envs/env_.*/Scene",
            translation=(0.0, 0.0, 0.0),
        )

        # Marker objects — same as parent but slightly larger (easier to see
        # in a cluttered warehouse) and emissive so they read against props.
        marker_offset = (0.0, 0.0, 1.5)
        cube_cfg = sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.15, 0.15),
                emissive_color=(0.6, 0.05, 0.05),
            ),
        )
        cube_cfg.func("/World/envs/env_.*/cube", cube_cfg, translation=marker_offset)

        sphere_cfg = sim_utils.SphereCfg(
            radius=0.35,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.25, 0.95),
                emissive_color=(0.05, 0.1, 0.5),
            ),
        )
        sphere_cfg.func("/World/envs/env_.*/sphere", sphere_cfg, translation=marker_offset)

        cylinder_cfg = sim_utils.CylinderCfg(
            radius=0.3, height=0.8,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.9, 0.2),
                emissive_color=(0.05, 0.5, 0.1),
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

        # Lights — the warehouse USD carries its own lights but a dome adds fill
        light_cfg = sim_utils.DomeLightCfg(
            intensity=800.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg)

        self._target_marker = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/World/Visuals/target_marker",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.25,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.85, 0.0),
                        emissive_color=(1.0, 0.85, 0.0),
                        opacity=0.85,
                    ),
                ),
            },
        ))

    # ---------------------------------------------------------------
    # Reset — pick POIs, place markers, build prompt, respawn drone
    # ---------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # `super().__init__` invokes this before our child-specific state
        # (_active_poi_idx, _poi_local) exists. Skip and rely on the
        # explicit _reset_idx call at the end of our __init__.
        if not hasattr(self, "_active_poi_idx"):
            return
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Logging — match parent's shape
        extras: dict = {}
        for key in self._episode_sums:
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Curriculum/nav_multiplier"] = self._get_nav_multiplier()
        extras["Curriculum/precision_scale"] = self._get_precision_scale()
        self.extras.setdefault("log", {}).update(extras)

        self._actions[env_ids] = 0.0

        # --- Drone spawn pose ---
        n = len(env_ids)
        env_origins = self._terrain.env_origins[env_ids]  # (n, 3)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        theta = torch.rand(n, device=self.device) * (2 * torch.pi)
        radius = torch.sqrt(torch.rand(n, device=self.device)) * self.cfg.spawn_xy_radius
        spawn_xy = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
        spawn_z = self.cfg.spawn_altitude + (
            torch.rand(n, device=self.device) - 0.5
        ) * 2 * self.cfg.spawn_altitude_jitter

        default_root_state[:, 0] = env_origins[:, 0] + spawn_xy[:, 0]
        default_root_state[:, 1] = env_origins[:, 1] + spawn_xy[:, 1]
        default_root_state[:, 2] = env_origins[:, 2] + spawn_z
        default_root_state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- POI selection per env ---
        num_pois = len(self._poi_bank)
        active = self.cfg.num_active_pois
        if num_pois < active:
            raise RuntimeError(
                f"Scene '{self.cfg.scene_name}' has {num_pois} POIs but "
                f"num_active_pois={active}. Add more entries to vla_warehouse/pois.py."
            )
        picks = torch.stack([
            torch.randperm(num_pois, device=self.device)[:active] for _ in range(n)
        ], dim=0)  # (n, active)
        self._active_poi_idx[env_ids] = picks

        # Place markers at picked POIs' local coords (world = env_origin + local)
        poi_local = self._poi_local[picks]                    # (n, active, 3)
        poi_world = env_origins.unsqueeze(1) + poi_local      # (n, active, 3)

        env_ids_list = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        self._cube_view.set_local_poses(translations=poi_local[:, 0], indices=env_ids_list)
        self._sphere_view.set_local_poses(translations=poi_local[:, 1], indices=env_ids_list)
        self._cylinder_view.set_local_poses(translations=poi_local[:, 2], indices=env_ids_list)

        self._obj_pos_w[env_ids, 0] = poi_world[:, 0]
        self._obj_pos_w[env_ids, 1] = poi_world[:, 1]
        self._obj_pos_w[env_ids, 2] = poi_world[:, 2]

        # --- Target POI + language prompt ---
        target_slot = torch.randint(0, active, (n,), device=self.device)
        self._target_obj_idx[env_ids] = target_slot

        target_pois_cpu = picks.gather(1, target_slot.unsqueeze(1)).squeeze(1).cpu().tolist()
        commands = [random.choice(self._poi_bank[p].prompts) for p in target_pois_cpu]

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
    # Reward — override for indoor distance scale
    # ---------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        drone_pos = self._robot.data.root_pos_w

        target_pos = self._obj_pos_w[env_ids, self._target_obj_idx]
        to_goal = target_pos - drone_pos
        dist_to_target = torch.linalg.norm(to_goal, dim=1)
        dist_mapped = 1.0 - torch.tanh(dist_to_target / self.cfg.distance_tanh_scale)

        to_goal_dir = to_goal / dist_to_target.unsqueeze(1).clamp(min=0.01)
        vel_toward = torch.sum(self._robot.data.root_lin_vel_w * to_goal_dir, dim=1)
        far_threshold = self.cfg.distance_tanh_scale * 0.2
        far_from_target = (dist_to_target > far_threshold).float()
        vel_cap = max(self.cfg.distance_tanh_scale * 0.2, 1.5)
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

        # Altitude relative to env origin (floor of this env's warehouse)
        alt_rel = drone_pos[:, 2] - self._terrain.env_origins[:, 2]
        too_low = torch.clamp(self.cfg.altitude_warning_low - alt_rel, min=0.0)
        too_high = torch.clamp(alt_rel - self.cfg.altitude_warning_high, min=0.0)
        altitude_penalty = (too_low + too_high) * self.cfg.crash_penalty_scale

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
    # Termination — warehouse-scale altitude bounds
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
        alt_rel = drone_pos[:, 2] - self._terrain.env_origins[:, 2]
        fell = (alt_rel < self.cfg.terminate_altitude_low) | (alt_rel > self.cfg.terminate_altitude_high)

        terminated = success | wrong_object | fell
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out


# sys.modules-patch aliases so train.py/play.py shims work
VLADroneEnvCfg = VLAWarehouseDroneEnvCfg
VLADroneEnv = VLAWarehouseDroneEnv

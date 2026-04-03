"""Language-and-vision-grounded drone navigation environment (VLA).

The drone receives a natural language command ("go to the square") and an
onboard camera image, and must navigate to the matching geometric object
in the scene. Three colored objects are placed at fixed positions:
  - Cube   (red)   at offset (-1.5,  0.0, 0.2) → index 0
  - Sphere (blue)  at offset ( 1.5,  0.0, 0.2) → index 1
  - Cylinder (green) at offset (0.0, 1.5, 0.2) → index 2

Observation (1033-dim):
  [0:3]     root_lin_vel_b
  [3:6]     root_ang_vel_b
  [6:9]     projected_gravity_b
  [9:521]   CLIP text embedding of the command (512-dim, frozen)
  [521:1033] CLIP image embedding from onboard camera (512-dim, frozen)

Reward (two-phase: stability always active, navigation fades in):
  Stability (always): alive bonus, uprightness, altitude penalty,
      lin/ang velocity penalties
  Navigation (gated): shaped distance-to-target, velocity-toward-goal,
      proximity bonus, +10 success / -3 wrong-object
"""

from __future__ import annotations

import random

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip

from .clip_grounder import CLIPGrounder
from .commands import COMMANDS, OBJECT_TYPES

# Fixed XYZ offsets for each object relative to env origin (cube, sphere, cylinder)
_OBJ_OFFSETS = [(-1.5, 0.0, 0.2), (1.5, 0.0, 0.2), (0.0, 1.5, 0.2)]

# CLIP embedding dimension (clip-vit-base-patch32)
_CLIP_DIM = 512


@configclass
class LangDroneEnvCfg(DirectRLEnvCfg):
    # Episode / stepping
    episode_length_s = 15.0
    decimation = 2
    action_space = 4
    # 9 drone state + 512 CLIP text + 512 CLIP image
    observation_space = 9 + _CLIP_DIM + _CLIP_DIM
    state_space = 0
    debug_vis = False

    # Simulation
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

    # Larger env spacing so objects don't overlap between envs
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=6.0, replicate_physics=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Onboard camera (64x64, resized to 224x224 for CLIP)
    # Spawned at env level; pose updated each step to follow drone body
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=100.0,
            horizontal_aperture=15.0,
            clipping_range=(0.01, 20.0),
        ),
        width=64,
        height=64,
    )
    # Camera offset from drone body centre (forward + slightly up)
    camera_body_offset_pos = (0.05, 0.0, 0.01)

    # How often to run CLIP image encoding (every N physics steps)
    clip_encode_every_n = 5

    # Reward scales — stability (always active)
    lin_vel_reward_scale = -0.02
    ang_vel_reward_scale = -0.005
    alive_reward = 1.5
    uprightness_reward_scale = 0.5
    altitude_warning_low = 0.3
    altitude_warning_high = 2.8
    crash_penalty_scale = -10.0

    # Reward scales — navigation (gated by nav_multiplier during warmup)
    distance_to_goal_reward_scale = 25.0
    velocity_toward_goal_scale = 4.0
    proximity_scale = 8.0
    proximity_radius = 1.5
    success_reward = 10.0
    wrong_object_penalty = -3.0

    # Task parameters
    success_threshold = 0.35  # metres — drone must get this close to target

    # Two-phase learning: stabilise hover first, then fade in navigation
    survival_only_steps = 3000
    nav_fadein_steps = 10000


class LangDroneEnv(DirectRLEnv):
    cfg: LangDroneEnvCfg

    def __init__(self, cfg: LangDroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action / thrust buffers
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Per-env language state
        self._target_obj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._clip_emb = torch.zeros(self.num_envs, _CLIP_DIM, device=self.device)
        self._current_commands: list[str] = [""] * self.num_envs

        # Per-env vision state (cached CLIP image embedding)
        self._clip_img_emb = torch.zeros(self.num_envs, _CLIP_DIM, device=self.device)
        self._steps_since_encode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Camera body-frame offset (constant, used to position camera relative to drone)
        self._cam_offset = torch.tensor(self.cfg.camera_body_offset_pos, dtype=torch.float32, device=self.device)

        # World positions of all 3 objects for every env: (num_envs, 3, 3)
        # Kept for reward/done computation (ground truth, not in obs)
        offsets = torch.tensor(_OBJ_OFFSETS, dtype=torch.float32, device=self.device)
        self._obj_pos_w = self._terrain.env_origins.unsqueeze(1) + offsets.unsqueeze(0)

        # Robot dynamics constants
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Logging accumulators
        self._episode_sums = {
            k: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for k in [
                "lin_vel", "ang_vel", "alive", "uprightness", "altitude_penalty",
                "distance_to_goal", "velocity_toward_goal", "proximity",
                "success", "wrong_object",
            ]
        }

        # CLIP grounder — loaded once, frozen (text + vision)
        self._grounder = CLIPGrounder(device=self.device)

        # Warm up with a full-env reset so CLIP text embeddings are populated
        self._reset_idx(None)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)

        # Onboard camera — mounted on drone body, auto-updated by scene
        self._camera = TiledCamera(self.cfg.tiled_camera)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Spawn visual-only geometric objects (no rigid body / physics needed)
        cube_cfg = sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        )
        cube_cfg.func(
            "/World/envs/env_.*/cube", cube_cfg, translation=_OBJ_OFFSETS[0]
        )

        sphere_cfg = sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
        )
        sphere_cfg.func(
            "/World/envs/env_.*/sphere", sphere_cfg, translation=_OBJ_OFFSETS[1]
        )

        cylinder_cfg = sim_utils.CylinderCfg(
            radius=0.2,
            height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        )
        cylinder_cfg.func(
            "/World/envs/env_.*/cylinder", cylinder_cfg, translation=_OBJ_OFFSETS[2]
        )

        # Coloured corner markers for spatial reference (matches hover/waypoint)
        for pos, color, name in [
            ((2.0, 0.0, 0.15), (0.8, 0.1, 0.1), "marker_red"),
            ((-2.0, 0.0, 0.15), (0.1, 0.1, 0.8), "marker_blue"),
            ((0.0, 2.0, 0.15), (0.1, 0.8, 0.1), "marker_green"),
            ((0.0, -2.0, 0.15), (0.8, 0.8, 0.1), "marker_yellow"),
        ]:
            m_cfg = sim_utils.CylinderCfg(
                radius=0.08,
                height=0.3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            m_cfg.func(f"/World/envs/env_.*/{name}", m_cfg, translation=pos)

        # Clone environments, then register assets and sensors
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._camera

        light_cfg = sim_utils.DomeLightCfg(
            intensity=1500.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg)
        dist_light = sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.95, 0.85))
        dist_light.func("/World/SunLight", dist_light)

        # Dynamic target marker — glowing sphere above the current target object
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
    # Vision encoding
    # ------------------------------------------------------------------

    def _update_camera_pose(self):
        """Move the standalone camera prim to follow the drone body each step."""
        drone_pos = self._robot.data.root_pos_w   # (N, 3)
        drone_quat = self._robot.data.root_quat_w  # (N, 4) wxyz
        # Apply body-frame offset to get camera world position
        cam_pos = drone_pos + quat_apply(drone_quat, self._cam_offset.expand(self.num_envs, -1))
        # Camera inherits drone orientation (forward-facing)
        self._camera._view.set_world_poses(cam_pos, drone_quat)

    def _maybe_encode_vision(self):
        """Encode onboard camera images with CLIP every N steps."""
        self._steps_since_encode += 1
        if (self._steps_since_encode >= self.cfg.clip_encode_every_n).any():
            rgb = self._camera.data.output["rgb"][:, :, :, :3]  # (N, 64, 64, 3) uint8
            self._clip_img_emb = self._grounder.encode_images(rgb)
            self._steps_since_encode.zero_()

    # ------------------------------------------------------------------
    # Two-phase learning
    # ------------------------------------------------------------------

    def _get_nav_multiplier(self) -> float:
        """0.0 during survival-only phase, linearly ramps to 1.0 during fade-in."""
        steps = self.common_step_counter
        if steps < self.cfg.survival_only_steps:
            return 0.0
        fade_progress = min(
            (steps - self.cfg.survival_only_steps) / self.cfg.nav_fadein_steps, 1.0
        )
        return fade_progress

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

        # Track drone with camera and encode vision periodically
        self._update_camera_pose()
        self._maybe_encode_vision()

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    # ------------------------------------------------------------------
    # MDP components
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Update target marker above the current target object
        env_ids = torch.arange(self.num_envs, device=self.device)
        marker_pos = self._obj_pos_w[env_ids, self._target_obj_idx].clone()
        marker_pos[:, 2] += 0.7
        self._target_marker.visualize(translations=marker_pos)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,        # (N, 3)
                self._robot.data.root_ang_vel_b,        # (N, 3)
                self._robot.data.projected_gravity_b,   # (N, 3)  → total 9
                self._clip_emb,                         # (N, 512)
                self._clip_img_emb,                     # (N, 512)
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        drone_pos = self._robot.data.root_pos_w

        target_pos = self._obj_pos_w[env_ids, self._target_obj_idx]
        to_goal = target_pos - drone_pos
        dist_to_target = torch.linalg.norm(to_goal, dim=1)
        dist_mapped = 1.0 - torch.tanh(dist_to_target / 0.8)

        # Velocity toward goal (world frame)
        to_goal_dir = to_goal / dist_to_target.unsqueeze(1).clamp(min=0.01)
        vel_toward = torch.sum(self._robot.data.root_lin_vel_w * to_goal_dir, dim=1)
        vel_toward_clamp = torch.clamp(vel_toward, 0.0, 2.0)

        # Proximity bonus: linear ramp inside radius
        inside_radius = (dist_to_target < self.cfg.proximity_radius).float()
        proximity = inside_radius * (1.0 - dist_to_target / self.cfg.proximity_radius)

        # Uprightness: +1 when level, -1 when inverted
        uprightness = -self._robot.data.projected_gravity_b[:, 2]

        # Altitude warning: soft penalty near floor/ceiling
        too_low = torch.clamp(self.cfg.altitude_warning_low - drone_pos[:, 2], min=0.0)
        too_high = torch.clamp(drone_pos[:, 2] - self.cfg.altitude_warning_high, min=0.0)
        altitude_penalty = (too_low + too_high) * self.cfg.crash_penalty_scale

        # Minimum distance to any wrong object
        wrong_mask = torch.ones(self.num_envs, 3, dtype=torch.bool, device=self.device)
        wrong_mask[env_ids, self._target_obj_idx] = False
        wrong_obj_pos = self._obj_pos_w[wrong_mask].reshape(self.num_envs, 2, 3)
        dist_to_wrong = torch.linalg.norm(
            wrong_obj_pos - drone_pos.unsqueeze(1), dim=-1
        ).min(dim=1).values

        success = dist_to_target < self.cfg.success_threshold
        wrong_object = dist_to_wrong < self.cfg.success_threshold

        # Two-phase: navigation rewards are zero during survival phase
        nav = self._get_nav_multiplier()

        rewards = {
            # Stability rewards — always active (preserve pretrained hover)
            "lin_vel": (
                torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
                * self.cfg.lin_vel_reward_scale
                * self.step_dt
            ),
            "ang_vel": (
                torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
                * self.cfg.ang_vel_reward_scale
                * self.step_dt
            ),
            "alive": (
                torch.ones(self.num_envs, device=self.device)
                * self.cfg.alive_reward
                * self.step_dt
            ),
            "uprightness": uprightness * self.cfg.uprightness_reward_scale * self.step_dt,
            "altitude_penalty": altitude_penalty * self.step_dt,
            # Navigation rewards — always active (waypoint learned with these;
            # gating them kills the transferred nav skill)
            "distance_to_goal": dist_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "velocity_toward_goal": vel_toward_clamp * self.cfg.velocity_toward_goal_scale * self.step_dt,
            "proximity": proximity * self.cfg.proximity_scale * self.step_dt,
            # CLIP-dependent rewards — gated (these are NEW skills the VLA must learn)
            "success": success.float() * self.cfg.success_reward * nav,
            "wrong_object": wrong_object.float() * self.cfg.wrong_object_penalty * nav,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v
        return reward

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

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        extras: dict = {}
        for key in self._episode_sums:
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras

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

        # Sample new language commands and encode with CLIP
        n = len(env_ids)
        obj_type_indices = torch.randint(0, len(OBJECT_TYPES), (n,))  # ground truth targets

        commands = [
            random.choice(COMMANDS[OBJECT_TYPES[i.item()]])
            for i in obj_type_indices
        ]
        clip_embs = self._grounder.encode_texts(commands)  # (n, 512)

        self._clip_emb[env_ids] = clip_embs.to(self.device)
        self._target_obj_idx[env_ids] = obj_type_indices.to(self.device)
        for i, eid in enumerate(env_ids):
            self._current_commands[int(eid)] = commands[i]

        # Force immediate vision encoding on next step for reset envs
        self._steps_since_encode[env_ids] = self.cfg.clip_encode_every_n

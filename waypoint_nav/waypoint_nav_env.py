"""Waypoint navigation environment — fly to a distant goal and stop.

Stage 1.5 of curriculum: the drone learns to navigate to randomised goal
positions, with a terminal success bonus on arrival.
This bridges hover (hold position) and lang_nav (fly to language-specified
objects) by teaching intentional point-to-point navigation.

Observation (15-dim, identical to hover for direct weight transfer):
  [0:3]   root_lin_vel_b      (body-frame linear velocity)
  [3:6]   root_ang_vel_b      (body-frame angular velocity)
  [6:9]   projected_gravity_b (gravity in body frame)
  [9:12]  target_pos_b        (goal position in body frame)
  [12:15] root_pos_error_w    (world-frame position error for shaping)

Action (4-dim): normalised thrust + 3-axis moment, same as hover and
lang_nav so weights transfer directly.
"""

from __future__ import annotations

import json
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip


@configclass
class WaypointNavEnvCfg(DirectRLEnvCfg):
    # Episode / stepping
    episode_length_s = 12.0
    decimation = 2
    action_space = 4
    observation_space = 15
    state_space = 0
    debug_vis = False

    # Camera — close tracking, matches hover
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 1.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        origin_type="asset_root",
        asset_name="robot",
        resolution=(1280, 720),
    )

    # Simulation — identical physics to hover and lang_nav
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
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Reward scales — simplified, no phase curriculum
    xy_reward_scale = 10.0
    z_reward_scale = 6.0
    velocity_toward_goal_scale = 4.0
    uprightness_reward_scale = 0.5
    lin_vel_penalty_scale = -0.01
    ang_vel_penalty_scale = -0.01
    alive_reward = 2.0                 # boosted from 0.5 — survival must compete with navigation
    success_reward = 50.0              # quality-gated, on waypoint reach
    proximity_scale = 8.0
    proximity_radius = 1.5
    pinpoint_scale = 20.0
    pinpoint_radius = 0.5
    crash_penalty_scale = -15.0        # penalty for being near ground/ceiling
    altitude_warning_low = 0.3         # soft penalty below this height
    altitude_warning_high = 2.8        # soft penalty above this height

    # Threshold curriculum — starts easy, tightens as the drone improves
    success_threshold_start = 1.0
    success_threshold_end = 0.15
    success_threshold_ramp_steps = 40_000

    # Out-of-bounds termination
    xy_boundary = 5.0

    # Goal distance curriculum — start close, ramp to full difficulty
    curriculum_start_min = 0.5
    curriculum_start_max = 1.5
    curriculum_end_min = 1.0
    curriculum_end_max = 3.0
    curriculum_ramp_steps = 60_000

    # Goal height
    goal_height_min = 0.3     # metres
    goal_height_max = 2.0     # metres


class WaypointNavEnv(DirectRLEnv):
    cfg: WaypointNavEnvCfg

    def __init__(self, cfg: WaypointNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action / wrench buffers (same layout as hover and lang_nav)
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal position in world frame (randomised per reset)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Track waypoints reached per episode (for multi-waypoint scoring)
        self._waypoints_reached = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Robot dynamics constants
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Logging accumulators — reward components
        self._episode_sums = {
            k: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for k in [
                "distance_to_goal_xy", "distance_to_goal_z", "velocity_toward_goal",
                "proximity", "pinpoint", "uprightness", "lin_vel", "ang_vel",
                "alive", "altitude_penalty", "success",
            ]
        }

        # Logging accumulators — raw physical metrics (not reward-scaled)
        self._episode_metrics = {
            k: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for k in [
                "dist_to_target",       # actual 3D distance in metres
                "xy_dist",              # horizontal distance
                "z_dist",               # vertical distance
                "closest_dist",         # minimum distance achieved in episode
                "speed",                # linear speed magnitude
                "uprightness_raw",      # raw uprightness value (-1 to 1)
                "ang_vel_mag",          # angular velocity magnitude
                "arrival_quality",      # quality gate value at success (0.04 to 1.0)
                "waypoints_reached",   # number of waypoints reached this episode
            ]
        }
        self._episode_step_counts = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Track closest approach per episode (reset to large value)
        self._closest_dist = torch.full((self.num_envs,), 100.0, device=self.device)
        # Track termination reasons
        self._term_counts = {
            k: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for k in ["success", "fell", "flipped", "oob", "timeout"]
        }

        # Metrics log file — fixed path so it can be read during training
        self._metrics_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "waypoint_nav_metrics.jsonl",
        )
        os.makedirs(os.path.dirname(self._metrics_path), exist_ok=True)
        self._metrics_file = open(self._metrics_path, "w")
        self._log_step = 0

        # Initial reset
        self._reset_idx(None)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Colored corner markers for spatial reference (matches hover)
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

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting — matches hover exactly
        light_cfg = sim_utils.DomeLightCfg(
            intensity=1500.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg)
        dist_light = sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.95, 0.85))
        dist_light.func("/World/SunLight", dist_light)

        # Dynamic waypoint marker — bright red sphere, updated each step
        self._wp_marker = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/World/Visuals/waypoint_markers",
            markers={
                "waypoint": sim_utils.SphereCfg(
                    radius=0.2,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        ))

    # ------------------------------------------------------------------
    # Reward curriculum — phase-dependent multipliers
    # ------------------------------------------------------------------

    def _get_success_threshold(self) -> float:
        """Linearly tighten the success threshold over training."""
        progress = min(self.common_step_counter / self.cfg.success_threshold_ramp_steps, 1.0)
        return (
            self.cfg.success_threshold_start
            + progress * (self.cfg.success_threshold_end - self.cfg.success_threshold_start)
        )

    # ------------------------------------------------------------------
    # Physics step — identical action mapping to hover and lang_nav
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    # ------------------------------------------------------------------
    # MDP components
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Update waypoint marker positions for visualization
        self._wp_marker.visualize(translations=self._target_pos_w)

        # Target position in body frame
        target_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._target_pos_w,
        )

        # World-frame position error (useful for shaping)
        pos_error_w = self._target_pos_w - self._robot.data.root_pos_w

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,       # (N, 3)
                self._robot.data.root_ang_vel_b,       # (N, 3)
                self._robot.data.projected_gravity_b,   # (N, 3)
                target_pos_b,                           # (N, 3)
                pos_error_w,                            # (N, 3)
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        drone_pos = self._robot.data.root_pos_w

        # -- Split XY and Z distance rewards (wider tanh for 1-3m goals) --
        xy_dist = torch.linalg.norm(self._target_pos_w[:, :2] - drone_pos[:, :2], dim=1)
        z_dist = torch.abs(self._target_pos_w[:, 2] - drone_pos[:, 2])
        xy_mapped = 1.0 - torch.tanh(xy_dist / 0.8)
        z_mapped = 1.0 - torch.tanh(z_dist / 0.5)

        # -- Velocity toward goal (world frame) --
        to_goal = self._target_pos_w - drone_pos
        to_goal_dist = torch.linalg.norm(to_goal, dim=1, keepdim=True).clamp(min=0.01)
        to_goal_dir = to_goal / to_goal_dist
        vel_toward = torch.sum(self._robot.data.root_lin_vel_w * to_goal_dir, dim=1)
        vel_toward_clamp = torch.clamp(vel_toward, 0.0, 2.0)

        # -- Uprightness reward --
        uprightness = -self._robot.data.projected_gravity_b[:, 2]

        # -- Success check (non-terminating — respawn waypoint instead) --
        dist = torch.linalg.norm(to_goal, dim=1)
        current_threshold = self._get_success_threshold()
        success = dist < current_threshold

        # -- Proximity bonus: linear ramp inside 1.5m --
        inside_radius = (dist < self.cfg.proximity_radius).float()
        proximity = inside_radius * (1.0 - dist / self.cfg.proximity_radius)

        # -- Pinpoint bonus: very steep ramp inside 0.5m — always pulls toward exact point --
        inside_pinpoint = (dist < self.cfg.pinpoint_radius).float()
        pinpoint = inside_pinpoint * (1.0 - dist / self.cfg.pinpoint_radius)

        # -- Quality-gated success --
        speed = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        ang_vel_mag = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
        speed_quality = torch.clamp(1.0 - speed / 1.5, min=0.2, max=1.0)
        stability_quality = torch.clamp(1.0 - ang_vel_mag / 3.0, min=0.2, max=1.0)
        arrival_quality = speed_quality * stability_quality

        # -- Altitude warning: soft penalty near ground/ceiling --
        too_low = torch.clamp(self.cfg.altitude_warning_low - drone_pos[:, 2], min=0.0)
        too_high = torch.clamp(drone_pos[:, 2] - self.cfg.altitude_warning_high, min=0.0)
        altitude_penalty = (too_low + too_high) * self.cfg.crash_penalty_scale

        rewards = {
            "distance_to_goal_xy": xy_mapped * self.cfg.xy_reward_scale * self.step_dt,
            "distance_to_goal_z": z_mapped * self.cfg.z_reward_scale * self.step_dt,
            "velocity_toward_goal": vel_toward_clamp * self.cfg.velocity_toward_goal_scale * self.step_dt,
            "proximity": proximity * self.cfg.proximity_scale * self.step_dt,
            "pinpoint": pinpoint * self.cfg.pinpoint_scale * self.step_dt,
            "uprightness": uprightness * self.cfg.uprightness_reward_scale * self.step_dt,
            "lin_vel": (
                torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
                * self.cfg.lin_vel_penalty_scale
                * self.step_dt
            ),
            "ang_vel": (
                torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
                * self.cfg.ang_vel_penalty_scale
                * self.step_dt
            ),
            "alive": torch.ones(self.num_envs, device=self.device) * self.cfg.alive_reward * self.step_dt,
            "altitude_penalty": altitude_penalty * self.step_dt,
            "success": success.float() * self.cfg.success_reward * arrival_quality,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v

        # Track raw physical metrics
        self._episode_metrics["dist_to_target"] += dist
        self._episode_metrics["xy_dist"] += xy_dist
        self._episode_metrics["z_dist"] += z_dist
        self._episode_metrics["speed"] += speed
        self._episode_metrics["uprightness_raw"] += uprightness
        self._episode_metrics["ang_vel_mag"] += ang_vel_mag
        self._episode_metrics["arrival_quality"] += arrival_quality
        self._closest_dist = torch.min(self._closest_dist, dist)
        self._episode_step_counts += 1

        # -- Respawn waypoint on success (don't terminate — keep flying!) --
        success_ids = success.nonzero(as_tuple=False).squeeze(-1)
        if len(success_ids) > 0:
            self._waypoints_reached[success_ids] += 1
            self._respawn_waypoints(success_ids)

        return reward

    def _respawn_waypoints(self, env_ids: torch.Tensor):
        """Randomise goal positions for the given envs (used on reset AND mid-episode success)."""
        import math
        progress = min(self.common_step_counter / self.cfg.curriculum_ramp_steps, 1.0)
        current_min = (
            self.cfg.curriculum_start_min
            + progress * (self.cfg.curriculum_end_min - self.cfg.curriculum_start_min)
        )
        current_max = (
            self.cfg.curriculum_start_max
            + progress * (self.cfg.curriculum_end_max - self.cfg.curriculum_start_max)
        )

        n = len(env_ids)
        angle = torch.rand(n, device=self.device) * 2 * math.pi
        distance = torch.rand(n, device=self.device) * (current_max - current_min) + current_min
        target_z = (
            torch.rand(n, device=self.device)
            * (self.cfg.goal_height_max - self.cfg.goal_height_min)
            + self.cfg.goal_height_min
        )

        self._target_pos_w[env_ids, 0] = self._terrain.env_origins[env_ids, 0] + distance * torch.cos(angle)
        self._target_pos_w[env_ids, 1] = self._terrain.env_origins[env_ids, 1] + distance * torch.sin(angle)
        self._target_pos_w[env_ids, 2] = target_z

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        drone_pos = self._robot.data.root_pos_w

        # Crashed into ground or flew way too high
        fell = (drone_pos[:, 2] < 0.1) | (drone_pos[:, 2] > 3.0)

        # Flipped upside down
        flipped = self._robot.data.projected_gravity_b[:, 2] > 0.5

        # Drifted too far from env origin
        xy_offset = drone_pos[:, :2] - self._terrain.env_origins[:, :2]
        out_of_bounds = torch.any(torch.abs(xy_offset) > self.cfg.xy_boundary, dim=1)

        # No success termination — waypoints respawn mid-episode in _get_rewards
        terminated = fell | flipped | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Track termination reasons
        self._term_counts["fell"] += fell.float()
        self._term_counts["flipped"] += (flipped & ~fell).float()
        self._term_counts["oob"] += (out_of_bounds & ~flipped & ~fell).float()
        self._term_counts["timeout"] += (time_out & ~terminated).float()

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging — reward components
        extras: dict = {}
        for key in self._episode_sums:
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # Logging — raw physical metrics (per-step averages)
        steps = self._episode_step_counts[env_ids].clamp(min=1)
        for key in self._episode_metrics:
            extras[f"Metrics/{key}"] = torch.mean(self._episode_metrics[key][env_ids] / steps)
            self._episode_metrics[key][env_ids] = 0.0
        extras["Metrics/closest_dist"] = torch.mean(self._closest_dist[env_ids])
        self._closest_dist[env_ids] = 100.0
        self._episode_step_counts[env_ids] = 0

        # Logging — termination breakdown (fraction of resets by cause)
        n_resets = len(env_ids)
        for key in self._term_counts:
            extras[f"Termination/{key}"] = torch.sum(self._term_counts[key][env_ids]) / max(n_resets, 1)
            self._term_counts[key][env_ids] = 0.0

        # Logging — curriculum state
        import math
        progress = min(self.common_step_counter / self.cfg.curriculum_ramp_steps, 1.0)
        current_min = self.cfg.curriculum_start_min + progress * (self.cfg.curriculum_end_min - self.cfg.curriculum_start_min)
        current_max = self.cfg.curriculum_start_max + progress * (self.cfg.curriculum_end_max - self.cfg.curriculum_start_max)
        extras["Curriculum/progress"] = progress
        extras["Curriculum/goal_dist_min"] = current_min
        extras["Curriculum/goal_dist_max"] = current_max
        extras["Curriculum/success_threshold"] = self._get_success_threshold()
        extras["Metrics/waypoints_reached"] = torch.mean(self._waypoints_reached[env_ids])
        self._waypoints_reached[env_ids] = 0

        self.extras["log"] = extras

        # Write to metrics log file (every 50 reset calls to limit I/O)
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

        self._respawn_waypoints(env_ids)

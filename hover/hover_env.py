"""Hover environment — teach the drone to fly and hold a target position.

Stage 1 of curriculum: the drone learns basic flight control (altitude hold,
stability, position tracking) before being exposed to language-grounded
navigation.

Observation (15-dim):
  [0:3]   root_lin_vel_b      (body-frame linear velocity)
  [3:6]   root_ang_vel_b      (body-frame angular velocity)
  [6:9]   projected_gravity_b (gravity in body frame — tells agent which way is up)
  [9:12]  target_pos_b        (hover target in body frame)
  [12:15] root_pos_error_w    (world-frame position error for shaping)

Action (4-dim): normalised thrust + 3-axis moment, same as lang_nav env so
weights transfer directly.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip


@configclass
class HoverEnvCfg(DirectRLEnvCfg):
    # Episode / stepping — same dt and decimation as lang_nav
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 15
    state_space = 0
    debug_vis = False

    # Camera positioned close to the drone, tracking the robot body
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 1.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        origin_type="asset_root",
        asset_name="robot",
        resolution=(1280, 720),
    )

    # Simulation — identical physics to lang_nav so dynamics transfer
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
        num_envs=1024, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Reward scales
    xy_reward_scale = 12.0         # shaped XY distance-to-hover-target
    z_reward_scale = 8.0           # shaped Z distance-to-hover-target
    uprightness_reward_scale = 0.5 # reduced from 2.0 to allow tilting for XY correction
    lin_vel_penalty_scale = -0.05  # penalise erratic movement
    ang_vel_penalty_scale = -0.01  # penalise spinning
    alive_reward = 0.5             # per-step bonus for not crashing
    success_reward = 5.0           # one-time terminal bonus (matches waypoint_nav)
    success_threshold = 0.8        # metres to count as "reached target"

    # Hover target randomisation
    hover_height_min = 0.3
    hover_height_max = 2.0
    hover_xy_range = 0.5  # target spawns within ±this from env origin
    hover_xy_boundary = 3.0  # terminate if drone drifts beyond ±this from env origin


class HoverEnv(DirectRLEnv):
    cfg: HoverEnvCfg

    def __init__(self, cfg: HoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action / wrench buffers (same layout as lang_nav)
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Hover target in world frame (randomised per reset)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Robot dynamics constants
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Logging accumulators — rewards
        self._episode_sums = {
            k: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for k in ["xy_pos", "z_pos", "uprightness", "lin_vel", "ang_vel", "alive", "success"]
        }
        # Logging accumulators — raw metrics (not reward-scaled)
        self._episode_metrics = {
            k: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for k in ["dist_to_target", "xy_dist", "z_dist", "uprightness_raw", "ang_vel_mag"]
        }
        self._episode_step_counts = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

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

        # Coloured corner markers so you can see the drone moving relative to something
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

        # Lighting — key light + fill for depth
        light_cfg = sim_utils.DomeLightCfg(
            intensity=1500.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg)
        dist_light = sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.95, 0.85))
        dist_light.func("/World/SunLight", dist_light)

    # ------------------------------------------------------------------
    # Physics step — identical action mapping to lang_nav
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

        # -- Separate XY and Z position rewards --
        xy_dist = torch.linalg.norm(self._target_pos_w[:, :2] - drone_pos[:, :2], dim=1)
        z_dist = torch.abs(self._target_pos_w[:, 2] - drone_pos[:, 2])
        xy_mapped = 1.0 - torch.tanh(xy_dist / 0.8)
        z_mapped = 1.0 - torch.tanh(z_dist / 0.8)

        # Full 3D distance for success check
        dist = torch.linalg.norm(self._target_pos_w - drone_pos, dim=1)

        # -- Uprightness reward: projected_gravity_b z-component should be -1 when level --
        uprightness = -self._robot.data.projected_gravity_b[:, 2]  # +1 when upright, -1 when inverted

        # -- Terminal success bonus (one-time, not scaled by step_dt) --
        success = (dist < self.cfg.success_threshold).float()

        rewards = {
            "xy_pos": xy_mapped * self.cfg.xy_reward_scale * self.step_dt,
            "z_pos": z_mapped * self.cfg.z_reward_scale * self.step_dt,
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
            "success": success * self.cfg.success_reward,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v

        # Track raw physical metrics (not reward-scaled)
        ang_vel_mag = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
        self._episode_metrics["dist_to_target"] += dist
        self._episode_metrics["xy_dist"] += xy_dist
        self._episode_metrics["z_dist"] += z_dist
        self._episode_metrics["uprightness_raw"] += uprightness
        self._episode_metrics["ang_vel_mag"] += ang_vel_mag
        self._episode_step_counts += 1

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        drone_pos = self._robot.data.root_pos_w

        # Crashed into ground or flew way too high
        fell = (drone_pos[:, 2] < 0.1) | (drone_pos[:, 2] > 3.0)

        # Flipped upside down (projected gravity z > 0 means inverted)
        flipped = self._robot.data.projected_gravity_b[:, 2] > 0.5

        # Drifted too far horizontally from env origin
        xy_offset = drone_pos[:, :2] - self._terrain.env_origins[:, :2]
        out_of_bounds_xy = torch.any(torch.abs(xy_offset) > self.cfg.hover_xy_boundary, dim=1)

        # Reached the hover target
        dist = torch.linalg.norm(self._target_pos_w - drone_pos, dim=1)
        success = dist < self.cfg.success_threshold

        terminated = fell | flipped | out_of_bounds_xy | success
        time_out = self.episode_length_buf >= self.max_episode_length - 1
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

        # Logging — raw stability metrics (per-step averages over the episode)
        steps = self._episode_step_counts[env_ids].clamp(min=1)
        for key in self._episode_metrics:
            extras[f"Metrics/{key}"] = torch.mean(self._episode_metrics[key][env_ids] / steps)
            self._episode_metrics[key][env_ids] = 0.0
        self._episode_step_counts[env_ids] = 0

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

        # Randomise hover target
        n = len(env_ids)
        xy_range = self.cfg.hover_xy_range
        target_x = (torch.rand(n, device=self.device) * 2 - 1) * xy_range
        target_y = (torch.rand(n, device=self.device) * 2 - 1) * xy_range
        target_z = (
            torch.rand(n, device=self.device)
            * (self.cfg.hover_height_max - self.cfg.hover_height_min)
            + self.cfg.hover_height_min
        )
        self._target_pos_w[env_ids, 0] = self._terrain.env_origins[env_ids, 0] + target_x
        self._target_pos_w[env_ids, 1] = self._terrain.env_origins[env_ids, 1] + target_y
        self._target_pos_w[env_ids, 2] = self._terrain.env_origins[env_ids, 2] + target_z

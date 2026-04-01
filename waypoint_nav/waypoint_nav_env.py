"""Waypoint navigation environment — fly to a distant goal and stop.

Stage 1.5 of curriculum: the drone learns to navigate to randomised goal
positions at 1-3m distance, with a terminal success bonus on arrival.
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

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
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

    # Reward scales
    distance_to_goal_reward_scale = 15.0   # shaped distance (matches lang_nav)
    uprightness_reward_scale = 0.5         # reduced from hover's 2.0 — allow tilting
    lin_vel_penalty_scale = -0.05          # same as hover and lang_nav
    ang_vel_penalty_scale = -0.01          # same as hover and lang_nav
    alive_reward = 0.2                     # reduced from hover's 0.5
    success_reward = 5.0                   # terminal bonus (matches lang_nav)
    success_threshold = 0.35              # metres — matches lang_nav

    # Goal randomisation — radial distribution
    goal_distance_min = 1.0   # metres
    goal_distance_max = 3.0   # metres
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

        # Robot dynamics constants
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Logging accumulators
        self._episode_sums = {
            k: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for k in ["distance_to_goal", "uprightness", "lin_vel", "ang_vel", "alive", "success"]
        }

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

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg)

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

        # -- Distance-to-goal reward (tanh-shaped, same as hover and lang_nav) --
        dist = torch.linalg.norm(self._target_pos_w - drone_pos, dim=1)
        dist_mapped = 1.0 - torch.tanh(dist / 0.8)

        # -- Uprightness reward (reduced from hover) --
        uprightness = -self._robot.data.projected_gravity_b[:, 2]

        # -- Terminal success bonus --
        success = dist < self.cfg.success_threshold

        rewards = {
            "distance_to_goal": dist_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
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
            "success": success.float() * self.cfg.success_reward,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        drone_pos = self._robot.data.root_pos_w

        # Reached the goal
        dist = torch.linalg.norm(self._target_pos_w - drone_pos, dim=1)
        success = dist < self.cfg.success_threshold

        # Crashed into ground or flew way too high
        fell = (drone_pos[:, 2] < 0.1) | (drone_pos[:, 2] > 3.0)

        # Flipped upside down
        flipped = self._robot.data.projected_gravity_b[:, 2] > 0.5

        terminated = success | fell | flipped
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

        # Randomise goal — radial distribution (ensures minimum 1m travel)
        n = len(env_ids)
        angle = torch.rand(n, device=self.device) * 2 * math.pi
        distance = (
            torch.rand(n, device=self.device)
            * (self.cfg.goal_distance_max - self.cfg.goal_distance_min)
            + self.cfg.goal_distance_min
        )
        target_z = (
            torch.rand(n, device=self.device)
            * (self.cfg.goal_height_max - self.cfg.goal_height_min)
            + self.cfg.goal_height_min
        )

        self._target_pos_w[env_ids, 0] = self._terrain.env_origins[env_ids, 0] + distance * torch.cos(angle)
        self._target_pos_w[env_ids, 1] = self._terrain.env_origins[env_ids, 1] + distance * torch.sin(angle)
        self._target_pos_w[env_ids, 2] = target_z

"""Language-grounded drone navigation environment.

The drone receives a natural language command ("go to the square") and must
navigate to the matching geometric object in the scene. Three colored objects
are placed at fixed positions in each env:
  - Cube   (red)   at offset (-1.5,  0.0, 0.2) → index 0
  - Sphere (blue)  at offset ( 1.5,  0.0, 0.2) → index 1
  - Cylinder (green) at offset (0.0, 1.5, 0.2) → index 2

Observation (533-dim):
  [0:3]   root_lin_vel_b
  [3:6]   root_ang_vel_b
  [6:9]   projected_gravity_b
  [9:12]  target object position in body frame
  [12:524] CLIP text embedding of the command (512-dim, frozen)
  [524:533] all 3 object positions relative to drone, flattened (9-dim)

Reward:
  + shaped distance-to-target
  - lin/ang velocity penalties (stability)
  + 5.0 success bonus  (drone reaches target, episode ends)
  - 3.0 wrong-object penalty (drone reaches wrong object, episode ends)
"""

from __future__ import annotations

import random

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

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
    # 12 drone state + 512 CLIP embedding + 9 object relative positions
    observation_space = 12 + _CLIP_DIM + 9
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
        debug_vis=False,
    )

    # Larger env spacing so objects don't overlap between envs
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=6.0, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    success_reward = 5.0
    wrong_object_penalty = -3.0

    # Task parameters
    success_threshold = 0.35  # metres — drone must get this close to target


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

        # World positions of all 3 objects for every env: (num_envs, 3, 3)
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
            for k in ["lin_vel", "ang_vel", "distance_to_goal", "success", "wrong_object"]
        }

        # CLIP grounder — loaded once, frozen
        self._grounder = CLIPGrounder(device=self.device)

        # Warm up with a full-env reset so CLIP embeddings are populated
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

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Physics step
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
        drone_pos = self._robot.data.root_pos_w  # (N, 3)
        env_ids = torch.arange(self.num_envs, device=self.device)

        # Target position in drone body frame
        target_pos_w = self._obj_pos_w[env_ids, self._target_obj_idx]  # (N, 3)
        target_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            target_pos_w,
        )

        # All object positions relative to drone (world frame, flattened)
        obj_rel_pos = (self._obj_pos_w - drone_pos.unsqueeze(1)).reshape(self.num_envs, -1)  # (N, 9)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,        # (N, 3)
                self._robot.data.root_ang_vel_b,        # (N, 3)
                self._robot.data.projected_gravity_b,   # (N, 3)
                target_pos_b,                           # (N, 3)  → total 12
                self._clip_emb,                         # (N, 512)
                obj_rel_pos,                            # (N, 9)
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        drone_pos = self._robot.data.root_pos_w

        target_pos = self._obj_pos_w[env_ids, self._target_obj_idx]
        dist_to_target = torch.linalg.norm(target_pos - drone_pos, dim=1)
        dist_mapped = 1.0 - torch.tanh(dist_to_target / 0.8)

        # Minimum distance to any wrong object
        wrong_mask = torch.ones(self.num_envs, 3, dtype=torch.bool, device=self.device)
        wrong_mask[env_ids, self._target_obj_idx] = False
        wrong_obj_pos = self._obj_pos_w[wrong_mask].reshape(self.num_envs, 2, 3)
        dist_to_wrong = torch.linalg.norm(
            wrong_obj_pos - drone_pos.unsqueeze(1), dim=-1
        ).min(dim=1).values

        success = dist_to_target < self.cfg.success_threshold
        wrong_object = dist_to_wrong < self.cfg.success_threshold

        rewards = {
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
            "distance_to_goal": (
                dist_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt
            ),
            "success": success.float() * self.cfg.success_reward,
            "wrong_object": wrong_object.float() * self.cfg.wrong_object_penalty,
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

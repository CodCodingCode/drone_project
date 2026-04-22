"""Play back a trained VLA policy and record video with text overlay + drone POV.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/vla/play.py \
        --checkpoint <path/to/model.pt> --enable_cameras --video
"""

import argparse
import glob
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["OMNI_LOG_LEVEL"] = "ERROR"

from isaaclab.app import AppLauncher

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Play trained VLA policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=750)
parser.add_argument("--video", action="store_true", default=True, help="Record video")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------

import cv2
import numpy as np
import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla  # noqa: F401

from vla.vla_drone_env import VLADroneEnvCfg, VLADroneEnv

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils.math import quat_from_euler_xyz
from vla.vla_policy import HierarchicalVLAActor


# ----------------------------------------------------------------------
# Subclass that adds a third-person observer camera following the drone
# ----------------------------------------------------------------------

class VLADroneEnvWithObserver(VLADroneEnv):
    """VLA env with an extra TiledCamera for third-person video recording.

    The observer camera follows the drone in WORLD frame (not body frame) so the
    view is stable as the drone yaws. Position is drone_pos + (-2, 0, 1.2) — 2m
    behind in world -X, 1.2m above. Orientation looks along world +X with a
    slight downward tilt so the drone stays centered in frame.
    """

    def _setup_scene(self):
        # Replicate parent _setup_scene logic but insert observer camera
        # BEFORE clone_environments so it gets properly registered with the scene
        from isaaclab.assets import Articulation
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        from isaaclab.sensors import TiledCamera
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        self._robot = Articulation(self.cfg.robot)

        # Standard 4 onboard cameras
        self._cameras = [
            TiledCamera(self.cfg.cam_front),
            TiledCamera(self.cfg.cam_right),
            TiledCamera(self.cfg.cam_back),
            TiledCamera(self.cfg.cam_left),
        ]

        # Observer camera (5th) — added BEFORE clone_environments so it's part of scene cloning
        # Longer focal length (24mm) zooms in so the small Crazyflie drone is visible
        observer_cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/CamObserver",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,         # 2x zoom vs 12mm — drone appears 2x larger
                focus_distance=100.0,
                horizontal_aperture=24.0,  # ~53° FOV (vs 90° at 12mm)
                clipping_range=(0.1, 50.0),
            ),
            width=640,
            height=480,
        )
        self._observer_camera = TiledCamera(observer_cfg)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Spawn objects
        cube_cfg = sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        )
        cube_cfg.func("/World/envs/env_.*/cube", cube_cfg, translation=(-1.5, 0.0, 0.2))

        sphere_cfg = sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
        )
        sphere_cfg.func("/World/envs/env_.*/sphere", sphere_cfg, translation=(1.5, 0.0, 0.2))

        cylinder_cfg = sim_utils.CylinderCfg(
            radius=0.2, height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        )
        cylinder_cfg.func("/World/envs/env_.*/cylinder", cylinder_cfg, translation=(0.0, 1.5, 0.2))

        self.scene.clone_environments(copy_from_source=False)

        # XformPrimViews for repositioning objects at reset
        from isaaclab.sim.views import XformPrimView
        self._cube_view = XformPrimView("/World/envs/env_.*/cube", device=self.device)
        self._sphere_view = XformPrimView("/World/envs/env_.*/sphere", device=self.device)
        self._cylinder_view = XformPrimView("/World/envs/env_.*/cylinder", device=self.device)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self._robot
        for i, cam in enumerate(self._cameras):
            self.scene.sensors[f"tiled_camera_{i}"] = cam
        self.scene.sensors["observer_camera"] = self._observer_camera

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

        # Pre-compute observer camera quaternion via look-at math.
        # Camera at offset (-1.2, 0, 0.7) from drone. Camera should look TOWARD drone.
        # Camera convention: -Z = view direction, +Y = up, +X = right
        from scipy.spatial.transform import Rotation as scipy_R
        cam_offset = np.array([-1.2, 0.0, 0.7])
        # Direction from camera to drone (this is what -Z should align with)
        forward = -cam_offset / np.linalg.norm(cam_offset)  # (0.866, 0, -0.503)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)
        # Rotation matrix: columns are camera basis vectors expressed in world coords
        # Camera +X = right, +Y = up, +Z = -forward (since -Z = forward)
        rot_mat = np.column_stack([right, cam_up, -forward])
        scipy_quat = scipy_R.from_matrix(rot_mat).as_quat()  # (x, y, z, w)
        # Isaac uses (w, x, y, z) order
        self._observer_cam_quat = torch.tensor(
            [scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]],
            dtype=torch.float32,
        )

    def _update_camera_pose(self):
        # Update onboard cameras using parent logic (body-frame composition)
        super()._update_camera_pose()
        # Chase camera: follows drone position AND yaw, but ignores roll/pitch
        # so the view stays level and the drone is always centered.
        # Camera offset is in drone's body frame (yaw-only): -1.2m behind, +0.7m above
        if hasattr(self, "_observer_camera"):
            drone_pos = self._robot.data.root_pos_w  # (N, 3) world
            drone_quat = self._robot.data.root_quat_w  # (N, 4) (w, x, y, z)

            # Extract yaw-only quaternion (project to world Z axis)
            # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            w, x, y, z = drone_quat[:, 0], drone_quat[:, 1], drone_quat[:, 2], drone_quat[:, 3]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            half = yaw * 0.5
            yaw_w = torch.cos(half)
            yaw_z = torch.sin(half)
            zeros = torch.zeros_like(yaw_w)
            yaw_quat = torch.stack([yaw_w, zeros, zeros, yaw_z], dim=-1)  # (N, 4)

            # Rotate the body-frame offset (-1.2, 0, 0.7) by yaw to get world offset
            from isaaclab.utils.math import quat_apply, quat_mul
            body_offset = torch.tensor([-1.2, 0.0, 0.7], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            world_offset = quat_apply(yaw_quat, body_offset)
            cam_pos = drone_pos + world_offset

            # Camera rotation: yaw_quat composed with the precomputed look-at quat
            # (the look-at quat looks toward drone in body frame; yaw_quat rotates it to world)
            cam_offset_quat = self._observer_cam_quat.to(self.device).unsqueeze(0).expand(self.num_envs, -1)
            cam_quat = quat_mul(yaw_quat, cam_offset_quat)
            self._observer_camera._view.set_world_poses(cam_pos, cam_quat)


# Register the observer-equipped env as a new gym ID
gym.register(
    id="Isaac-VLADrone-Direct-Observer-v0",
    entry_point=f"{__name__}:VLADroneEnvWithObserver",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "vla.vla_drone_env:VLADroneEnvCfg"},
)



class OnboardCameraWrapper(gym.Wrapper):
    """Render observer camera (third-person) + 4 onboard cameras + diagnostics.

    Layout (1280x720):
    ┌────────────────────────────┬──────────────────┐
    │  Command: "fly to ..."     │                  │
    ├────────────────────────────┤   FRONT | RIGHT  │
    │                            │   BACK  | LEFT   │
    │    Third-person observer   ├──────────────────┤
    │    (top-down or angled)    │   Diagnostics    │
    │                            │                  │
    └────────────────────────────┴──────────────────┘
    """

    _OUT_W = 1280
    _OUT_H = 720
    _BANNER_H = 50
    _MAIN_W = 820   # observer camera width
    _SIDE_W = 460   # right panel width

    def __init__(self, env):
        super().__init__(env)
        self._step_count = 0

    def render(self):
        unwrapped = self.unwrapped
        frame = np.zeros((self._OUT_H, self._OUT_W, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- Main view: third-person observer camera (left side) ---
        main_h = self._OUT_H - self._BANNER_H
        observer_img = np.zeros((main_h, self._MAIN_W, 3), dtype=np.uint8)

        if hasattr(unwrapped, "_observer_camera"):
            try:
                obs_rgb = unwrapped._observer_camera.data.output["rgb"][0, :, :, :3]  # (H, W, 3) uint8
                obs_np = obs_rgb.cpu().numpy().astype(np.uint8)
                observer_img = cv2.resize(obs_np, (self._MAIN_W, main_h))
                # Convert RGB → BGR for OpenCV
                observer_img = cv2.cvtColor(observer_img, cv2.COLOR_RGB2BGR)
                cv2.putText(observer_img, "Third-Person View", (15, 25),
                            font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                cv2.putText(observer_img, f"Observer cam: {e}", (15, 25),
                            font, 0.5, (200, 200, 200), 1)

        # --- Top-down minimap (PiP in bottom-left of observer view) ---
        # World-fixed orientation: +X world → right, +Y world → up (top of image).
        # Drone icon has a heading arrow showing where it's currently facing so the
        # user can correlate observer view ("what's ahead of the drone") with map.
        if hasattr(unwrapped, "_robot") and hasattr(unwrapped, "_obj_pos_w"):
            pos = unwrapped._robot.data.root_pos_w[0].cpu().numpy()
            origin = unwrapped._terrain.env_origins[0].cpu().numpy()
            quat = unwrapped._robot.data.root_quat_w[0].cpu().numpy()  # (w, x, y, z)
            yaw = math.atan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
            )

            mm_size = 200
            minimap = np.zeros((mm_size, mm_size, 3), dtype=np.uint8)
            minimap[:] = (40, 40, 40)
            arena_half = 2.5

            def world_to_mm(wx, wy):
                lx, ly = wx - origin[0], wy - origin[1]
                mx = int((lx / arena_half + 1.0) * 0.5 * (mm_size - 20) + 10)
                my = int((-ly / arena_half + 1.0) * 0.5 * (mm_size - 20) + 10)
                return mx, my

            obj_colors = [(0, 0, 220), (220, 0, 0), (0, 200, 0)]  # cube=red, sphere=blue, cyl=green
            obj_letters = ["C", "S", "Y"]  # Cube, Sphere, cYlinder
            tgt_idx = unwrapped._target_obj_idx[0].item()
            for i in range(3):
                op = unwrapped._obj_pos_w[0, i].cpu().numpy()
                ox, oy = world_to_mm(op[0], op[1])
                # Larger filled circle for object
                cv2.circle(minimap, (ox, oy), 9, obj_colors[i], -1)
                # Label letter inside (in white)
                cv2.putText(minimap, obj_letters[i], (ox - 4, oy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                # Target gets a thick magenta highlight (very distinct from drone's yellow)
                if i == tgt_idx:
                    cv2.circle(minimap, (ox, oy), 16, (255, 0, 255), 3)
                    cv2.putText(minimap, "TGT", (ox - 12, oy - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

            # Drone with yellow ring + heading arrow + label
            dx, dy = world_to_mm(pos[0], pos[1])
            cv2.circle(minimap, (dx, dy), 6, (255, 255, 255), -1)
            cv2.circle(minimap, (dx, dy), 9, (0, 255, 255), 2)
            # Heading arrow: 20px long, pointing in world direction of drone forward (+X body).
            # In image coords, +X world = +x pixel, +Y world = -y pixel (image y is flipped).
            arrow_len = 22
            hx = int(dx + arrow_len * math.cos(yaw))
            hy = int(dy - arrow_len * math.sin(yaw))
            cv2.arrowedLine(minimap, (dx, dy), (hx, hy), (0, 255, 255), 2, tipLength=0.35)
            cv2.putText(minimap, "DRONE", (dx - 18, dy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(minimap, (0, 0), (mm_size - 1, mm_size - 1), (100, 100, 100), 1)
            # Compass labels so the user knows orientation: +X world = RIGHT, +Y world = UP
            cv2.putText(minimap, "+X", (mm_size - 22, mm_size // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(minimap, "+Y", (mm_size // 2 - 10, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

            # Place minimap in bottom-left of observer view
            mm_y = main_h - mm_size - 10
            mm_x = 10
            observer_img[mm_y:mm_y + mm_size, mm_x:mm_x + mm_size] = minimap
            cv2.putText(observer_img, "Map", (mm_x + 5, mm_y - 5),
                        font, 0.5, (200, 200, 200), 1)

        frame[self._BANNER_H:, :self._MAIN_W] = observer_img

        # --- 4-camera 2x2 grid (top-right) ---
        cam_grid_size = self._SIDE_W
        cam_cell = cam_grid_size // 2  # 230
        if hasattr(unwrapped, "_cached_rgb"):
            try:
                rgb = unwrapped._cached_rgb[0]  # (4, 224, 224, 3)
                views = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                labels = ["FRONT", "RIGHT", "BACK", "LEFT"]
                for i, (view, label) in enumerate(zip(views, labels)):
                    # Isaac TiledCamera outputs RGB; OpenCV expects BGR. Without
                    # this swap, red cubes render blue and blue spheres render red
                    # in the PiP — making it look like "find the red cube" points
                    # at a blue object.
                    view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                    resized = cv2.resize(view, (cam_cell, cam_cell))
                    row, col = divmod(i, 2)
                    y0 = self._BANNER_H + row * cam_cell
                    x0 = self._MAIN_W + col * cam_cell
                    if y0 + cam_cell <= self._OUT_H and x0 + cam_cell <= self._OUT_W:
                        frame[y0:y0 + cam_cell, x0:x0 + cam_cell] = resized
                        cv2.putText(frame, label, (x0 + 5, y0 + 20),
                                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x0, y0), (x0 + cam_cell, y0 + cam_cell), (80, 80, 80), 1)
            except Exception:
                pass

        # --- Command text banner ---
        cmd = ""
        if hasattr(unwrapped, "_current_commands") and unwrapped._current_commands:
            cmd = unwrapped._current_commands[0]
        cv2.rectangle(frame, (0, 0), (self._OUT_W, self._BANNER_H), (30, 30, 30), -1)
        if cmd:
            cv2.putText(frame, f'Command: "{cmd}"', (15, 35),
                        font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Diagnostics panel (bottom-right, below camera grid) ---
        px = self._MAIN_W + 15
        py = self._BANNER_H + 2 * cam_cell + 25

        if hasattr(unwrapped, "_robot"):
            pos = unwrapped._robot.data.root_pos_w[0].cpu().numpy()
            origin = unwrapped._terrain.env_origins[0].cpu().numpy()
            local_pos = pos - origin

        if hasattr(unwrapped, "_target_obj_idx") and hasattr(unwrapped, "_obj_pos_w"):
            obj_names = ["Red Cube", "Blue Sphere", "Green Cylinder"]
            tgt_idx = unwrapped._target_obj_idx[0].item()
            tgt_pos = unwrapped._obj_pos_w[0, tgt_idx].cpu().numpy()
            dist = np.linalg.norm(pos - tgt_pos)

            if dist < 0.35:
                color, status = (0, 255, 0), "SUCCESS"
            elif dist < 1.0:
                color, status = (0, 200, 255), "APPROACHING"
            else:
                color, status = (100, 100, 255), "NAVIGATING"

            cv2.putText(frame, f"Target: {obj_names[tgt_idx]}", (px, py), font, 0.6, (0, 255, 100), 2)
            py += 28
            cv2.putText(frame, f"Dist: {dist:.2f}m  {status}", (px, py), font, 0.6, color, 2)
            py += 28

        if hasattr(unwrapped, "_robot"):
            vel = unwrapped._robot.data.root_lin_vel_w[0].cpu().numpy()
            speed = np.linalg.norm(vel)
            cv2.putText(frame, f"Speed: {speed:.1f} m/s  Step: {self._step_count}", (px, py), font, 0.5, (180, 180, 180), 1)

        self._step_count += 1
        return frame


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    env_cfg = VLADroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # Render every physics step for smooth video AND step the env once per
    # physics tick so the onboard-camera PiP stays in sync. (Training uses
    # decimation=2 for throughput, but in play the PiP was corrupted when we
    # tried that.) The policy being called 2x more often is acceptable for
    # playback visualization purposes.
    env_cfg.sim.render_interval = 1
    env_cfg.decimation = 1

    # Capture 4-cam obs every step too, so the PiP grid stays in sync with the
    # third-person observer view (default is 4 for training throughput; in play
    # the 3-step lag made the PiP look like a different scenario).
    env_cfg.camera_every_n = 1

    # Long episodes so we can watch full flights
    env_cfg.episode_length_s = 60.0
    # Widen success threshold so it doesn't terminate too early
    env_cfg.success_threshold = 0.05  # practically disables termination on arrival

    # Viewer resolution (used by RecordVideo even though main content is our custom render)
    env_cfg.viewer.resolution = (1280, 720)

    video_dir = os.path.join(_DRONE_PROJECT, "videos", "vla")
    os.makedirs(video_dir, exist_ok=True)

    for old in glob.glob(os.path.join(video_dir, "vla_playback-*.mp4")):
        os.remove(old)

    warmup_steps = 50

    # Create environment with the observer-camera-equipped subclass
    if args_cli.video:
        env = gym.make("Isaac-VLADrone-Direct-Observer-v0", cfg=env_cfg, render_mode="rgb_array")
    else:
        env = gym.make("Isaac-VLADrone-Direct-Observer-v0", cfg=env_cfg)

    env.metadata["render_fps"] = 50

    # Overlay onboard cameras + diagnostics + third-person observer view
    if args_cli.video:
        env = OnboardCameraWrapper(env)

    # Wrap with RecordVideo then RSL-RL wrapper
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_dir,
            name_prefix="vla_playback",
            step_trigger=lambda step: step == warmup_steps,
            video_length=args_cli.num_steps,
            disable_logger=True,
        )

    env_unwrapped = env.unwrapped if not hasattr(env, 'unwrapped') else env.unwrapped
    env = RslRlVecEnvWrapper(env)

    device = env_cfg.sim.device

    # Construct hierarchical actor and load weights
    print("[INFO] Loading Hierarchical VLA actor (PaliGemma + frozen waypoint policy)...")
    waypoint_ckpt = os.path.join(_DRONE_PROJECT, "model_2998_waypoint.pt")
    actor = HierarchicalVLAActor(
        waypoint_checkpoint_path=waypoint_ckpt,
        paligemma_model_name="google/paligemma-3b-pt-224",
        init_std=0.3,
        target_range=3.0,
        lstm_hidden_dim=256,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("actor_state_dict", {}))
    actor_state = {k.removeprefix("actor."): v for k, v in state_dict.items() if k.startswith("actor.")}
    if not actor_state:
        actor_state = state_dict
    tgt_sd = actor.state_dict()
    skipped_shape = []
    filtered = {}
    for k, v in actor_state.items():
        if k in tgt_sd and tgt_sd[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(tgt_sd[k].shape)))
            continue
        filtered[k] = v
    missing, unexpected = actor.load_state_dict(filtered, strict=False)
    print(f"[INFO] Loaded actor: {len(missing)} missing, {len(unexpected)} unexpected, {len(skipped_shape)} shape-mismatched keys")
    for k, src_shape, tgt_shape in skipped_shape:
        print(f"   skip (shape) {k}: ckpt {src_shape} vs model {tgt_shape}")
    actor.eval()

    obs = env.get_observations()

    total_steps = warmup_steps + args_cli.num_steps
    print(f"[INFO] Warming up {warmup_steps} steps, then recording {args_cli.num_steps} steps")

    with torch.inference_mode():
        for step in range(total_steps):
            obs_dict = {k: obs[k] for k in obs.keys()}
            actions = actor(obs_dict, stochastic_output=False)
            obs, _, dones, extras = env.step(actions)
            obs = obs.to(device)
            actor.paligemma.clear_cache()

            if step % 100 == 0:
                cmd = env_unwrapped._current_commands[0] if hasattr(env_unwrapped, '_current_commands') else "?"
                print(f"  Step {step}/{total_steps} | Command: {cmd}")
                # Diagnostic: compare cached _obj_pos_w to actual prim world poses.
                # If they disagree, set_world_poses() isn't taking effect and the
                # minimap is drawing from stale data.
                try:
                    cached = env_unwrapped._obj_pos_w[0].cpu().numpy()
                    c_pos, _ = env_unwrapped._cube_view.get_world_poses()
                    s_pos, _ = env_unwrapped._sphere_view.get_world_poses()
                    y_pos, _ = env_unwrapped._cylinder_view.get_world_poses()
                    actual = np.stack([c_pos[0].cpu().numpy(), s_pos[0].cpu().numpy(), y_pos[0].cpu().numpy()])
                    drone = env_unwrapped._robot.data.root_pos_w[0].cpu().numpy()
                    names = ["cube ", "sph  ", "cyl  "]
                    print(f"    drone @ ({drone[0]:+.2f},{drone[1]:+.2f},{drone[2]:+.2f})")
                    for i in range(3):
                        diff = np.linalg.norm(cached[i] - actual[i])
                        mark = " MISMATCH" if diff > 0.05 else ""
                        print(f"    {names[i]} cached=({cached[i,0]:+.2f},{cached[i,1]:+.2f},{cached[i,2]:+.2f})  actual=({actual[i,0]:+.2f},{actual[i,1]:+.2f},{actual[i,2]:+.2f}) Δ={diff:.3f}m{mark}")
                except Exception as e:
                    print(f"    [diag] failed: {e}")

    env.close()
    # Find recorded video
    vids = sorted(glob.glob(os.path.join(video_dir, "vla_playback-*.mp4")))
    if vids:
        print(f"[INFO] Video saved to {vids[-1]}")
    else:
        print("[WARN] No video file found")


if __name__ == "__main__":
    main()
    simulation_app.close()

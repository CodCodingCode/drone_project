"""Play back a trained lang nav policy and record video.

Shows a glowing sphere above the target object so viewers can see
which object the drone is flying to.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/lang_nav/play.py

Optional args:
    --checkpoint   Path to model .pt (defaults to best known checkpoint)
    --num_envs     Number of environments (default: 1)
    --num_steps    Steps to record (default: 500)
    --video        Record video to disk
    --headless     Run without GUI
"""

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rsl_rl")
os.environ["OMNI_LOG_LEVEL"] = "ERROR"

from isaaclab.app import AppLauncher

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULT_CKPT = os.path.join(
    _DRONE_PROJECT,
    "logs", "rsl_rl", "lang_drone_direct",
    "2026-03-27_19-01-28", "model_2999.pt",
)

parser = argparse.ArgumentParser(description="Play trained lang nav policy.")
parser.add_argument("--checkpoint", type=str, default=_DEFAULT_CKPT, help="Path to model .pt file.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--video", action="store_true", help="Record video to disk.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import lang_nav  # noqa: F401 — registers Isaac-LangDrone-Direct-v0

from lang_nav.lang_drone_env import LangDroneEnvCfg
from lang_nav.agents.rsl_rl_ppo_cfg import LangDronePPORunnerCfg

import cv2
import numpy as np


class TextOverlayWrapper(gym.Wrapper):
    """Burn command text + onboard camera PiP onto rendered video frames."""

    _PIP_SCALE = 3  # upscale 64x64 → 192x192
    _PIP_MARGIN = 16
    _PIP_BORDER = 2

    def render(self):
        frame = self.env.render()
        if frame is None:
            return frame

        unwrapped = self.unwrapped
        frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]

        # --- Command text banner at the top ---
        cmd = ""
        if hasattr(unwrapped, "_current_commands") and unwrapped._current_commands:
            cmd = unwrapped._current_commands[0]
        if cmd:
            text = f'Command: "{cmd}"'
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 1.2, 2
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            banner_h = th + baseline + 30
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            x = (w - tw) // 2
            y = th + 15
            cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # --- Onboard camera PiP (bottom-right corner) ---
        if hasattr(unwrapped, "_camera"):
            try:
                rgb = unwrapped._camera.data.output["rgb"][0, :, :, :3]  # (64, 64, 3)
                pip = rgb.cpu().numpy().astype(np.uint8)
                pip_size = pip.shape[0] * self._PIP_SCALE  # 192
                pip = cv2.resize(pip, (pip_size, pip_size), interpolation=cv2.INTER_NEAREST)

                # Position: bottom-right with margin
                m = self._PIP_MARGIN
                b = self._PIP_BORDER
                y1 = h - m - pip_size
                x1 = w - m - pip_size

                # White border
                frame[y1 - b : y1 + pip_size + b, x1 - b : x1 + pip_size + b] = 255
                # Inset the PiP
                frame[y1 : y1 + pip_size, x1 : x1 + pip_size] = pip

                # Label above PiP
                label = "Drone Camera"
                lscale, lthick = 0.6, 1
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lscale, lthick)
                lx = x1 + (pip_size - lw) // 2
                ly = y1 - b - 6
                cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, lscale,
                            (255, 255, 255), lthick, cv2.LINE_AA)
            except Exception:
                pass  # camera not ready during warmup

        return frame


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    env_cfg = LangDroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # Render every physics step for smooth video
    env_cfg.sim.render_interval = 1
    env_cfg.decimation = 1

    # Quality rendering — denoiser + DLAA to eliminate noise/grain
    from isaaclab.sim import RenderCfg
    env_cfg.sim.render = RenderCfg(
        enable_dl_denoiser=True,
        antialiasing_mode="DLAA",
        dome_light_upper_lower_strategy=4,
        enable_direct_lighting=True,
        samples_per_pixel=2,
    )

    # Camera tracks the drone, wide enough to see the 3 objects
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (2.5, 2.5, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env_cfg.viewer.resolution = (1920, 1080)

    video_dir = os.path.join(_DRONE_PROJECT, "videos", "lang_nav")
    os.makedirs(video_dir, exist_ok=True)

    # Clean up old lang nav playback videos
    for old in glob.glob(os.path.join(video_dir, "lang_nav_playback-*.mp4")):
        os.remove(old)

    warmup_steps = 50

    # Create environment
    if args_cli.video:
        env = gym.make("Isaac-LangDrone-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
    else:
        env = gym.make("Isaac-LangDrone-Direct-v0", cfg=env_cfg)

    # Encode at 50fps — 0.5x slow-mo (smooth & cinematic)
    env.metadata["render_fps"] = 50

    # Overlay command text on rendered frames
    if args_cli.video:
        env = TextOverlayWrapper(env)

    # Wrap with RecordVideo then RSL-RL wrapper
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_dir,
            name_prefix="lang_nav_playback",
            step_trigger=lambda step: step == warmup_steps,
            video_length=args_cli.num_steps,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env)

    agent_cfg = LangDronePPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device

    from importlib.metadata import version as pkg_version
    handle_deprecated_rsl_rl_cfg(agent_cfg, pkg_version("rsl-rl-lib"))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()

    total_steps = warmup_steps + args_cli.num_steps
    print(f"[INFO] Warming up {warmup_steps} steps, then running {args_cli.num_steps} steps")
    if args_cli.video:
        print(f"[INFO] Recording to {video_dir}/")

    with torch.inference_mode():
        for step in range(total_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            if step % 100 == 0:
                print(f"  Step {step}/{total_steps}")

    env.close()
    print("[INFO] Done.")
    if args_cli.video:
        print(f"[INFO] Video saved to {video_dir}/")


if __name__ == "__main__":
    main()
    simulation_app.close()

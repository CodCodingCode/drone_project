"""Play back a trained Pi0 policy and record video with text overlay + drone POV.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/pi/play.py \
        --checkpoint <path/to/model.pt> --enable_cameras
"""

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["OMNI_LOG_LEVEL"] = "ERROR"

from isaaclab.app import AppLauncher

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Play trained Pi0 policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=750)
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

import pi  # noqa: F401

from vla.vla_drone_env import VLADroneEnvCfg
from pi.pi0_policy import Pi0ActorModel


def add_text_overlay(frame: np.ndarray, command: str, step: int) -> np.ndarray:
    """Add text command and step counter to the frame."""
    h, w = frame.shape[:2]
    # Semi-transparent black bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    # Command text
    cv2.putText(frame, f'Command: "{command}"', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Step: {step}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def add_drone_pov(frame: np.ndarray, pov_image: np.ndarray) -> np.ndarray:
    """Add drone's camera POV as picture-in-picture in bottom-right corner."""
    h, w = frame.shape[:2]
    pov_h, pov_w = 160, 160
    # Resize POV to thumbnail
    pov_resized = cv2.resize(pov_image, (pov_w, pov_h))
    # Add border
    cv2.rectangle(frame, (w - pov_w - 12, h - pov_h - 32),
                  (w - 2, h - 2), (255, 255, 255), 2)
    # Place POV
    frame[h - pov_h - 10:h - 10, w - pov_w - 10:w - 10] = pov_resized
    # Label
    cv2.putText(frame, "Drone POV", (w - pov_w - 10, h - pov_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    env_cfg = VLADroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.sim.render_interval = 1
    env_cfg.decimation = 1

    video_dir = os.path.join(_DRONE_PROJECT, "videos")
    os.makedirs(video_dir, exist_ok=True)

    for old in glob.glob(os.path.join(video_dir, "pi0_playback*")):
        os.remove(old)

    env = gym.make("Isaac-Pi0Drone-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
    env_unwrapped = env.unwrapped
    env = RslRlVecEnvWrapper(env)

    device = env_cfg.sim.device

    # Construct actor and load weights
    print("[INFO] Loading Pi0 actor...")
    actor = Pi0ActorModel(
        flight_state_dim=9, action_dim=4, hidden_dims=[256, 256],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("actor_state_dict", {}))
    actor_state = {k.removeprefix("actor."): v for k, v in state_dict.items() if k.startswith("actor.")}
    if not actor_state:
        actor_state = state_dict
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    print(f"[INFO] Loaded actor: {len(missing)} missing, {len(unexpected)} unexpected keys")
    actor.eval()

    # Setup video writer
    video_path = os.path.join(video_dir, "pi0_playback.mp4")
    fps = 30
    writer = None

    obs = env.get_observations()

    warmup_steps = 30
    total_steps = warmup_steps + args_cli.num_steps
    print(f"[INFO] Warming up {warmup_steps} steps, then recording {args_cli.num_steps} steps")

    with torch.inference_mode():
        for step in range(total_steps):
            obs_dict = {k: obs[k] for k in obs.keys()}
            actions = actor(obs_dict, stochastic_output=False)
            obs, _, dones, extras = env.step(actions)
            obs = obs.to(device)
            actor.pi0.clear_cache()

            # Record after warmup
            if step >= warmup_steps:
                # Get the rendered frame from Isaac Sim
                frame = env_unwrapped.render()
                if frame is not None:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).clip(0, 255).astype(np.uint8)

                    # Get current command text
                    command = env_unwrapped._current_commands[0] if hasattr(env_unwrapped, '_current_commands') else "N/A"

                    # Get drone POV (the 64x64 camera image)
                    pov = obs_dict.get("rgb", None)
                    if pov is not None:
                        pov_np = (pov[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    else:
                        pov_np = np.zeros((64, 64, 3), dtype=np.uint8)

                    # Add overlays
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 else frame
                    frame_bgr = add_text_overlay(frame_bgr, command, step - warmup_steps)
                    frame_bgr = add_drone_pov(frame_bgr, cv2.cvtColor(pov_np, cv2.COLOR_RGB2BGR))

                    if writer is None:
                        h, w = frame_bgr.shape[:2]
                        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        print(f"[INFO] Recording at {w}x{h} to {video_path}")

                    writer.write(frame_bgr)

            if step % 100 == 0:
                cmd = env_unwrapped._current_commands[0] if hasattr(env_unwrapped, '_current_commands') else "?"
                print(f"  Step {step}/{total_steps} | Command: {cmd}")

    if writer is not None:
        writer.release()
    env.close()
    print(f"[INFO] Video saved to {video_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

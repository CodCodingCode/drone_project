"""Play back a trained waypoint nav policy and record video.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/waypoint_nav/play.py \
        --checkpoint /home/ubuntu/drone_project/logs/rsl_rl/waypoint_nav/<timestamp>/model_399.pt \
        --num_envs 1 --headless --video

Without --headless, it opens the Isaac Sim GUI so you can watch live.
With --video, it records to drone_project/videos/waypoint_nav_playback.mp4.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained waypoint nav policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")
parser.add_argument("--num_steps", type=int, default=500, help="Steps to run.")
parser.add_argument("--video", action="store_true", help="Record video to disk.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# When recording video we need cameras enabled
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------

import os
import sys

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

_DRONE_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import waypoint_nav  # noqa: F401 — registers Isaac-WaypointNav-Direct-v0

from waypoint_nav.waypoint_nav_env import WaypointNavEnvCfg
from waypoint_nav.agents.rsl_rl_ppo_cfg import WaypointNavPPORunnerCfg


def main():
    env_cfg = WaypointNavEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # Close-up camera tracking the drone (matches hover play.py)
    env_cfg.viewer.eye = (1.0, 1.0, 0.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # Enable rendering every step for video capture
    if args_cli.video:
        env_cfg.sim.render_interval = 1

    # Video directory
    video_dir = os.path.join(_DRONE_PROJECT, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # Create environment — use render_mode="rgb_array" for video recording
    if args_cli.video:
        env = gym.make("Isaac-WaypointNav-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
        # Wrap with RecordVideo before the RSL-RL wrapper
        env = gym.wrappers.RecordVideo(
            env,
            video_dir,
            name_prefix="waypoint_nav_playback",
            step_trigger=lambda step: step == 0,
            video_length=args_cli.num_steps,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to: {video_dir}")
    else:
        env = gym.make("Isaac-WaypointNav-Direct-v0", cfg=env_cfg)

    # Wrap for RSL-RL compatibility (must be last wrapper)
    env = RslRlVecEnvWrapper(env)

    # Build runner with same config used during training
    agent_cfg = WaypointNavPPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device

    from importlib.metadata import version as pkg_version
    handle_deprecated_rsl_rl_cfg(agent_cfg, pkg_version("rsl-rl-lib"))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    resume_path = os.path.abspath(args_cli.checkpoint)
    runner.load(resume_path)

    # Get inference policy
    policy = runner.get_inference_policy(device=env.device)

    # Run policy
    obs, _ = env.reset()

    print(f"[INFO] Running for {args_cli.num_steps} steps...")
    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            if step % 100 == 0:
                print(f"  Step {step}/{args_cli.num_steps}")

    print("[INFO] Done.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

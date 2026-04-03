"""Play back a trained waypoint nav policy and record video.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/waypoint_nav/play.py \
        --checkpoint /home/ubuntu/drone_project/logs/rsl_rl/waypoint_nav/<timestamp>/model_399.pt

Without --headless, it opens the Isaac Sim GUI so you can watch live.
Always records video to drone_project/videos/waypoint_nav_playback.mp4.
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

parser = argparse.ArgumentParser(description="Play trained waypoint nav policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=750)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless + video (matches hover play.py)
args_cli.headless = True
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

import waypoint_nav  # noqa: F401 — registers Isaac-WaypointNav-Direct-v0

from waypoint_nav.waypoint_nav_env import WaypointNavEnvCfg
from waypoint_nav.agents.rsl_rl_ppo_cfg import WaypointNavPPORunnerCfg


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    env_cfg = WaypointNavEnvCfg()
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

    # Track the drone — camera follows the robot
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (1.0, 1.0, 0.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env_cfg.viewer.resolution = (1920, 1080)

    video_dir = os.path.join(_DRONE_PROJECT, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # Clean up old waypoint nav playback videos
    for old in glob.glob(os.path.join(video_dir, "waypoint_nav_playback-*.mp4")):
        os.remove(old)

    # Warmup steps let the renderer fully initialise before recording
    warmup_steps = 50

    env = gym.make("Isaac-WaypointNav-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
    # Encode at 50fps — 0.5x slow-mo (smooth & cinematic)
    env.metadata["render_fps"] = 50
    env = gym.wrappers.RecordVideo(
        env,
        video_dir,
        name_prefix="waypoint_nav_playback",
        step_trigger=lambda step: step == warmup_steps,
        video_length=args_cli.num_steps,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env)

    agent_cfg = WaypointNavPPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device

    from importlib.metadata import version as pkg_version
    handle_deprecated_rsl_rl_cfg(agent_cfg, pkg_version("rsl-rl-lib"))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()

    total_steps = warmup_steps + args_cli.num_steps
    print(f"[INFO] Warming up {warmup_steps} steps, then recording {args_cli.num_steps} steps to {video_dir}/")
    with torch.inference_mode():
        for step in range(total_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if step % 100 == 0:
                print(f"  Step {step}/{total_steps}")

    env.close()
    print(f"[INFO] Video saved to {video_dir}/")


if __name__ == "__main__":
    main()
    simulation_app.close()

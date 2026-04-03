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


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    env_cfg = LangDroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    if args_cli.video:
        env_cfg.sim.render_interval = 1

    # Camera tracks the drone, wide enough to see the 3 objects
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (2.5, 2.5, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env_cfg.viewer.resolution = (1920, 1080)

    video_dir = os.path.join(_DRONE_PROJECT, "videos")
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

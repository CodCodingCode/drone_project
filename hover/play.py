"""Play back a trained hover policy and record video.

Auto-detects the latest checkpoint from logs/rsl_rl/hover_pretrain/.
Always runs headless with video recording.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/hover/play.py
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


def find_latest_checkpoint() -> str:
    """Find the best checkpoint: latest run dir, then highest iteration number."""
    log_root = os.path.join(_DRONE_PROJECT, "logs", "rsl_rl", "hover_pretrain")
    pts = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {log_root}")

    def sort_key(p):
        # Sort by (run directory name descending, iteration number descending)
        run_dir = os.path.basename(os.path.dirname(p))
        fname = os.path.basename(p)
        # Extract iteration number from model_123.pt
        iter_num = int(fname.replace("model_", "").replace(".pt", ""))
        return (run_dir, iter_num)

    pts.sort(key=sort_key)
    return pts[-1]


parser = argparse.ArgumentParser(description="Play trained hover policy.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model .pt (auto-detects latest if omitted).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=750)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless + video
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

import hover  # noqa: F401

from hover.hover_env import HoverEnvCfg
from hover.agents.rsl_rl_ppo_cfg import HoverPPORunnerCfg


def main():
    # Auto-detect checkpoint
    ckpt_path = args_cli.checkpoint or find_latest_checkpoint()
    ckpt_path = os.path.abspath(ckpt_path)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    env_cfg = HoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    # render_interval must be 1 for video capture — the rgb annotator needs
    # the renderer to run every step, otherwise frames come back black
    env_cfg.sim.render_interval = 1
    # Capture every physics step (decimation=1) for smoother video.
    # Sim dt is 1/100, so 750 steps = 7.5s of sim time at 100 frames.
    env_cfg.decimation = 1

    # Quality rendering — denoiser + DLAA to eliminate ground noise/grain
    from isaaclab.sim import RenderCfg
    env_cfg.sim.render = RenderCfg(
        enable_dl_denoiser=True,
        antialiasing_mode="DLAA",
        dome_light_upper_lower_strategy=4,
        enable_direct_lighting=True,
        samples_per_pixel=2,
    )

    # Track the drone — camera follows the robot so it's always visible
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (1.0, 1.0, 0.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env_cfg.viewer.resolution = (1920, 1080)

    video_dir = os.path.join(_DRONE_PROJECT, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # Clean up old hover playback videos
    for old in glob.glob(os.path.join(video_dir, "hover_playback-*.mp4")):
        os.remove(old)

    # Warmup steps let the renderer fully initialise before recording
    warmup_steps = 50

    env = gym.make("Isaac-Hover-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
    # Encode at 50fps instead of 100fps → 0.5x slow-mo (smooth & cinematic)
    env.metadata["render_fps"] = 50
    env = gym.wrappers.RecordVideo(
        env,
        video_dir,
        name_prefix="hover_playback",
        step_trigger=lambda step: step == warmup_steps,
        video_length=args_cli.num_steps,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env)

    agent_cfg = HoverPPORunnerCfg()
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

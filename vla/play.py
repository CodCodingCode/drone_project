"""Play back a trained VLA policy and record video.

Launch:
    cd /home/ubuntu/IsaacLab
    ./isaaclab.sh -p /home/ubuntu/drone_project/vla/play.py \
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

parser = argparse.ArgumentParser(description="Play trained VLA policy.")
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

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

if _DRONE_PROJECT not in sys.path:
    sys.path.insert(0, _DRONE_PROJECT)

import vla  # noqa: F401

from vla.vla_drone_env import VLADroneEnvCfg
from vla.vla_policy import VLAActorModel, VLACriticModel


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    env_cfg = VLADroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.sim.render_interval = 1
    env_cfg.decimation = 1

    from isaaclab.sim import RenderCfg
    env_cfg.sim.render = RenderCfg(
        enable_dl_denoiser=True,
        antialiasing_mode="DLAA",
        dome_light_upper_lower_strategy=4,
        enable_direct_lighting=True,
        samples_per_pixel=2,
    )

    from isaaclab.envs.common import ViewerCfg
    env_cfg.viewer = ViewerCfg(
        eye=(1.0, 1.0, 0.8),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset_root",
        asset_name="robot",
        resolution=(1920, 1080),
    )

    video_dir = os.path.join(_DRONE_PROJECT, "videos")
    os.makedirs(video_dir, exist_ok=True)

    for old in glob.glob(os.path.join(video_dir, "vla_playback-*.mp4")):
        os.remove(old)

    warmup_steps = 50

    env = gym.make("Isaac-VLADrone-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
    env.metadata["render_fps"] = 50
    env = gym.wrappers.RecordVideo(
        env, video_dir, name_prefix="vla_playback",
        step_trigger=lambda step: step == warmup_steps,
        video_length=args_cli.num_steps, disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env)

    device = env_cfg.sim.device

    # Construct actor and load weights
    actor = VLAActorModel(
        flight_state_dim=9, action_dim=4, hidden_dims=[256, 256],
        paligemma_model_name="google/paligemma-3b-pt-224",
        lora_rank=8, lora_alpha=16.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt.get("actor_state_dict", {}), strict=False)
    actor.eval()

    obs, _ = env.reset()

    total_steps = warmup_steps + args_cli.num_steps
    print(f"[INFO] Warming up {warmup_steps} steps, then recording {args_cli.num_steps} steps")
    with torch.inference_mode():
        for step in range(total_steps):
            actions = actor(obs)
            obs, _, _, _ = env.step(actions)
            if step % 100 == 0:
                print(f"  Step {step}/{total_steps}")

    env.close()
    print(f"[INFO] Video saved to {video_dir}/")


if __name__ == "__main__":
    main()
    simulation_app.close()

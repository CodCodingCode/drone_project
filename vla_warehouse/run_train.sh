#!/bin/bash
# Launcher for VLA-in-warehouse training. Mirrors run_vla_train.sh's conda +
# env-var setup so isaaclab.sh picks up the `isaac` conda env's Python
# (not base). Usage:
#     bash ~/drone_project/vla_warehouse/run_train.sh             # defaults
#     bash ~/drone_project/vla_warehouse/run_train.sh --num_envs 32 --max_iterations 1

set -euo pipefail

source /home/ubuntu/miniconda3/bin/activate isaac
cd /home/ubuntu/IsaacLab

export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Default args — override from CLI. Passing any --num_envs etc. below
# wins over these defaults because Python argparse takes last value.
exec stdbuf -oL ./isaaclab.sh -p /home/ubuntu/drone_project/vla_warehouse/train.py \
    --num_envs 64 \
    --max_iterations 3000 \
    --headless \
    --enable_cameras \
    "$@"

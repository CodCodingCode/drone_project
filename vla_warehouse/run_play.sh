#!/bin/bash
# Launcher for VLA-in-warehouse playback. Usage:
#     bash ~/drone_project/vla_warehouse/run_play.sh --checkpoint <path.pt>

set -euo pipefail

source /home/ubuntu/miniconda3/bin/activate isaac
cd /home/ubuntu/IsaacLab

export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

exec stdbuf -oL ./isaaclab.sh -p /home/ubuntu/drone_project/vla_warehouse/play.py \
    --num_steps 1500 \
    --video \
    "$@"

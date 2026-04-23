#!/bin/bash
# Navigate to a prompt target using a pre-built semantic map.
# Usage:
#     bash ~/drone_project/vla_universal/run_navigate.sh \
#         --scene warehouse_full --prompt "fly to the forklift"
#     # Disambiguate when 2+ matches:
#     bash ~/drone_project/vla_universal/run_navigate.sh \
#         --scene warehouse_full --prompt "fly to the forklift" --pick index --index 1

set -euo pipefail

source /home/ubuntu/miniconda3/bin/activate isaac
cd /home/ubuntu/IsaacLab

export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

exec stdbuf -oL ./isaaclab.sh -p /home/ubuntu/drone_project/vla_universal/navigate.py "$@"

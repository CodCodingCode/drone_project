#!/bin/bash
# Scan any scene into a semantic_map.json.
# Usage:
#     bash ~/drone_project/vla_universal/run_scan.sh --scene warehouse_full
#     bash ~/drone_project/vla_universal/run_scan.sh --scene hospital --quick
#     bash ~/drone_project/vla_universal/run_scan.sh --scene office \
#           --extra_classes "coffee mug,phone,lamp"

set -euo pipefail

source /home/ubuntu/miniconda3/bin/activate isaac
cd /home/ubuntu/IsaacLab

export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

exec stdbuf -oL ./isaaclab.sh -p /home/ubuntu/drone_project/vla_universal/scan.py "$@"

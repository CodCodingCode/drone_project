#!/bin/bash
# List USDs available on your Nucleus — one-shot diagnostic.
set -euo pipefail

source /home/ubuntu/miniconda3/bin/activate isaac
cd /home/ubuntu/IsaacLab

export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
export PYTHONUNBUFFERED=1

exec ./isaaclab.sh -p /home/ubuntu/drone_project/vla_warehouse/list_nucleus.py

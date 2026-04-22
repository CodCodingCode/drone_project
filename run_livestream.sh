#!/usr/bin/env bash
# Launch any drone_project train.py / play.py with WebRTC livestream (private/LAN).
# Usage:
#   bash ~/drone_project/run_livestream.sh hover/train.py --num_envs 64
#   bash ~/drone_project/run_livestream.sh lang_nav/play.py --checkpoint path/to.pt --num_envs 1
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate isaac

export LD_PRELOAD="${LD_PRELOAD:-}:/lib/aarch64-linux-gnu/libgomp.so.1"
export LIVESTREAM=2
export ENABLE_CAMERAS=${ENABLE_CAMERAS:-1}

SCRIPT="$1"; shift
cd "$HOME/IsaacLab"
exec ./isaaclab.sh -p "$HOME/drone_project/$SCRIPT" --livestream 2 "$@"

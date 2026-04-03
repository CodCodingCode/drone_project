#!/bin/bash
# =============================================================================
# Drone Project — Full Setup Script
# Sets up Isaac Lab + conda env + drone_project on a fresh GPU machine
# Tested on: GH200 (ARM64 H100), A10 (x86)
#
# Usage:
#   git clone https://github.com/CodCodingCode/drone_project.git ~/drone_project
#   cd ~/drone_project
#   bash setup.sh
# =============================================================================

set -e

echo "============================================"
echo "  Drone Project Setup"
echo "============================================"

# -------------------------------------------------------------------
# 1. Install Miniconda (if not present)
# -------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "[1/6] Installing Miniconda..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
    else
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi
    wget -q "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "[1/6] Miniconda installed."
else
    echo "[1/6] Miniconda already installed, skipping."
    eval "$(conda shell.bash hook)"
fi

# -------------------------------------------------------------------
# 2. Clone Isaac Lab (if not present)
# -------------------------------------------------------------------
ISAACLAB_PATH="$HOME/IsaacLab"
if [ ! -d "$ISAACLAB_PATH" ]; then
    echo "[2/6] Cloning Isaac Lab v2.3.2..."
    git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
    echo "[2/6] Isaac Lab cloned."
else
    echo "[2/6] Isaac Lab already exists at $ISAACLAB_PATH, skipping."
fi

# -------------------------------------------------------------------
# 3. Create conda environment
# -------------------------------------------------------------------
if ! conda env list | grep -q "^isaac "; then
    echo "[3/6] Creating conda environment 'isaac' (Python 3.11)..."
    conda create -n isaac python=3.11 -y
    echo "[3/6] Conda env created."
else
    echo "[3/6] Conda env 'isaac' already exists, skipping."
fi

conda activate isaac

# -------------------------------------------------------------------
# 4. Install Isaac Lab + Isaac Sim + dependencies
# -------------------------------------------------------------------
echo "[4/6] Installing Isaac Lab and Isaac Sim..."

# Install Isaac Sim (the simulation backend)
pip install isaacsim==5.1.0.0 isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit \
    --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab (editable install of all modules)
cd "$ISAACLAB_PATH"
./isaaclab.sh -i

echo "[4/6] Isaac Lab installed."

# -------------------------------------------------------------------
# 5. Install project-specific dependencies
# -------------------------------------------------------------------
echo "[5/6] Installing project dependencies..."

pip install \
    transformers>=4.50.0 \
    wandb>=0.25.0 \
    rsl-rl-lib>=5.0.0 \
    opencv-python-headless>=4.10.0 \
    tensorboard>=2.18.0 \
    moviepy>=2.0.0

echo "[5/6] Dependencies installed."

# -------------------------------------------------------------------
# 6. Verify installation
# -------------------------------------------------------------------
echo "[6/6] Verifying installation..."

cd "$ISAACLAB_PATH"

# Quick import test
./isaaclab.sh -p -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import isaaclab
print(f'  Isaac Lab: OK')

from transformers import CLIPModel
print(f'  CLIP: OK')

from rsl_rl.runners import OnPolicyRunner
print(f'  RSL-RL: OK')
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "To train hover (Stage 1):"
echo "  conda activate isaac"
echo "  cd ~/IsaacLab"
echo "  ./isaaclab.sh -p ~/drone_project/hover/train.py --num_envs 1024 --max_iterations 1500 --headless"
echo ""
echo "To train waypoint nav (Stage 2):"
echo "  ./isaaclab.sh -p ~/drone_project/waypoint_nav/train.py --num_envs 1024 --max_iterations 1500 --headless \\"
echo "      --resume_path ~/drone_project/logs/rsl_rl/hover_pretrain/<timestamp>/model_1499.pt"
echo ""
echo "To train lang nav (Stage 3):"
echo "  python ~/drone_project/transfer_waypoint_to_vla.py --waypoint_checkpoint <waypoint.pt> --output_path ~/drone_project/logs/rsl_rl/lang_drone_direct/vla_init.pt"
echo "  ./isaaclab.sh -p ~/drone_project/lang_nav/train.py --num_envs 256 --max_iterations 3000 --headless --enable_cameras \\"
echo "      --resume_path ~/drone_project/logs/rsl_rl/lang_drone_direct/vla_init.pt"
echo ""

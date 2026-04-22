#!/usr/bin/env bash
# Install Cesium for Omniverse into the Isaac Sim extension search path.
#
# Cesium for Omniverse is a Kit extension distributed via GitHub releases
# (https://github.com/CesiumGS/cesium-omniverse/releases) as a .zip that we
# unpack into `~/.local/share/ov/pkg/isaac-sim-*/kit/exts/` (Isaac Sim picks
# it up automatically on next launch).
#
# Alternative: install via Omniverse Launcher (if you use it) — in that case
# the extension is auto-registered and this script is unnecessary.
#
# Usage:
#     bash vla_cesium/install_cesium.sh [version]
#     # e.g. bash vla_cesium/install_cesium.sh 0.25.0

set -euo pipefail

VERSION="${1:-0.26.0}"
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)  PLATFORM="linux-x86_64"  ;;
    aarch64) PLATFORM="linux-aarch64" ;;
    *) echo "Unsupported arch: $ARCH (expected x86_64 or aarch64)"; exit 1 ;;
esac

REL_URL="https://github.com/CesiumGS/cesium-omniverse/releases/download/v${VERSION}/cesium-omniverse-${PLATFORM}-v${VERSION}.zip"

# Find Isaac Sim install
ISAAC_DIR=""
for candidate in \
    "$HOME/.local/share/ov/pkg"/isaac-sim-* \
    "$HOME/isaacsim" \
    "/isaac-sim"; do
    if [[ -d "$candidate" ]]; then
        ISAAC_DIR="$candidate"
        break
    fi
done
if [[ -z "$ISAAC_DIR" ]]; then
    echo "ERROR: could not find Isaac Sim install. Set ISAAC_DIR env var."
    echo "Looked in ~/.local/share/ov/pkg/isaac-sim-*, ~/isaacsim, /isaac-sim"
    exit 1
fi
echo "Found Isaac Sim at: $ISAAC_DIR"

EXT_DIR="$ISAAC_DIR/kit/exts"
if [[ ! -d "$EXT_DIR" ]]; then
    # Isaac Sim 5.x layout: extensions under kit/extsPhysics or exts
    EXT_DIR="$ISAAC_DIR/exts"
fi
mkdir -p "$EXT_DIR"

TMPDIR="$(mktemp -d)"
trap "rm -rf $TMPDIR" EXIT

echo "Downloading Cesium for Omniverse v${VERSION} ($PLATFORM)..."
curl -L -o "$TMPDIR/cesium.zip" "$REL_URL"

echo "Unpacking into $EXT_DIR ..."
unzip -q -o "$TMPDIR/cesium.zip" -d "$EXT_DIR"

echo ""
echo "Install complete. Next steps:"
echo "  1) export CESIUM_ION_TOKEN=<your-token>  # get one at cesium.com/ion/tokens"
echo "  2) ./isaaclab.sh -p ~/drone_project/vla_cesium/train.py --num_envs 4 --max_iterations 1 --headless --enable_cameras"
echo "     (quick smoke test — will take ~5 min for first-time 3D Tile cache fill)"

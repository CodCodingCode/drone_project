# VLA Drone — Cesium Real-World Navigation

Stage 4b of the curriculum: fine-tune the VLA policy in a real city streamed
from Cesium 3D Tiles (Google Photorealistic / OSM Buildings / Cesium World
Terrain). Same drone, same cameras, same PPO loop — only the scene and
target geometry change.

## What it does

The Cesium env streams a real city (Manhattan / SF / London by default)
into Isaac Sim as a USD tileset. At each episode:
- 3 real-world POIs are picked from the city bank (`pois.py`)
- 3 visible marker shapes (cube/sphere/cylinder) are repositioned to those
  POIs' lat/lon/alt locations
- One is chosen as the target; its natural-language prompt (e.g., "fly to
  the empire state building") is tokenized by PaliGemma
- The drone spawns ~80m above street level and must fly to the target
- The existing reward machinery (distance shaping, proximity, hover,
  wrong-object penalty) works unchanged — only scales are retuned for a
  city-scale arena (success within 8m, altitude envelope 20–300m, tanh
  shaping characteristic length 40m)

Your existing `vla/` checkpoint transfers directly: `train.py` uses the
same shape-mismatch-tolerant loader, so PaliGemma LoRA weights,
cross-attention head, depth encoder, and the frozen waypoint policy all
warm-start.

## Prerequisites

### 1. Cesium Ion token (free)
Sign up at https://cesium.com/ion/tokens and copy your default token.
```bash
export CESIUM_ION_TOKEN=eyJhbGciOiJIUzI1...
```
Free tier allows ~5GB/month of 3D Tile streaming — plenty for training
(tiles cache locally after first load).

### 2. Cesium for Omniverse extension
Install once:
```bash
bash ~/drone_project/vla_cesium/install_cesium.sh
```
Fetches the latest release from the CesiumGS/cesium-omniverse GitHub repo
and drops it into Isaac Sim's extension directory. The `train.py` shim
enables it automatically at startup via `enable_extension("cesium.omniverse")`.

## Running

All commands run from `~/IsaacLab`.

### Smoke test (1 iter, 4 envs, ~5 min with 3D Tile cache fill)
```bash
./isaaclab.sh -p ~/drone_project/vla_cesium/train.py \
    --num_envs 4 --max_iterations 1 --headless --enable_cameras
```
If this completes without Cesium errors, you're good.

### Fine-tune from the existing VLA checkpoint
```bash
./isaaclab.sh -p ~/drone_project/vla_cesium/train.py \
    --num_envs 128 --max_iterations 3000 \
    --headless --enable_cameras \
    --resume_path ~/drone_project/logs/rsl_rl/vla_drone_direct/<YYYY-MM-DD_HH-MM-SS>/model_4999.pt
```
Lower `--num_envs` from the empty-env default (256) — 3D Tile streaming
and real-world geometry are more memory-intensive. On an A100 80GB, 128
envs works; on smaller cards try 64.

### Playback with video
```bash
./isaaclab.sh -p ~/drone_project/vla_cesium/play.py \
    --checkpoint ~/drone_project/logs/rsl_rl/vla_drone_cesium/<run>/model_XXXX.pt \
    --num_steps 1500 --video
```

## Switching cities

Edit `VLACesiumDroneEnvCfg.city` in `vla_cesium_env.py` or pass a runtime
override. Built-in options: `"manhattan"` (default), `"san_francisco"`,
`"london"`. Add more by extending `CITY_BANKS` in `pois.py` — each entry
needs a `(lat, lon, alt)` origin tuple and a list of `POI` instances with
WGS84 coordinates and prompt templates.

## Tuning knobs

All in `vla_cesium_env.py` — `VLACesiumDroneEnvCfg`:

| Field | Default | What it controls |
|---|---|---|
| `distance_tanh_scale` | 40.0 | Characteristic length for distance-reward shaping. Drop to ~10 for tight targeting, raise for larger arenas. |
| `success_threshold` | 8.0 m | How close to the target POI counts as success. |
| `spawn_altitude` | 80.0 m | Drone's initial AGL altitude. Raise for skyline flights, lower for street-level. |
| `spawn_xy_radius` | 30.0 m | Randomization radius around georef origin at reset. |
| `altitude_warning_low/high` | 20 / 300 m | Soft airspace envelope — going outside incurs `crash_penalty_scale` per step. |
| `terminate_altitude_low/high` | 5 / 400 m | Hard termination bounds. |
| `num_active_pois` | 3 | Must match the number of marker shapes (cube+sphere+cylinder). |

## Files

| File | Purpose |
|---|---|
| `vla_cesium_env.py` | Env subclass — scene, reset, reward overrides |
| `pois.py` | POI database: Manhattan, SF, London |
| `cesium_setup.py` | Extension enable + USD tileset builder + ENU transforms |
| `__init__.py` | Registers `Isaac-VLADrone-Cesium-v0` |
| `agents/rsl_rl_ppo_cfg.py` | PPO config (same as vla/ except experiment_name) |
| `train.py` / `play.py` | Thin shims — patch `sys.modules` so `vla/train.py` and `vla/play.py` run unmodified against the Cesium env |
| `install_cesium.sh` | Downloads + installs Cesium for Omniverse from GitHub releases |

## Known limitations

* **No collision with buildings.** 3D Tiles don't carry collision meshes by
  default. The drone flies through buildings. We rely on altitude bounds
  for "crash" detection — good enough for first-pass training. To add
  collision, enable `UsdPhysics.CollisionAPI` on the tileset mesh prims
  post-load (expensive for streaming tiles, but doable for a fixed bounding
  region).
* **Flat-earth ENU approximation.** `cesium_setup.latlon_to_enu` is
  accurate within ~1m over 5 km. For continental-scale flight, swap in
  `pyproj` ECEF conversion.
* **Shared world, different env origins.** All 256 envs render the same
  Cesium tileset. Each env's drone is offset by `env_spacing` in local XY,
  so drones see the same city from slightly different viewpoints. Target
  POIs resolve to the same world positions across envs.
* **POIs are hand-curated.** For automatic target extraction from 3D Tiles
  (e.g., "fly to any car"), you'd need a separate object detection pass
  on the tileset or integration with Google Places API.

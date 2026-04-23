# VLA Drone — NVIDIA Sample Environments

Fine-tune the VLA policy in Isaac Sim's built-in warehouse / hospital /
office USD scenes instead of the empty 3-object arena. No extensions, no
tokens, no external downloads — the scenes stream from the same NVIDIA
Nucleus server that `ISAAC_NUCLEUS_DIR` already resolves to.

## Why this instead of Cesium?

Cesium for Omniverse ships only x86_64 prebuilt binaries and targets Isaac
Sim 4.5 or 6.0. This machine is aarch64 on Isaac Sim 5.1, so Cesium doesn't
work out of the box. Building it from source for aarch64 is a 2-4 hour
rabbit hole with uncertain payoff. Sample scenes deliver the key value
(rich visual context + named targets like forklifts, pallets, beds) with
zero setup.

If you later move to an x86_64 box, the existing `vla_cesium/` code is
still there — just install the extension and use those shims.

## What it does

At each episode, the env:
1. Clones a warehouse/hospital/office USD per env
2. Picks 3 real-object POIs from that scene's bank (`pois.py`)
3. Places the 3 marker shapes (cube/sphere/cylinder) at those POI positions
4. Chooses one as the target; its natural-language prompt (e.g., "fly to
   the forklift") drives PaliGemma
5. Spawns the drone at ~2.5m altitude with random XY jitter
6. Reward is the same parent machinery, rescaled for a ~20m indoor arena
   (success within 1m, altitude envelope 0.3–5.5m, tanh shaping length 5m)

## Prerequisites

None. Just your existing `isaac` conda env. You should be able to train
the empty-env vla/ successfully first — if that works, this does too.

## Running

All from `~/IsaacLab`.

### Smoke test (4 envs, 1 iter)
```bash
./isaaclab.sh -p ~/drone_project/vla_warehouse/train.py \
    --num_envs 4 --max_iterations 1 --headless --enable_cameras
```
First run streams + caches the warehouse USD (~30s), then completes one
iteration. If this runs without errors, you're good.

### Fine-tune from the existing VLA checkpoint
```bash
# Find your VLA checkpoint
ls -t ~/drone_project/logs/rsl_rl/vla_drone_direct/*/model_*.pt | head -1

# Train
./isaaclab.sh -p ~/drone_project/vla_warehouse/train.py \
    --num_envs 64 --max_iterations 3000 \
    --headless --enable_cameras \
    --resume_path <path-from-above>
```
Note: `--num_envs 64` not 256 — a cloned warehouse per env uses much more
memory than the empty arena. On A100 80GB, 64 is comfortable. Drop to 32
if you see OOMs, or switch to a smaller scene (see below).

### Playback with video
```bash
./isaaclab.sh -p ~/drone_project/vla_warehouse/play.py \
    --checkpoint ~/drone_project/logs/rsl_rl/vla_drone_warehouse/<run>/model_XXXX.pt \
    --num_steps 1500 --video
```

## Switching scenes

Edit `VLAWarehouseDroneEnvCfg.scene_name` in
[vla_warehouse_env.py](vla_warehouse_env.py) — or override at import time.
Built-in options:

| key | USD | POI count | env_spacing |
|---|---|---|---|
| `warehouse` (default) | `Simple_Warehouse/warehouse.usd` | 8 | 40m |
| `warehouse_full` | `Simple_Warehouse/full_warehouse.usd` | 8 | 40m |
| `warehouse_shelves` | `Simple_Warehouse/warehouse_multiple_shelves.usd` | 8 | 40m |
| `hospital` | `Hospital/hospital.usd` | 7 | 30m |
| `office` | `Office/office.usd` | 6 | 25m |

If Nucleus doesn't have one of these USDs (they're part of NVIDIA's sample
scene set), you'll see a "file not found" error from the USD loader —
check via `omni.client.list(f"{ISAAC_NUCLEUS_DIR}/Environments")`.

## Tuning knobs

All in [vla_warehouse_env.py](vla_warehouse_env.py) — `VLAWarehouseDroneEnvCfg`:

| Field | Default | What it controls |
|---|---|---|
| `distance_tanh_scale` | 5.0 m | Characteristic length for distance shaping. Drop for tighter targeting in small rooms. |
| `success_threshold` | 1.0 m | Win radius. Drop to 0.5 for pickier targeting. |
| `spawn_altitude` | 2.5 m | Drone initial height above env origin. |
| `spawn_xy_radius` | 3.0 m | Horizontal spawn jitter. |
| `altitude_warning_low/high` | 0.3 / 5.5 m | Soft airspace bounds (reward penalty outside). |
| `terminate_altitude_low/high` | 0.15 / 6.5 m | Hard termination bounds. |
| `num_active_pois` | 3 | Must equal the number of marker shapes (= 3). |
| `scene_name` | `warehouse` | Which USD scene to load. |

POI positions are hand-approximated in [pois.py](pois.py). If markers end
up inside walls or floating in the middle of shelves, edit the `x, y, z`
values for the affected POI — no code change elsewhere needed.

## Files

| File | Purpose |
|---|---|
| `vla_warehouse_env.py` | Env subclass — scene, reset, reward overrides |
| `pois.py` | POI banks for warehouse / hospital / office |
| `scene_setup.py` | Thin helper that spawns the USD via `UsdFileCfg` |
| `__init__.py` | Registers `Isaac-VLADrone-Warehouse-v0` |
| `agents/rsl_rl_ppo_cfg.py` | PPO config (same as vla/ except experiment_name) |
| `train.py` / `play.py` | Shims — patch `sys.modules` so `vla/train.py` and `vla/play.py` run unmodified against this env |

## Known limitations

* **No collision on the warehouse geometry by default.** Depending on the
  USD, shelves and props may have collision authored, or they may not.
  The drone can fly through walls if they don't. You'll notice during
  playback. To force collisions, enable `UsdPhysics.CollisionAPI` on the
  scene root after load — add it to `scene_setup.py`.
* **POI positions are hand-approximated.** They were chosen from typical
  Simple_Warehouse layouts but may not match your Nucleus's exact USD.
  Inspect during playback and tune if markers are hovering awkwardly.
* **Per-env scene clone is memory-heavy.** If you hit OOMs, lower
  `--num_envs`, or swap to `office.usd` (the smallest scene).

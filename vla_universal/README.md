# VLA Universal — Scan Then Navigate (Any USD Scene)

One drone, any Isaac Sim USD scene, natural-language commands. No per-scene
hand-curation; the drone scans the space once and builds its own semantic map.

## Two-step workflow

### 1. Scan — build a semantic map for a scene
```bash
bash ~/drone_project/vla_universal/run_scan.sh --scene warehouse_full
# takes ~3–5 min, writes logs/maps/warehouse_full.json
```

### 2. Navigate — fly to anything on the map
```bash
bash ~/drone_project/vla_universal/run_navigate.sh \
    --scene warehouse_full --prompt "fly to the forklift"
# takes ~20–60 s, records video to videos/vla_universal/navigate_<ts>.mp4
```

That's it. No training, no checkpoint management; reuses the frozen waypoint
policy from `model_2998_waypoint.pt` for low-level flight and
`google/paligemma-3b-pt-224` for object detection.

## How it works

**Scan** (`scan.py`):
1. Spawns a `UniversalDroneEnv` (single env, no POI markers, long episode).
2. Generates a scripted waypoint grid — 3 altitudes × ~6–8 xy points.
3. At each waypoint, teleports the drone, lets physics settle, captures
   RGB + depth + pose from all 4 onboard cameras.
4. Post-flight: runs PaliGemma in `"detect ..."` mode on each batch of 4
   images, parses `<loc####>` tokens to bounding boxes.
5. Projects each bbox center to 3D using `depth × camera intrinsics ×
   camera_world_pose`.
6. Clusters cross-view detections per class (greedy radius merge) and
   keeps clusters with ≥ 2 distinct views → drops hallucinations.
7. Saves a `semantic_map.json` with POI entries: class, xyz, confidence,
   n_observations, n_views, auto-generated prompts.

**Navigate** (`navigate.py`):
1. Loads the scene's semantic map.
2. Matches the user's prompt (case-insensitive substring + synonym table)
   against POI classes.
3. Picks a target via `--pick` policy: `closest` (default), `highest_conf`,
   `index N`, or `ask` (prints candidates and exits).
4. Feeds `(flight_state, target_body, pos_error_world)` to the frozen
   waypoint MLP each physics step.
5. Terminates when within 0.5 m of target with speed < 0.3 m/s for 50
   consecutive ticks, or after `--timeout_s` (default 60 s).
6. Writes an MP4 with front + right camera views, distance overlay.

## Supported scenes

Set via `--scene <name>`. The following are wired in (from
`vla_warehouse/pois.py:SCENES`):

| Scene key | USD | Notes |
|---|---|---|
| `warehouse_full` (default) | Simple_Warehouse/full_warehouse.usd | Largest; rich content |
| `warehouse` | Simple_Warehouse/warehouse.usd | Empty shell (no props) |
| `warehouse_shelves` | Simple_Warehouse/warehouse_multiple_shelves.usd | Shelves only |
| `hospital` | Hospital/hospital.usd | Beds, carts, doors |
| `office` | Office/office.usd | Desks, chairs, conf rooms |

To add a new scene, append an entry to
[vla_warehouse/pois.py](../vla_warehouse/pois.py)'s `SCENES` dict and
optionally add a default bbox in `scan.py`'s `default_bboxes`.

## Tuning knobs

| Flag | Default | What it does |
|---|---|---|
| `--quick` | off | halves altitude count + 1.8× xy spacing → ~2 min scan vs ~4 min |
| `--xy_spacing` | 6.0 m | horizontal waypoint grid spacing |
| `--settle_ticks` | 8 | physics ticks to run after teleport before capture |
| `--extra_classes` | "" | comma-separated extra detection classes |
| `--pick` | closest | disambiguation policy when ≥ 2 POIs match |
| `--target_id` | — | skip pick logic with explicit POI id |
| `--timeout_s` | 60.0 | navigation timeout |

Default detection classes (edit `DEFAULT_CLASSES` in
[detector.py](detector.py) to change):
`forklift, pallet, shelf, rack, cardboard box, loading dock, crate, barrel,
hospital bed, wheelchair, medical cart, monitor, doorway, desk, chair,
conference table, whiteboard, plant, computer, person, ladder, trash can,
cone, door`.

## Files

| File | Purpose |
|---|---|
| `scan.py` | Scan entry point |
| `navigate.py` | Navigate entry point |
| `universal_env.py` | `UniversalDroneEnv` — scene-agnostic drone env |
| `waypoint_controller.py` | Standalone frozen waypoint MLP (no PaliGemma dep) |
| `detector.py` | `PaliGemmaDetector` + `<locNNNN>` regex parser + default class list |
| `projection.py` | pixel → world math, 7×7 depth patch median |
| `semantic_map.py` | `POI` / `SemanticMap` dataclasses, JSON I/O, dedup, query |
| `flight_path.py` | `generate_scan_waypoints()` — boustrophedon grid |
| `run_scan.sh` / `run_navigate.sh` | conda + LD_PRELOAD launchers |

## Known limitations

- **PaliGemma accuracy on synthetic USD textures is unverified.** Works
  well on natural photos; may miss some objects in sim-rendered scenes.
  Fallback path (not yet implemented): switch to captioning + embedding
  similarity if detection recall is poor.
- **Depth noise** (~0.5–1 m) can displace projections; mitigated by 7×7
  median patch + 2-view requirement.
- **See-through walls**: Isaac Sim visual meshes don't always have
  collision; we filter projections outside `scene_bbox_world` + require
  multi-view confirmation, but it's not perfect.
- **Moved objects invalidate the map.** Re-run the scan if the scene changes.
- **Waypoint policy only goes to local goals.** It was trained on a 2.5 m
  arena; target positions are clamped to ±3 m in body frame each step
  (policy re-plans continuously), so long-distance navigation works but
  is slow.

## Verification steps

1. **Scan smoke**:
   ```
   bash run_scan.sh --scene warehouse_full --quick
   ```
   Completes in < 5 min, prints POI-by-class summary, writes
   `logs/maps/warehouse_full.json` with ≥ 3 POIs.

2. **Navigate**:
   ```
   bash run_navigate.sh --scene warehouse_full --prompt "fly to the forklift"
   ```
   Reaches target in < 60 s, saves video, final distance < 1 m.

3. **Cross-scene** (validates the "any scene" claim):
   ```
   bash run_scan.sh --scene hospital
   bash run_navigate.sh --scene hospital --prompt "fly to the bed"
   ```

## Future work

- Use scanned POIs as RL training targets → fine-tune the hierarchical
  VLA to generalize navigation over scanned maps.
- Online re-scanning for dynamic scenes.
- Replace string-match query with embedding similarity (e.g. SigLIP text)
  for free-form prompts like "the red one next to the pallet".

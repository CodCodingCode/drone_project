"""Points-of-interest for VLA drone navigation in Isaac Sim's sample scenes.

Positions are in local Isaac Lab coordinates (meters, Z-up) relative to the
env origin. The warehouse is ~20m × 30m, ceiling ~6m. POI positions below
are hand-curated from the standard Simple_Warehouse / Hospital / Office
USDs — tune them against your actual scene if targets end up buried in
shelving or floating in walls.

Each POI has the same structure as vla_cesium.pois.POI:
  - name: identifier for logs
  - x, y, z: local position (meters)
  - cls: class label
  - prompts: natural-language commands
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class POI:
    name: str
    x: float
    y: float
    z: float
    cls: str
    prompts: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Simple_Warehouse (warehouse.usd / full_warehouse.usd)
# Floor plan roughly -10..+10 in X, -15..+15 in Y, ceiling at Z≈6m
# --------------------------------------------------------------------------
WAREHOUSE_POIS: list[POI] = [
    POI("forklift_main",     4.0,  -6.0, 1.5, "forklift",
        ["fly to the forklift", "go to the yellow forklift",
         "navigate to the lift truck"]),
    POI("forklift_corner",  -6.0,   8.0, 1.5, "forklift",
        ["fly to the other forklift", "go to the forklift in the corner"]),
    POI("pallet_stack",      0.0,   4.0, 0.5, "pallet",
        ["fly to the pallet", "go to the wooden pallet",
         "navigate to the cargo pallet"]),
    POI("shelf_row_a",       6.0,   0.0, 2.5, "shelf",
        ["fly to the shelves", "go to the storage rack",
         "navigate to the racking"]),
    POI("shelf_row_b",      -6.0,   0.0, 2.5, "shelf",
        ["fly to the other shelves", "go to the far rack"]),
    POI("cardboard_boxes",   3.0,   9.0, 1.0, "box",
        ["fly to the boxes", "go to the cardboard boxes",
         "navigate to the stack of boxes"]),
    POI("loading_dock",      0.0, -12.0, 1.0, "dock",
        ["fly to the loading dock", "go to the dock door",
         "navigate to the exit"]),
    POI("workstation",      -4.0,  -4.0, 1.5, "workstation",
        ["fly to the workstation", "go to the desk"]),
]


# --------------------------------------------------------------------------
# Hospital (hospital.usd) — rooms, corridors, medical equipment
# --------------------------------------------------------------------------
HOSPITAL_POIS: list[POI] = [
    POI("reception",      0.0,  0.0, 1.5, "desk",
        ["fly to reception", "go to the front desk"]),
    POI("hospital_bed_a", 5.0,  3.0, 1.0, "bed",
        ["fly to the hospital bed", "go to the bed", "find the patient bed"]),
    POI("hospital_bed_b",-5.0,  3.0, 1.0, "bed",
        ["fly to the other bed", "go to the second bed"]),
    POI("wheelchair",     2.0, -5.0, 1.0, "wheelchair",
        ["fly to the wheelchair", "go to the chair on wheels"]),
    POI("medical_cart",  -3.0, -4.0, 1.0, "cart",
        ["fly to the medical cart", "go to the supply cart"]),
    POI("doorway",        0.0,  8.0, 1.5, "door",
        ["fly to the doorway", "go to the door",
         "navigate to the corridor exit"]),
    POI("monitor",        4.0,  5.0, 1.8, "monitor",
        ["fly to the monitor", "go to the screen"]),
]


# --------------------------------------------------------------------------
# Office (office.usd) — desks, chairs, conference rooms
# --------------------------------------------------------------------------
OFFICE_POIS: list[POI] = [
    POI("desk_main",      0.0,  0.0, 1.3, "desk",
        ["fly to the desk", "go to the workstation"]),
    POI("chair_a",        2.0,  0.0, 1.0, "chair",
        ["fly to the chair", "go to the office chair"]),
    POI("conference_table", 4.0,  4.0, 1.2, "table",
        ["fly to the conference table", "go to the meeting room"]),
    POI("whiteboard",    -3.0,  3.0, 1.8, "whiteboard",
        ["fly to the whiteboard", "go to the white board"]),
    POI("plant",          3.0, -3.0, 1.0, "plant",
        ["fly to the plant", "go to the potted plant"]),
    POI("doorway",        0.0,  7.0, 1.5, "door",
        ["fly to the door", "go to the exit"]),
]


# Scene registry — select via VLAWarehouseDroneEnvCfg.scene_name
#   usd_path: relative to ISAAC_NUCLEUS_DIR (gets prepended at scene build time)
#   env_spacing: per-env spacing so cloned scenes don't visually collide
#   floor_z: terrain plane height in scene coords (usually 0)
SCENES: dict[str, dict] = {
    "warehouse": {
        "usd_path": "/Environments/Simple_Warehouse/warehouse.usd",
        "pois": WAREHOUSE_POIS,
        "env_spacing": 40.0,     # warehouse is ~20x30m; clones need room
        "floor_z": 0.0,
        "ceiling_z": 6.0,
    },
    "warehouse_full": {
        "usd_path": "/Environments/Simple_Warehouse/full_warehouse.usd",
        "pois": WAREHOUSE_POIS,
        "env_spacing": 40.0,
        "floor_z": 0.0,
        "ceiling_z": 6.0,
    },
    "warehouse_shelves": {
        "usd_path": "/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
        "pois": WAREHOUSE_POIS,
        "env_spacing": 40.0,
        "floor_z": 0.0,
        "ceiling_z": 6.0,
    },
    "hospital": {
        "usd_path": "/Environments/Hospital/hospital.usd",
        "pois": HOSPITAL_POIS,
        "env_spacing": 30.0,
        "floor_z": 0.0,
        "ceiling_z": 4.0,
    },
    "office": {
        "usd_path": "/Environments/Office/office.usd",
        "pois": OFFICE_POIS,
        "env_spacing": 25.0,
        "floor_z": 0.0,
        "ceiling_z": 3.5,
    },
}


def get_scene(name: str) -> dict:
    if name not in SCENES:
        raise KeyError(f"Unknown scene '{name}'. Choices: {list(SCENES)}")
    return SCENES[name]

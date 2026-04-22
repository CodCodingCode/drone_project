"""Real-world Points-of-Interest for VLA drone navigation in Cesium 3D Tiles.

Each POI has:
  - name: human-readable identifier
  - lat, lon, alt: WGS84 (degrees + meters MSL) — anchor for reward computation
  - cls: one of {"building", "car", "tower", "bridge", "park", "stadium"}
  - prompts: natural-language commands referring to this POI

At reset, an episode picks `num_active_pois` POIs from the bank of the
current origin. One is the target (its prompt becomes the language command),
the rest are distractors — reaching them triggers the wrong-object penalty
(same machinery as the cube/sphere/cylinder design in vla/).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class POI:
    name: str
    lat: float
    lon: float
    alt: float          # meters MSL
    cls: str            # class label (building, car, ...)
    prompts: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Manhattan Midtown — dense skyline, iconic landmarks
# Origin ~ Times Square (40.758, -73.9855)
# --------------------------------------------------------------------------
MANHATTAN_POIS: list[POI] = [
    POI("empire_state", 40.7484, -73.9857, 220.0, "building",
        ["fly to the empire state building", "go to the tall tower",
         "navigate to the skyscraper", "find the empire state"]),
    POI("chrysler", 40.7516, -73.9755, 180.0, "building",
        ["fly to the chrysler building", "go to the silver-topped tower",
         "navigate to the art deco skyscraper"]),
    POI("rockefeller", 40.7587, -73.9787, 120.0, "building",
        ["fly to rockefeller center", "go to the plaza building"]),
    POI("grand_central", 40.7527, -73.9772, 40.0, "building",
        ["fly to grand central terminal", "go to the train station",
         "find the beaux-arts station"]),
    POI("bryant_park", 40.7536, -73.9832, 5.0, "park",
        ["fly to bryant park", "go to the green park", "find the city park"]),
    POI("times_square", 40.7580, -73.9855, 5.0, "park",
        ["fly to times square", "go to the bright billboards",
         "navigate to the neon intersection"]),
    POI("flatiron", 40.7411, -73.9897, 80.0, "building",
        ["fly to the flatiron building", "go to the triangular building"]),
    POI("madison_square_garden", 40.7505, -73.9934, 50.0, "stadium",
        ["fly to madison square garden", "go to the arena",
         "navigate to the sports venue"]),
]

# --------------------------------------------------------------------------
# San Francisco — waterfront + hills
# Origin ~ Embarcadero / Ferry Building
# --------------------------------------------------------------------------
SAN_FRANCISCO_POIS: list[POI] = [
    POI("transamerica", 37.7952, -122.4028, 260.0, "building",
        ["fly to the transamerica pyramid", "go to the pyramid building",
         "navigate to the pointed skyscraper"]),
    POI("ferry_building", 37.7956, -122.3933, 75.0, "building",
        ["fly to the ferry building", "go to the clock tower"]),
    POI("salesforce_tower", 37.7897, -122.3972, 326.0, "building",
        ["fly to salesforce tower", "go to the tallest tower"]),
    POI("coit_tower", 37.8024, -122.4058, 210.0, "tower",
        ["fly to coit tower", "go to the white tower on the hill"]),
    POI("embarcadero_plaza", 37.7945, -122.3950, 10.0, "park",
        ["fly to embarcadero plaza", "go to the waterfront square"]),
    POI("oakland_bay_bridge", 37.7983, -122.3778, 70.0, "bridge",
        ["fly to the bay bridge", "navigate to the suspension bridge"]),
]

# --------------------------------------------------------------------------
# London — Westminster / Thames
# Origin ~ Westminster Bridge
# --------------------------------------------------------------------------
LONDON_POIS: list[POI] = [
    POI("big_ben", 51.5007, -0.1246, 100.0, "tower",
        ["fly to big ben", "go to the clock tower",
         "navigate to the westminster clock"]),
    POI("london_eye", 51.5033, -0.1196, 120.0, "tower",
        ["fly to the london eye", "go to the ferris wheel",
         "navigate to the observation wheel"]),
    POI("tower_bridge", 51.5055, -0.0754, 60.0, "bridge",
        ["fly to tower bridge", "go to the gothic bridge"]),
    POI("shard", 51.5045, -0.0865, 310.0, "building",
        ["fly to the shard", "go to the glass pyramid tower",
         "navigate to the pointed skyscraper"]),
    POI("parliament", 51.4995, -0.1248, 60.0, "building",
        ["fly to the houses of parliament", "go to the palace of westminster"]),
    POI("westminster_abbey", 51.4993, -0.1273, 70.0, "building",
        ["fly to westminster abbey", "go to the gothic church"]),
]


# City registry — select via VLACesiumDroneEnvCfg.city
CITY_BANKS: dict[str, dict] = {
    "manhattan":   {"origin": (40.7580, -73.9855, 10.0), "pois": MANHATTAN_POIS},
    "san_francisco": {"origin": (37.7956, -122.3950, 10.0), "pois": SAN_FRANCISCO_POIS},
    "london":      {"origin": (51.5020, -0.1250, 10.0), "pois": LONDON_POIS},
}


def get_city(name: str) -> dict:
    if name not in CITY_BANKS:
        raise KeyError(f"Unknown city '{name}'. Choices: {list(CITY_BANKS)}")
    return CITY_BANKS[name]

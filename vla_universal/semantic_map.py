"""Semantic-map data model, dedup/clustering, and query logic.

A SemanticMap is a per-scene JSON artifact produced by scan.py. It lists
POIs (point-of-interest clusters) with their class, 3D position, and
natural-language prompts. navigate.py loads it and matches user commands
against it.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from vla_universal.detector import Detection


# --------------------------------------------------------------------------
# Class-level defaults for dedup
# --------------------------------------------------------------------------
DEFAULT_RADIUS = 1.5  # meters
RADIUS_BY_CLASS: dict[str, float] = {
    # big stuff needs a wider merge radius — multiple detections on a
    # single object can be several meters apart at its surface.
    "forklift": 2.5,
    "shelf": 2.5, "rack": 2.5,
    "conference table": 2.0, "desk": 1.5,
    # small stuff wants a tight radius so we don't merge neighbors
    "pallet": 1.0, "cardboard box": 1.0, "crate": 1.0, "barrel": 0.8,
    "monitor": 0.6, "computer": 0.6,
    "door": 0.8, "doorway": 1.0,
    "chair": 0.8, "plant": 0.8,
    "person": 0.8, "cone": 0.5, "trash can": 0.6,
    "wheelchair": 1.0, "medical cart": 1.0,
    "hospital bed": 2.0, "whiteboard": 1.5,
    "ladder": 1.0, "loading dock": 2.5,
}

MAX_CLUSTERS_PER_CLASS = 8


# --------------------------------------------------------------------------
# Synonym table — "lift truck" ↔ "forklift", etc.
# --------------------------------------------------------------------------
SYNONYMS: dict[str, list[str]] = {
    "forklift": ["lift truck", "lifter"],
    "shelf": ["rack", "racking", "shelving"],
    "rack": ["shelf", "shelving"],
    "cardboard box": ["box", "carton"],
    "pallet": ["skid"],
    "hospital bed": ["bed", "patient bed"],
    "wheelchair": ["chair on wheels"],
    "monitor": ["screen", "display"],
    "doorway": ["door", "entrance", "exit"],
    "door": ["doorway", "entrance", "exit"],
    "conference table": ["meeting table", "table"],
    "desk": ["workstation", "table"],
    "whiteboard": ["board"],
    "trash can": ["garbage can", "bin", "dustbin"],
}


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class POI:
    id: str
    cls: str
    xyz_world: list[float]
    confidence: float
    n_observations: int
    n_views: int
    prompts: list[str] = field(default_factory=list)


@dataclass
class SemanticMap:
    scene_name: str
    scene_usd_path: str
    scene_bbox_world: list[list[float]]   # [[min_x,min_y,min_z],[max_x,max_y,max_z]]
    pois: list[POI] = field(default_factory=list)
    detector: str = "google/paligemma-3b-pt-224"
    schema_version: int = 1
    scan_timestamp: str = field(
        default_factory=lambda: _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )
    scan_params: dict = field(default_factory=dict)

    # ----------------- JSON I/O -----------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[semantic_map] saved {len(self.pois)} POIs to {path}")

    @classmethod
    def load(cls, path: str) -> "SemanticMap":
        with open(path, "r") as f:
            raw = json.load(f)
        pois = [POI(**p) for p in raw.pop("pois", [])]
        return cls(pois=pois, **raw)

    # ----------------- Query -----------------
    def classes(self) -> list[str]:
        return sorted({p.cls for p in self.pois})

    def query(self, prompt: str) -> list[POI]:
        """Return POIs whose class (or synonyms) appears in the prompt
        (case-insensitive, substring match). Empty list if no match."""
        p = prompt.lower()
        hits: list[POI] = []
        for poi in self.pois:
            names = {poi.cls, *SYNONYMS.get(poi.cls, [])}
            if any(name.lower() in p for name in names):
                hits.append(poi)
        return hits


# --------------------------------------------------------------------------
# Dedup across views
# --------------------------------------------------------------------------
def _cluster_radius(cls: str) -> float:
    return RADIUS_BY_CLASS.get(cls.lower(), DEFAULT_RADIUS)


def cluster_detections(
    records: list[tuple[Detection, np.ndarray]],  # (Detection, xyz_world)
    scene_bbox: Optional[tuple[np.ndarray, np.ndarray]] = None,
    min_views: int = 2,
) -> list[POI]:
    """Greedy per-class clustering.
    - Reject points outside scene_bbox (guards against see-through-wall hits).
    - Within a class, assign each detection to the nearest existing cluster
      centroid if within RADIUS_BY_CLASS[cls], else start a new cluster.
    - Drop clusters with fewer than `min_views` distinct cam+stop observations.
    - Cap at MAX_CLUSTERS_PER_CLASS per class.
    """
    # Filter by scene bbox
    if scene_bbox is not None:
        bmin, bmax = scene_bbox
        def _in_bbox(xyz: np.ndarray) -> bool:
            return bool(np.all(xyz >= bmin - 1.0) and np.all(xyz <= bmax + 1.0))
        records = [(d, x) for (d, x) in records if _in_bbox(x)]

    # Group by class
    by_cls: dict[str, list[tuple[Detection, np.ndarray]]] = {}
    for det, xyz in records:
        by_cls.setdefault(det.cls, []).append((det, xyz))

    pois: list[POI] = []
    for cls, items in by_cls.items():
        r = _cluster_radius(cls)
        clusters: list[dict] = []  # each: {centroid (np.ndarray), members: [(det,xyz)]}

        # Greedy — order doesn't matter much since we add all then recompute centroid
        for det, xyz in items:
            best_i = -1; best_d = r
            for i, c in enumerate(clusters):
                d = float(np.linalg.norm(xyz - c["centroid"]))
                if d < best_d:
                    best_d = d; best_i = i
            if best_i == -1:
                clusters.append({"centroid": xyz.copy(),
                                  "members": [(det, xyz)]})
            else:
                clusters[best_i]["members"].append((det, xyz))
                mem_xyzs = np.stack([m[1] for m in clusters[best_i]["members"]])
                clusters[best_i]["centroid"] = mem_xyzs.mean(axis=0)

        # Drop low-view clusters
        surviving = []
        for c in clusters:
            n_obs = len(c["members"])
            unique_views = len({(m[0].cam_idx, m[0].frame_idx)
                                 for m in c["members"]})
            if unique_views < min_views:
                continue
            surviving.append((c, n_obs, unique_views))

        # Cap and sort by n_obs descending
        surviving.sort(key=lambda t: t[1], reverse=True)
        surviving = surviving[:MAX_CLUSTERS_PER_CLASS]

        for idx, (c, n_obs, n_views) in enumerate(surviving):
            conf = 1.0 - math.exp(-n_obs / 3.0)
            pois.append(POI(
                id=f"{cls.replace(' ', '_')}_{idx}",
                cls=cls,
                xyz_world=[float(x) for x in c["centroid"]],
                confidence=round(conf, 3),
                n_observations=int(n_obs),
                n_views=int(n_views),
                prompts=[
                    f"fly to the {cls}",
                    f"go to the {cls}",
                    f"navigate to the {cls}",
                ],
            ))

    return pois

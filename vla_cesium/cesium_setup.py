"""Cesium-for-Omniverse helpers: extension enable, world build, ENU transforms.

Cesium for Omniverse (https://github.com/CesiumGS/cesium-omniverse) streams
photogrammetric 3D Tiles (Google Photorealistic, OSM Buildings, custom Ion
assets) into the USD stage. We enable the Kit extension at sim startup, then
author a `CesiumGeoreference` + `CesiumTileset` prim pair anchored at a
chosen WGS84 origin. Everything else in the scene (drone, markers) stays in
Isaac Lab's local ENU frame with origin at the georeference.

USAGE (from Isaac Lab env's _setup_scene):
    enable_cesium_extension()            # once, at app startup
    load_cesium_world(lat, lon, alt, ...) # per-stage, after scene clone

NOTES:
- The user must set CESIUM_ION_TOKEN in their environment (free token at
  cesium.com/ion/tokens). Google Photorealistic 3D Tiles is Ion asset 2275207;
  OSM Buildings is 96188 (free tier friendly).
- 3D Tiles DO NOT have collision meshes by default. The drone flies through
  buildings unless you enable collisions in the USD prim. We currently rely
  on altitude bounds for "crash" detection — good enough for a first pass.
"""

from __future__ import annotations

import math
import os

import torch


# Cesium Ion asset IDs (public)
ASSET_GOOGLE_PHOTOREALISTIC = 2275207  # high quality, per-call billed
ASSET_CESIUM_WORLD_TERRAIN  = 1         # free tier, no buildings
ASSET_OSM_BUILDINGS         = 96188    # free OSM-derived building footprints
ASSET_BING_MAPS_IMAGERY     = 2          # imagery overlay


# ---------------------------------------------------------------------------
# Extension loader
# ---------------------------------------------------------------------------

def enable_cesium_extension(verbose: bool = True) -> bool:
    """Enable the `cesium.omniverse` Kit extension. Returns True on success.

    Must be called AFTER AppLauncher has started SimulationApp but BEFORE
    any scene code that references Cesium USD schemas.
    """
    try:
        from isaacsim.core.utils.extensions import enable_extension
    except ImportError:
        # Older Isaac Sim: try omni.isaac.core
        try:
            from omni.isaac.core.utils.extensions import enable_extension
        except ImportError:
            if verbose:
                print("[cesium] ERROR: could not import enable_extension from Isaac Sim")
            return False

    ok = enable_extension("cesium.omniverse")
    if verbose:
        print(f"[cesium] enable_extension('cesium.omniverse') -> {ok}")
    return bool(ok)


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

def get_ion_token(env_var: str = "CESIUM_ION_TOKEN") -> str:
    token = os.environ.get(env_var, "").strip()
    if not token:
        raise RuntimeError(
            f"No Cesium Ion token found in ${env_var}. Get a free token at "
            f"https://cesium.com/ion/tokens and export it:\n"
            f"    export {env_var}=eyJhbGciOiJIUzI1...   # your token"
        )
    return token


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------

def load_cesium_world(
    lat: float,
    lon: float,
    alt: float,
    ion_asset_id: int = ASSET_GOOGLE_PHOTOREALISTIC,
    ion_token_env: str = "CESIUM_ION_TOKEN",
    georef_prim_path: str = "/World/CesiumGeoreference",
    tileset_prim_path: str = "/World/CesiumGeoreference/Tileset",
) -> None:
    """Author a Cesium Georeference + Tileset under /World on the current stage.

    After this call, the 3D tiles stream in around the origin. Local ENU
    coordinates (Isaac Lab's world frame) are centered at (lat, lon, alt),
    so a drone at (0, 0, 50) in Isaac is 50m above that lat/lon point.
    """
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available. Is SimulationApp initialized?")

    # Import Cesium USD schemas. These come from the cesium.omniverse extension.
    try:
        from pxr import Sdf
        from cesium.usd.plugins.CesiumUsdSchemas import (
            Georeference as CesiumGeoreference,
            Tileset as CesiumTileset,
            IonServer as CesiumIonServer,
        )
    except ImportError as e:
        raise RuntimeError(
            "Cesium USD schemas unavailable. Did enable_cesium_extension() succeed? "
            "Extension must be installed — see vla_cesium/install_cesium.sh"
        ) from e

    token = get_ion_token(ion_token_env)

    # 1) IonServer — credential holder for Ion asset requests
    ion_server_path = "/World/CesiumIonServer"
    ion_server = CesiumIonServer.Define(stage, Sdf.Path(ion_server_path))
    ion_server.GetDisplayNameAttr().Set("Cesium Ion")
    ion_server.GetIonServerUrlAttr().Set("https://ion.cesium.com/")
    ion_server.GetIonServerApiUrlAttr().Set("https://api.cesium.com/")
    ion_server.GetIonServerApplicationIdAttr().Set(413)
    ion_server.GetProjectDefaultIonAccessTokenAttr().Set(token)
    ion_server.GetProjectDefaultIonAccessTokenIdAttr().Set("")

    # 2) Georeference — anchors the local ENU frame at (lat, lon, alt)
    georef = CesiumGeoreference.Define(stage, Sdf.Path(georef_prim_path))
    georef.GetGeoreferenceOriginLatitudeAttr().Set(float(lat))
    georef.GetGeoreferenceOriginLongitudeAttr().Set(float(lon))
    georef.GetGeoreferenceOriginHeightAttr().Set(float(alt))

    # 3) Tileset — the actual streaming 3D Tiles
    tileset = CesiumTileset.Define(stage, Sdf.Path(tileset_prim_path))
    tileset.GetIonAssetIdAttr().Set(int(ion_asset_id))
    tileset.GetIonAccessTokenAttr().Set(token)
    tileset.GetSuspendUpdateAttr().Set(False)
    # Bind to our IonServer so the extension knows which credentials to use
    tileset.GetIonServerBindingRel().SetTargets([Sdf.Path(ion_server_path)])

    print(f"[cesium] Tileset {ion_asset_id} authored at {georef_prim_path} "
          f"(lat={lat:.5f}, lon={lon:.5f}, alt={alt:.1f}m)")


# ---------------------------------------------------------------------------
# ENU ↔ WGS84 conversion (flat-earth approximation, good within ~5km of origin)
# ---------------------------------------------------------------------------

_EARTH_R = 6_378_137.0  # WGS84 equatorial radius, meters


def latlon_to_enu(
    lat: float, lon: float, alt: float,
    lat0: float, lon0: float, alt0: float,
) -> tuple[float, float, float]:
    """Convert a WGS84 point to local East-North-Up meters relative to origin.

    Flat-earth approximation. Accurate to ~1m within ~5km of origin, which is
    plenty for city-scale navigation. For longer flights, use pyproj/ECEF.
    """
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    lat0r = math.radians(lat0)
    east  = dlon * _EARTH_R * math.cos(lat0r)
    north = dlat * _EARTH_R
    up    = alt - alt0
    return east, north, up


def pois_to_enu_tensor(
    pois,                           # list[POI]
    origin_lat: float, origin_lon: float, origin_alt: float,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Return (N, 3) tensor of POI positions in local ENU frame.

    Isaac Lab convention: Z is up, X is forward-ish. We map ENU → (X=east,
    Y=north, Z=up) — matches the default USD stage up-axis.
    """
    xyz = []
    for p in pois:
        e, n, u = latlon_to_enu(p.lat, p.lon, p.alt, origin_lat, origin_lon, origin_alt)
        xyz.append((e, n, u))
    return torch.tensor(xyz, dtype=torch.float32, device=device)

"""USD scene loader for Isaac Sim sample environments.

Authors a `sim_utils.UsdFileCfg` reference under `/World/envs/env_.*/Scene`
so the warehouse/hospital/office is cloned once per env alongside the drone.
No extension install, no Ion token — streams from the same NVIDIA Nucleus
server `ISAAC_NUCLEUS_DIR` already resolves to.

Handles USD payloads (deferred references) explicitly — many NVIDIA sample
scenes author their geometry as unloaded payloads, so a vanilla reference
authors an empty prim. We force-load payloads after authoring.
"""

from __future__ import annotations

import logging

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

_LOG = logging.getLogger("vla_warehouse.scene_setup")


def _force_load_and_inspect(prim_path_sample: str) -> None:
    """After authoring the USD reference, force-load payloads and log a
    bounding-box + prim-count snapshot so we can tell whether geometry
    actually resolved.
    """
    try:
        import omni.usd
        from pxr import Usd, UsdGeom, Gf
    except ImportError as e:
        _LOG.warning(f"skipping inspection — USD imports failed: {e}")
        return

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path_sample)
    if not prim.IsValid():
        msg = f"⚠ scene prim {prim_path_sample} is INVALID after authoring"
        print(f"[vla_warehouse] {msg}")
        _LOG.warning(msg)
        return

    # Force-load payloads on this prim and all descendants (the common case
    # for NVIDIA sample scenes is payloads that are unloaded by default).
    stage.Load(prim.GetPath())
    # Kit can take a tick to populate payloads; poke Fabric to flush.
    try:
        import omni.kit.app
        omni.kit.app.get_app().update()
    except Exception:
        pass

    # Re-fetch after load
    prim = stage.GetPrimAtPath(prim_path_sample)

    mesh_count = 0
    xform_count = 0
    descendants = 0
    for p in Usd.PrimRange(prim):
        descendants += 1
        if p.IsA(UsdGeom.Mesh):
            mesh_count += 1
        if p.IsA(UsdGeom.Xform):
            xform_count += 1

    # World-space bbox (useful to know where content actually sits)
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                        includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        if bbox.GetRange().IsEmpty():
            bbox_str = "EMPTY"
        else:
            mn = bbox.GetRange().GetMin()
            mx = bbox.GetRange().GetMax()
            bbox_str = f"[{mn[0]:.1f},{mn[1]:.1f},{mn[2]:.1f}] .. [{mx[0]:.1f},{mx[1]:.1f},{mx[2]:.1f}]"
    except Exception as e:
        bbox_str = f"<bbox error: {e}>"

    report = (f"scene {prim_path_sample}: "
              f"{descendants} prims ({mesh_count} meshes, {xform_count} xforms), "
              f"worldBBox {bbox_str}")
    print(f"[vla_warehouse] {report}")
    _LOG.warning(report)

    if mesh_count == 0:
        print("[vla_warehouse] ⚠ NO MESHES FOUND — scene will render empty. "
              "Either the USD is empty or payloads failed to resolve.")


def load_scene(
    usd_path_relative: str,
    prim_regex: str = "/World/envs/env_.*/Scene",
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Spawn a shared USD scene (warehouse/hospital/office) in each env.

    Must be called during `_setup_scene` BEFORE `scene.clone_environments`,
    matching the pattern used for cube/sphere/cylinder in vla_drone_env.py.
    Raises FileNotFoundError if the USD path doesn't resolve on Nucleus.

    Args:
      usd_path_relative: path from Nucleus root, e.g.
                         "/Environments/Simple_Warehouse/full_warehouse.usd".
      prim_regex:        per-env prim path (Isaac Lab regex-clones across envs).
      translation:       scene translation within each env frame.
    """
    usd_path = f"{ISAAC_NUCLEUS_DIR}{usd_path_relative}"
    print(f"[vla_warehouse] Attempting to load scene USD: {usd_path}")
    print(f"[vla_warehouse] (ISAAC_NUCLEUS_DIR = {ISAAC_NUCLEUS_DIR})")
    _LOG.warning(f"loading scene USD: {usd_path}")

    scene_cfg = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=(1.0, 1.0, 1.0),
    )
    # spawn_from_usd raises FileNotFoundError if the USD isn't reachable.
    scene_cfg.func(prim_regex, scene_cfg, translation=translation)

    # Force-load payloads and report what actually landed in the stage.
    sample_path = prim_regex.replace(".*", "0")
    _force_load_and_inspect(sample_path)

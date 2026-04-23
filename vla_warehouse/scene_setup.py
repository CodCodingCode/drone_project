"""USD scene loader for Isaac Sim sample environments.

Authors a `sim_utils.UsdFileCfg` reference under `/World/envs/env_.*/Scene`
so the warehouse/hospital/office is cloned once per env alongside the drone.
No extension install, no Ion token — streams from the same NVIDIA Nucleus
server `ISAAC_NUCLEUS_DIR` already resolves to.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def load_scene(
    usd_path_relative: str,
    prim_regex: str = "/World/envs/env_.*/Scene",
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Spawn a shared USD scene (warehouse/hospital/office) in each env.

    Must be called during `_setup_scene` BEFORE `scene.clone_environments`,
    matching the pattern used for cube/sphere/cylinder in vla_drone_env.py.

    Args:
      usd_path_relative: path from Nucleus root, e.g.
                         "/Environments/Simple_Warehouse/warehouse.usd".
      prim_regex:        per-env prim path (Isaac Lab regex-clones across envs).
      translation:       scene translation within each env frame.
    """
    usd_path = f"{ISAAC_NUCLEUS_DIR}{usd_path_relative}"
    scene_cfg = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=(1.0, 1.0, 1.0),
    )
    scene_cfg.func(prim_regex, scene_cfg, translation=translation)
    print(f"[vla_warehouse] Authored scene {usd_path} at {prim_regex}")

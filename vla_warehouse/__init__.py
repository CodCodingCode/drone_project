"""VLA drone — Isaac Sim sample environments (warehouse/hospital/office)."""

import gymnasium as gym

from . import agents  # noqa: F401

gym.register(
    id="Isaac-VLADrone-Warehouse-v0",
    entry_point=f"{__name__}.vla_warehouse_env:VLAWarehouseDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vla_warehouse_env:VLAWarehouseDroneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VLAWarehousePPORunnerCfg",
    },
)

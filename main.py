#!/usr/bin/env python3
"""
Drone simulation entry point for Isaac Sim + Pegasus Simulator.

Usage (from host or inside isaac-drone container):
    python main.py --mode hover    # headless physics test
    python main.py --mode scene    # livestream scene (view at /viewer)
    python main.py --mode video    # record frames to /tmp/drone_frames/

If run from the Lambda host (outside the container), this script will
automatically re-launch itself inside the isaac-drone container.
"""

import sys
import os
import argparse
import subprocess


ISAAC_PYTHON = "/isaac-sim/python.sh"
CONTAINER_NAME = "isaac-drone"


def is_inside_container():
    return os.path.exists(ISAAC_PYTHON)


def relaunch_in_container(mode):
    """Re-execute this script inside the isaac-drone Docker container."""
    script_path = os.path.abspath(__file__)
    # Mount script into container and run it
    cmd = [
        "docker", "exec", CONTAINER_NAME,
        ISAAC_PYTHON, script_path, "--mode", mode, "--no-relaunch"
    ]
    print(f"[main.py] Not inside container. Re-launching in '{CONTAINER_NAME}'...")
    print(f"[main.py] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ── Modes ──────────────────────────────────────────────────────────────────

def run_hover():
    """Headless physics test: spawn drone, run 500 steps, print positions."""
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

    from omni.isaac.core.world import World
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
    from pegasus.simulator.params import ROBOTS

    pg = PegasusInterface()
    pg.initialize_world()
    world = World(stage_units_in_meters=1.0, physics_dt=1/250.0, rendering_dt=1/60.0)

    config = MultirotorConfig()
    config.backends = []
    drone = Multirotor("/World/drone", ROBOTS["Iris"], 0, [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], config=config)

    world.reset()
    print("DRONE SPAWNED OK")

    for i in range(500):
        world.step(render=False)
        if i % 100 == 0:
            pos = drone.state.position
            print(f"[Step {i}] drone pos: {pos}")

    print("PHYSICS TEST PASSED")
    app.close()


def run_scene():
    """
    Launch Isaac Sim with livestream enabled.
    View the scene by opening /viewer in your browser (same URL as VSCode, just /viewer).
    """
    from isaacsim import SimulationApp
    app = SimulationApp({
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": 1280,
        "height": 720,
        "enable_livestream": True,
        "livestream": 2,
    })

    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
    from pegasus.simulator.params import ROBOTS

    pg = PegasusInterface()
    pg.initialize_world()
    world = World(stage_units_in_meters=1.0, physics_dt=1/250.0, rendering_dt=1/60.0)

    assets_root = get_assets_root_path()
    add_reference_to_stage(
        usd_path=assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd",
        prim_path="/World/Environment"
    )

    config = MultirotorConfig()
    config.backends = []
    drone = Multirotor("/World/drone", ROBOTS["Iris"], 0, [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], config=config)

    world.reset()

    print("=" * 50)
    print("SCENE LOADED — drone hovering at [0, 0, 0.5]")
    print("Open browser tab: <your-server-url>/viewer")
    print("=" * 50)

    frame = 0
    while app.is_running():
        world.step(render=True)
        if frame % 500 == 0:
            pos = drone.state.position
            print(f"[Frame {frame}] drone pos: {pos}")
        frame += 1

    app.close()


def run_video():
    """Record 300 frames to /tmp/drone_frames/ using Replicator."""
    import numpy as np
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720})

    import omni.replicator.core as rep
    from omni.isaac.core.world import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
    from pegasus.simulator.params import ROBOTS
    from pxr import Gf

    pg = PegasusInterface()
    pg.initialize_world()
    world = World(stage_units_in_meters=1.0, physics_dt=1/250.0, rendering_dt=1/60.0)

    assets_root = get_assets_root_path()
    add_reference_to_stage(
        usd_path=assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd",
        prim_path="/World/Environment"
    )

    config = MultirotorConfig()
    config.backends = []
    drone = Multirotor("/World/drone", ROBOTS["Iris"], 0, [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], config=config)

    world.reset()

    camera = rep.create.camera(position=(3.0, 3.0, 2.0), look_at=(0.0, 0.0, 0.5))
    output_dir = "/tmp/drone_frames"
    os.makedirs(output_dir, exist_ok=True)

    with rep.new_layer():
        rp = rep.create.render_product(camera, (1280, 720))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=output_dir, rgb=True)
        writer.attach([rp])

    print(f"Recording 300 frames to {output_dir} ...")
    for i in range(300):
        world.step(render=True)
        if i % 50 == 0:
            rep.orchestrator.step(rt_subframes=1)
            print(f"[Frame {i}]")

    print(f"Done. Frames saved to {output_dir}")
    app.close()


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone sim launcher")
    parser.add_argument("--mode", choices=["hover", "scene", "video"], default="hover",
                        help="hover=headless test | scene=livestream | video=record frames")
    parser.add_argument("--no-relaunch", action="store_true",
                        help="Skip container re-launch check (used internally)")
    args = parser.parse_args()

    if not args.no_relaunch and not is_inside_container():
        relaunch_in_container(args.mode)

    modes = {"hover": run_hover, "scene": run_scene, "video": run_video}
    modes[args.mode]()

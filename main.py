#!/usr/bin/env python3
"""
Drone simulation entry point for Isaac Sim + Pegasus Simulator.

Requires: conda activate isaac

Usage:
    python main.py --mode hover    # headless physics test
    python main.py --mode scene    # livestream scene (view at <server-url>/viewer)
    python main.py --mode video    # record frames to /tmp/drone_frames/
"""

import os
import argparse
import warnings

# Suppress noisy Isaac Sim deprecation/extension warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["OMNI_LOG_LEVEL"] = "ERROR"

import numpy as np


class HoverBackend:
    """Simple PD altitude-hold hover controller.

    Implements the Pegasus Backend interface without inheriting from it to
    avoid import-order issues (Isaac Sim must be started first).

    input_reference() returns a list of 4 rotor angular velocities [rad/s].
    The thrust model is:  T = kT * omega^2  (quadratic thrust curve).
    """

    KP   = 50.0         # altitude proportional gain
    KD   = 20.0         # altitude derivative gain
    MASS = 1.50         # kg — overridden at runtime from USD if possible
    G    = 9.81         # m/s²
    KT   = 8.54858e-6   # N·s²/rad²  (Pegasus default quadratic thrust coeff)
    MAX_OMEGA = 2100.0  # rad/s — raised above Pegasus's soft 1100 limit for safety

    def __init__(self, target_z: float = 0.5):
        self._target_z = target_z
        self._state    = None
        self._omega    = [0.0, 0.0, 0.0, 0.0]
        self._mass     = self.MASS

    # ------------------------------------------------------------------ #
    # Pegasus Backend interface                                            #
    # ------------------------------------------------------------------ #
    def initialize(self, vehicle):
        """Read actual USD physics mass so the hover thrust is correct."""
        try:
            import omni.isaac.core.utils.prims as prim_utils
            from pxr import UsdPhysics
            prim = prim_utils.get_prim_at_path(vehicle._stage_prefix + "/body")
            mass_api = UsdPhysics.MassAPI(prim)
            usd_mass = mass_api.GetMassAttr().Get()
            if usd_mass and usd_mass > 0:
                self._mass = float(usd_mass)
                print(f"[HoverBackend] USD mass = {self._mass:.3f} kg")
        except Exception:
            pass  # keep default MASS

    def update_sensor(self, sensor_type, data):
        pass

    def update_graphical_sensor(self, sensor_type, data):
        pass

    def update_state(self, state):
        self._state = state

    def input_reference(self):
        return self._omega

    def update(self, dt: float):
        if self._state is None:
            return

        z  = self._state.position[2]
        vz = self._state.linear_velocity[2]

        error_z  = self._target_z - z
        error_vz = -vz  # target vertical velocity is 0

        # Desired total thrust from PD law
        total_thrust = self._mass * (self.G + self.KP * error_z + self.KD * error_vz)
        total_thrust = max(0.0, total_thrust)

        # Equal thrust per rotor → rotor angular velocity
        thrust_per_rotor = total_thrust / 4.0
        omega = float(np.sqrt(thrust_per_rotor / self.KT))
        omega = min(omega, self.MAX_OMEGA)

        self._omega = [omega, omega, omega, omega]

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self._state = None
        self._omega = [0.0, 0.0, 0.0, 0.0]


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
    config.backends = [HoverBackend(target_z=0.5)]
    drone = Multirotor("/World/drone", ROBOTS["Iris"], 0, [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], config=config)

    world.reset()
    print("DRONE SPAWNED OK")

    for i in range(500):
        world.step(render=False)
        if i % 100 == 0:
            pos = drone.state.position
            print(f"[Step {i}] drone pos: {pos}")

    final_z = drone.state.position[2]
    if abs(final_z - 0.5) < 0.1:
        print(f"HOVER TEST PASSED  (final z={final_z:.3f}m, target=0.5m)")
    else:
        print(f"HOVER TEST FAILED  (final z={final_z:.3f}m, target=0.5m)")
    app.close()


STREAMING_KIT = (
    "/home/ubuntu/miniconda3/envs/isaac/lib/python3.11/site-packages/"
    "isaacsim/apps/isaacsim.exp.full.streaming.kit"
)


def run_scene():
    """
    Launch Isaac Sim with WebRTC livestream enabled.
    Always records frames and stitches drone.mp4 on exit.

    SSH tunnel first (run on your local machine):
        ssh -L 49100:localhost:49100 ubuntu@<server-ip> -N

    Then open:  http://localhost:49100/streaming/client/
    """
    import socket, subprocess, glob, re
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "width": 1280, "height": 720, "hide_ui": False},
        experience=STREAMING_KIT,
    )

    import omni.replicator.core as rep
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    import omni.isaac.core.utils.prims as prim_utils
    from pxr import UsdGeom, Gf
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
    config.backends = [HoverBackend(target_z=1.5)]
    drone = Multirotor("/World/drone", ROBOTS["Iris"], 0, [0.0, 0.0, 1.5], [1.0, 0.0, 0.0, 0.0], config=config)

    world.reset()

    camera = rep.create.camera(position=(4.0, 4.0, 3.0), look_at=(0.0, 0.0, 1.5))

    # FPV camera attached to drone body — moves with the drone
    fpv_cam_path = "/World/drone/fpv_camera"
    fpv_cam_prim = prim_utils.create_prim(fpv_cam_path, "Camera")
    xform = UsdGeom.Xformable(fpv_cam_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.15, 0.0, 0.05))   # front of drone, slightly above
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-10.0, -90.0, 0.0))  # face forward along +X, 10° nose-down tilt

    fpv_output_dir = "/tmp/drone_fpv"
    os.makedirs(fpv_output_dir, exist_ok=True)
    fpv_rp = rep.create.render_product(fpv_cam_path, (1280, 720))
    fpv_writer = rep.WriterRegistry.get("BasicWriter")
    fpv_writer.initialize(output_dir=fpv_output_dir, rgb=True)
    fpv_writer.attach([fpv_rp])

    try:
        server_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        server_ip = "<server-ip>"

    print("=" * 60)
    print("SCENE LOADED — drone hovering at z=1.5m")
    print(f"Direct:          http://{server_ip}:49100/streaming/client/")
    print("Via SSH tunnel:  http://localhost:49100/streaming/client/")
    print(f"FPV frames:      {fpv_output_dir}/")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    def stitch_video(frames_dir, video_name):
        print(f"Creating video from {frames_dir}...")
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not frames:
            print(f"No frames in {frames_dir} — skipping.")
            return
        first = os.path.basename(frames[0])
        pattern = re.sub(r'\d+', lambda m: f'%0{len(m.group())}d', first)
        video_path = os.path.join(frames_dir, video_name)
        cmd = [
            "ffmpeg", "-y", "-framerate", "30",
            "-i", os.path.join(frames_dir, pattern),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path,
        ]
        print(f"Stitching {len(frames)} frames → {video_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Video saved to {video_path}")
        else:
            print(f"ffmpeg failed:\n{result.stderr}")

    frame = 0
    try:
        while app.is_running():
            world.step(render=True)
            rep.orchestrator.step(rt_subframes=1)
            if frame % 500 == 0:
                pos = drone.state.position
                print(f"[Frame {frame}] drone pos: {pos}")
            frame += 1
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        try:
            app.close()
        except Exception:
            pass
        stitch_video("/tmp/drone_frames", "drone.mp4")
        stitch_video(fpv_output_dir, "drone_fpv.mp4")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone sim launcher")
    parser.add_argument("--mode", choices=["hover", "scene"], default="hover",
                        help="hover=headless test | scene=livestream + auto video")
    args = parser.parse_args()

    modes = {"hover": run_hover, "scene": run_scene}
    modes[args.mode]()

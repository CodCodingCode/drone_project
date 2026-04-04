#!/usr/bin/env python3
"""Quick FPV camera angle test — pure USD prims, no Isaac wrappers.

Usage:
    LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
      /home/ubuntu/miniconda3/envs/isaac/bin/python test_fpv_camera.py
"""

import os, glob
import numpy as np
os.environ["OMNI_LOG_LEVEL"] = "ERROR"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting",
                     "width": 1280, "height": 720})

print("[test_fpv] App started", flush=True)

import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom, UsdShade, UsdLux, Gf, Sdf, Vt

stage = omni.usd.get_context().get_stage()

# ---- Lighting ----
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(1500.0)

distant = UsdLux.DistantLight.Define(stage, "/World/Sun")
distant.CreateIntensityAttr(5000.0)
UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

# ---- Ground plane (large flat cube) ----
ground = UsdGeom.Cube.Define(stage, "/World/Ground")
UsdGeom.Xformable(ground.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
UsdGeom.Xformable(ground.GetPrim()).AddScaleOp().Set(Gf.Vec3d(20, 20, 0.01))

def make_colored_cube(path, pos, size, color_rgb):
    cube = UsdGeom.Cube.Define(stage, path)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3d(size, size, size))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color_rgb)])

print("[test_fpv] Creating scene...", flush=True)

# Cubes at all 4 cardinal directions from drone (0,0,1.5), 3m away, same height
make_colored_cube("/World/Red_posX",  (3, 0, 1.5),   0.5, (1, 0, 0))
make_colored_cube("/World/Green_negX",(-3, 0, 1.5),   0.5, (0, 1, 0))
make_colored_cube("/World/Blue_posY", (0, 3, 1.5),   0.5, (0, 0, 1))
make_colored_cube("/World/Yellow_negY",(0,-3, 1.5),   0.5, (1, 1, 0))
# Also one on the ground in front
make_colored_cube("/World/White_floor",(2, 0, 0.25),  0.25, (1, 1, 1))

# ---- Camera at "drone" position ----
cam = UsdGeom.Camera.Define(stage, "/World/FPV_Camera")
cam_xf = UsdGeom.Xformable(cam.GetPrim())
cam_xf.ClearXformOpOrder()
cam_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.5))

# USD camera default looks down -Z.
# To look along +X (forward): rotate 90° around Y
# Then tilt down 10°: rotate -10° around the new local X
# RotateXYZ applies: first X, then Y, then Z (intrinsic)
# +90 Y pointed at -X (green). So -90 Y should point at +X (red = forward).
# Add -10 X tilt for slight nose-down.
cam_xf.AddRotateXYZOp().Set(Gf.Vec3f(-10.0, -90.0, 0.0))

print("[test_fpv] Camera set, rendering...", flush=True)

out_dir = "/tmp/drone_fpv_test"
os.makedirs(out_dir, exist_ok=True)
# Clean old frames
for f in glob.glob(os.path.join(out_dir, "*.png")):
    os.remove(f)

rp = rep.create.render_product("/World/FPV_Camera", (1280, 720))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=out_dir, rgb=True)
writer.attach([rp])

from omni.isaac.core import World
world = World()
world.reset()

# Warmup
for _ in range(30):
    world.step(render=True)

# Capture
for i in range(10):
    world.step(render=True)
    rep.orchestrator.step(rt_subframes=8)
    if i % 5 == 0:
        print(f"  [Frame {i}]", flush=True)

frames = sorted(glob.glob(os.path.join(out_dir, "*.png")))
print(f"\nDone. {len(frames)} frames in {out_dir}/", flush=True)
if frames:
    print(f"  View: {frames[-1]}", flush=True)

app.close()

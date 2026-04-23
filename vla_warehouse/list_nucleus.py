"""List available USD files under ISAAC_NUCLEUS_DIR/Environments/.

Run once to discover the exact filenames your Nucleus has. Use the output
to fix usd_path values in vla_warehouse/pois.py.

Launch:
    bash ~/drone_project/vla_warehouse/run_list_nucleus.sh
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args(["--headless"])
args_cli.headless = True

app = AppLauncher(args_cli)
sim_app = app.app

# --- After app init, pxr and omni.client are available ---
import omni.client
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

print("\n" + "=" * 72)
print(f"ISAAC_NUCLEUS_DIR = {ISAAC_NUCLEUS_DIR}")
print("=" * 72)

dirs_to_explore = [
    "Environments",
    "Environments/Simple_Warehouse",
    "Environments/Hospital",
    "Environments/Office",
    "Environments/Simple_Room",
    "Environments/Grid",
]

for subpath in dirs_to_explore:
    full_path = f"{ISAAC_NUCLEUS_DIR}/{subpath}/"
    result, entries = omni.client.list(full_path)
    print(f"\n--- {subpath}/ ---")
    if result != omni.client.Result.OK:
        print(f"  (not accessible — result: {result})")
        continue
    if not entries:
        print("  (empty)")
        continue
    for e in entries:
        is_dir = bool(e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
        suffix = "/" if is_dir else ""
        print(f"  {e.relative_path}{suffix}")

print("\n" + "=" * 72)
print("Done. Look for .usd files under Simple_Warehouse/, Hospital/, Office/.")
print("Copy the exact filenames into vla_warehouse/pois.py's SCENES dict.")
print("=" * 72 + "\n")

sim_app.close()

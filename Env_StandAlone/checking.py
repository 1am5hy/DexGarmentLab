from random import randint
import carb

"""Launch Isaac Sim Simulator first."""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "0"

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.device = 'cuda:0'
# args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import random
import torch
import tqdm
import pickle
import datetime
import contextlib

from pxr import Usd
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app


def inspect_usd_structure(usd_path):
    """
    Load a USD file and print its prim hierarchy.

    Args:
        usd_path (str): Path to the USD file
    """
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Could not open USD file: {usd_path}")

    print(f"USD Structure for {usd_path}:\n")

    # Traverse all prims in the stage
    for prim in stage.Traverse():
        print(prim.GetPath())

    # Print the defaultPrim if set
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        print(f"\nDefaultPrim: {default_prim.GetPath()}")
    else:
        print("\nNo defaultPrim set in this USD file.")

    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        stage_utils.open_stage(usd_path)
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


# Example usage
if __name__ == "__main__":
    # /home/ubuntu/Github/DexGarmentLab/Assets/Garment/Tops/NoCollar_Ssleeve_FrontClose/TNSC_091/TNSC_091_obj.usd
    inspect_usd_structure("/home/ubuntu/Downloads/World.usd")


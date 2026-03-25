from random import randint

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

# check_usd_mesh_prims.py
from pxr import Usd, UsdGeom, Sdf
import sys
from typing import List, Optional

def open_stage(file_path: Optional[str] = None) -> Usd.Stage:
    """
    Open a USD stage from file_path if provided, otherwise try to get the current stage.
    Raises RuntimeError if no stage can be obtained.
    """
    if file_path:
        stage = Usd.Stage.Open(file_path)
        if not stage:
            raise RuntimeError(f"Failed to open USD file: {file_path}")
        return stage

    # Try to get the current stage (works inside apps that set a current stage)
    stage = Usd.Stage.GetCurrent()
    if stage and stage.GetRootLayer():
        return stage

    raise RuntimeError("No USD file path provided and no current stage available.")

def get_prim_at_path(stage: Usd.Stage, prim_path: str) -> Optional[Usd.Prim]:
    """Return the prim at prim_path or None if not found."""
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    return prim if prim.IsValid() else None

def is_mesh_prim(prim: Usd.Prim) -> bool:
    """Return True if prim is a UsdGeom.Mesh."""
    return UsdGeom.Mesh(prim).IsValid()

def list_all_mesh_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """Return a list of all UsdGeom.Mesh prims on the stage."""
    meshes = []
    for prim in stage.Traverse():
        if is_mesh_prim(prim):
            meshes.append(prim)
    return meshes

def find_mesh_by_path(stage: Usd.Stage, prim_path: str) -> Optional[Usd.Prim]:
    """Check whether a mesh prim exists at prim_path and return it if valid."""
    prim = get_prim_at_path(stage, prim_path)
    if prim and is_mesh_prim(prim):
        return prim
    return None

def find_meshes_by_name(stage: Usd.Stage, name_substring: str) -> List[Usd.Prim]:
    """Return mesh prims whose name contains name_substring (case-sensitive)."""
    matches = []
    for prim in list_all_mesh_prims(stage):
        if name_substring in prim.GetName():
            matches.append(prim)
    return matches

def print_prim_summary(prim: Usd.Prim):
    """Print a short summary for a prim."""
    print(f"Path: {prim.GetPath()}")
    print(f"  Name: {prim.GetName()}")
    print(f"  TypeName: {prim.GetTypeName()}")
    print(f"  IsMesh: {is_mesh_prim(prim)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Locate mesh prims in a USD file or current stage.")
    parser.add_argument("--usd", "-u", help="Path to USD file (optional). If omitted, uses current stage.", default="/home/ubuntu/Downloads/World.usd")
    parser.add_argument("--path", "-p", help="Exact prim path to check (e.g. /World/envs/env_0/Garment/geometry/mesh).", default="/World/geometry/mesh")
    parser.add_argument("--name", "-n", help="Search mesh prims by name substring.")
    parser.add_argument("--list", "-l", action="store_true", help="List all mesh prims.")
    args = parser.parse_args()

    try:
        stage = open_stage(args.usd)
    except RuntimeError as e:
        print("Error:", e)
        sys.exit(1)

    if args.path:
        prim = get_prim_at_path(stage, args.path)
        if not prim:
            print(f"No prim found at path: {args.path}")
            sys.exit(0)
        print("Prim found at path.")
        print_prim_summary(prim)
        if not is_mesh_prim(prim):
            print("Note: prim exists but is not a UsdGeom.Mesh.")
        sys.exit(0)

    if args.name:
        matches = find_meshes_by_name(stage, args.name)
        if not matches:
            print(f"No mesh prims found with name containing: {args.name}")
            sys.exit(0)
        print(f"Found {len(matches)} mesh prim(s) matching name '{args.name}':")
        for m in matches:
            print_prim_summary(m)
        sys.exit(0)

    if args.list:
        meshes = list_all_mesh_prims(stage)
        if not meshes:
            print("No mesh prims found on the stage.")
            sys.exit(0)
        print(f"Found {len(meshes)} mesh prim(s):")
        for m in meshes:
            print_prim_summary(m)
        sys.exit(0)

    # Default behavior: show short help
    parser.print_help()

if __name__ == "__main__":
    main()

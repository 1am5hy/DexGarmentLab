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
import trimesh
from pxr import Usd, UsdGeom, Gf

def obj_to_usd(obj_path, usd_path, scale=1.0):
    mesh = trimesh.load(obj_path, force='scene')  # load as scene to capture multiple geometries

    stage = Usd.Stage.CreateNew(usd_path)
    world = UsdGeom.Xform.Define(stage, "/World")
    # mesh_container = UsdGeom.Xform.Define(stage, "/World/mesh")

    # Iterate over geometries
    for i, geom in enumerate(mesh.geometry.values()):
        mesh_path = f"/World/mesh{i}" if i > 0 else "/World/mesh"
        mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)

        # Vertices
        mesh_prim.CreatePointsAttr([Gf.Vec3f(*v) for v in geom.vertices])
        # Faces
        mesh_prim.CreateFaceVertexIndicesAttr([int(i) for face in geom.faces for i in face])
        mesh_prim.CreateFaceVertexCountsAttr([len(face) for face in geom.faces])

        # Normals if available
        if geom.vertex_normals is not None:
            mesh_prim.CreateNormalsAttr([Gf.Vec3f(*n) for n in geom.vertex_normals])
            mesh_prim.SetNormalsInterpolation("vertex")

    # Set defaultPrim
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()
    print(f"✅ Converted {obj_path} → {usd_path} with {len(mesh.geometry)} meshes")

# Example usage
if __name__ == "__main__":
    obj_to_usd("/home/ubuntu/Downloads/Tshirt.obj", "/home/ubuntu/Github/DexGarmentLab/Assets/Tshirt.usd", scale=0.01)

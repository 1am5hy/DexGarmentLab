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

from pxr import Usd, UsdGeom, Sdf

stage = Usd.Stage.Open('/home/ubuntu/Downloads/World.usd')
prim = stage.GetPrimAtPath('/World/geometry/mesh')

print('prim:', prim)
print('prim.IsValid():', prim.IsValid())
print('GetTypeName():', prim.GetTypeName())

def is_mesh_prim(prim):
    return prim.IsValid() and prim.IsA(UsdGeom.Mesh)

print('IsMesh:', is_mesh_prim(prim))

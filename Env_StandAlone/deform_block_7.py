# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""
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
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
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


import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaacsim.core.utils.prims import get_prim_at_path, delete_prim, create_prim
from isaaclab.assets import ParticleObject, ParticleObjectCfg, RigidObjectCfg, RigidObject, DeformableObject, DeformableObjectCfg
from pxr import PhysxSchema, Usd, UsdPhysics, Gf, Sdf
from isaacsim.core.cloner import GridCloner
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.objects.sphere import DynamicSphere
# from isaaclab.assets import Rigi

from isaacsim.core.utils.stage import get_current_stage

from Data_Gen_Utils import find_grasp_points, mk_dir
# from gen_utils.traj_gen import generate_bimanual_arc_from_given_picks
from gen_utils.grasp_selection import select_two_boundary_picks_batch
from gen_utils.traj_gen_2 import bimanual_arc_trajectories, mask_actions_torch

from attach_block import AttachmentBlock
from Collision import CollisionGroup
##
# Pre-defined configs
##
# from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

@configclass
class ParticleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1, dynamic_friction=1)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Camera rot - (0, 0, 0.5555702, 0.8314696 )
    # 0.3826834, 0, 0, 0.9238795
    tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.0, 0.5, 1.25), rot=(0, 0,  0.258819, 0.9659258), convention="opengl"),
        data_types=["distance_to_image_plane", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        colorize_instance_id_segmentation=False,
        width=640,
        height=480,
    )

    # long sleeve : /home/ubuntu/Github/DexGarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Top074/TNLC_Top074_obj.usd
    # Short Sleeve : /home/ubuntu/Github/DexGarmentLab/Assets/Garment/Tops/NoCollar_Ssleeve_FrontClose/TNSC_091/TNSC_091_obj.usd

    # particle_obj = ParticleObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Garment",
    #     spawn=sim_utils.SoftUsdFileCfg(
    #         usd_path=f"/home/ubuntu/Github/DexGarmentLab/Assets/Garment/Tops/NoCollar_Ssleeve_FrontClose/TNSC_091/TNSC_091_obj.usd",
    #         translation=(0, -1.25, 0.075),
    #         # translation=(0, -1.25, 0.2),
    #         # translation=(0, 0, 0),
    #         scale=(0.01, 0.01, 0.01),
    #         # scale=(0.015, 0.015, 0.015),
    #         particle_props=sim_utils.ParticleBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1e-1),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0., 0.5), metallic=0.2),
    #         physics_material=sim_utils.ParticleBodyMaterialCfg(),
    #     ),
    # )

    particle_obj = ParticleObjectCfg(
        prim_path="{ENV_REGEX_NS}/Garment",
        spawn=sim_utils.SoftUsdFileCfg(
            usd_path=f"/home/ubuntu/Downloads/World.usd",
            # translation=(0, -1.25, 0.075),
            # translation=(0, -1.25, 0.2),
            translation=(0, 0, 0.0),
            # orientation=(0.7071068, 0.7071068, 0, 0),
            scale=(1, 1, 1),
            # scale=(0.015, 0.015, 0.015),
            particle_props=sim_utils.ParticleBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.,  0.5), metallic=0.2),
            physics_material=sim_utils.ParticleBodyMaterialCfg(),
        ),
    )


    # particle_obj = ParticleObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Garment",
    #     spawn=sim_utils.MeshCuboidCfg(
    #         size=(0.5, 0.5, 0.0001),
    #         visible=True,
    #         particle_props=sim_utils.ParticleBodyPropertiesCfg(),
    #         # collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.ParticleBodyMaterialCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0., 0.5), metallic=0.2),
    #     ),
    #     init_state=ParticleObjectCfg.InitialStateCfg(),
    # )

    sphere1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sphere1",
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            # radius=0.02,
            visible=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100000),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False, contact_offset=0.005, rest_offset=0.001),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, -0.2, 0.05))
    )

    sphere2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sphere2",
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            # radius=0.02,
            visible=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100000),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False, contact_offset=0.005, rest_offset=0.001),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.1, -0.2, 0.05))
    )

def create_attachment(attachment_prim, attachment_prim2,  stage, custom=False):
    # attachment_path_2
    if custom == "Cloth":
        for i in range(len(attachment_prim.root_physx_view.prim_paths)):
            attachment_path = attachment_prim.root_physx_view.prim_paths[i].replace("Sphere2/geometry/mesh", "Garment/geometry/Geom/Towel") + "/attachment_2"
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
            attachment.GetActor0Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i].replace("Sphere2/geometry/mesh",
                                                                       "Garment/geometry/Geom/Towel")])
            attachment.GetActor1Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i]])
            att = PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
            att.Apply(attachment.GetPrim())
            # _ = att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.02)
    else:
        for i in range(len(attachment_prim.root_physx_view.prim_paths)):
            attachment_path1 = attachment_prim.root_physx_view.prim_paths[i].replace("Sphere1",
                                                                                    "Garment") + "/attachment_1"
            attachment_path2 = attachment_prim2.root_physx_view.prim_paths[i].replace("Sphere2", "Garment") + "/attachment_2"
            attachment1 = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path1)
            attachment2 = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path2)
            attachment1.GetActor0Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i].replace("Sphere1", "Garment")])
            attachment2.GetActor0Rel().SetTargets(
                [attachment_prim2.root_physx_view.prim_paths[i].replace("Sphere2", "Garment")])
            attachment1.GetActor1Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i]])
            attachment2.GetActor1Rel().SetTargets(
                [attachment_prim2.root_physx_view.prim_paths[i]])
            att = PhysxSchema.PhysxAutoAttachmentAPI(attachment1.GetPrim())
            att2 = PhysxSchema.PhysxAutoAttachmentAPI(attachment2.GetPrim())
            att.Apply(attachment1.GetPrim())
            _ = att.CreateCollisionFilteringOffsetAttr(defaultValue=0.4)
            _ = att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.04)
            att2.Apply(attachment2.GetPrim())
            # 3
            _ = att2.CreateCollisionFilteringOffsetAttr(defaultValue=0.4)
            # 0.01
            _ = att2.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.04)

def attachment_check(obj_poses, part_poses, threshold):
    """
    PyTorch version. Accepts tensors on any device.
    Args:
      obj_poses: torch.Tensor shape (B,2,3)
      part_poses: torch.Tensor shape (B,2,3)
      threshold: float or torch.Tensor broadcastable to (B,2)
    Returns:
      failed_mask: torch.BoolTensor shape (B,2)
      distances: torch.Tensor shape (B,2)
    """
    diff = obj_poses - part_poses                     # (B,2,3)
    distances = torch.norm(diff, dim=-1)              # (B,2)
    failed_mask = distances < threshold
    return failed_mask, distances

def delete_attachment(scene):
    for i in range(scene.num_envs):
        delete_prim(f"/World/envs/env_{i}/Garment/geometry/mesh/attachment_1")
        delete_prim(f"/World/envs/env_{i}/Garment/geometry/mesh/attachment_2")

def delete_rigid_prim(scene):
    for i in range(scene.num_envs):
        delete_prim(f"/World/envs/env_{i}/Sphere1")
        delete_prim(f"/World/envs/env_{i}/Sphere2")

def change_reset_point(scene, pos):
    stage = get_current_stage()
    prim_paths = scene.env_prim_paths

    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        i = 0
        for prim_path in prim_paths:
            # spawn single instance
            prim_path1 = prim_path + "/Sphere1"
            prim_path2 = prim_path + "/Sphere2"
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path1)
            prim_spec2 = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path2)

            sphere_pose = prim_spec.GetAttributeAtPath(prim_path1 + ".xformOp:translate")
            sphere_pose2 = prim_spec2.GetAttributeAtPath(prim_path2 + ".xformOp:translate")

            sphere_pose.default = Gf.Vec3f(float(pos[i, 0, 0]), float(pos[i, 0, 1]), float(pos[i, 0, 2]))
            sphere_pose2.default = Gf.Vec3f(float(pos[i, 1, 0]), float(pos[i, 1, 1]), float(pos[i, 1, 2]))
            # sphere_pose.default = Gf.Vec3f(0, 0, 0)
            # sphere_pose2.default = Gf.Vec3f(0, 0, 3)
            i = i + 1

    return


# ── Randomize cloth stiffness across all envs simultaneously ─────────────────
def _debug_find_particle_material(stage, root_path: str):
    """Helper to find the correct prim path and attribute names for particle material."""
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        print(f"  [DEBUG] Root prim {root_path} not valid.")
        return
    for prim in Usd.PrimRange(root_prim):
        apis = prim.GetAppliedSchemas()
        if any("Particle" in api for api in apis):
            print(f"  [DEBUG] Found particle-related prim : {prim.GetPath()}")
            print(f"  [DEBUG] Applied schemas             : {apis}")
            print(f"  [DEBUG] Attributes                  : {[a.GetName() for a in prim.GetAttributes()]}")


def randomize_cloth_stiffness(scene: InteractiveScene) -> dict:
    """
    Randomizes spring_bend_stiffness for all envs simultaneously using a
    single Sdf.ChangeBlock() — identical pattern to change_reset_point().

    Sampling is log-uniform over [1e3, 1e8] to cover the physical range
    of cloth materials (silky -> stiff denim) evenly in log space.

    Returns a dict {env_index: sampled_stiffness} for dataset logging.
    """
    stage = get_current_stage()
    num_envs = scene.num_envs

    # Sample all values before entering the change block
    ranges = [(-1, 0), (4, 5), (6, 7)]
    weights = [0.75, 0.20, 0.05]
    sampled = [float(round(10 ** np.random.uniform(*ranges[np.random.choice(len(ranges), p=weights)]), 1)) for _ in
               range(num_envs)]

    # sampled = [1e6 for _ in range(num_envs)]

    with Sdf.ChangeBlock():
        for i in range(num_envs):
            mat_prim_path = (
                f"/World/envs/env_{i}/Garment/geometry/mesh"
            )

            prim_spec = Sdf.CreatePrimInLayer(
                stage.GetRootLayer(), mat_prim_path
            )

            attr_spec = prim_spec.GetAttributeAtPath(
                mat_prim_path + ".physxAutoParticleCloth:springBendStiffness"
            )

            if attr_spec:
                attr_spec.default = float(sampled[i])
            else:
                # Create the attribute spec if it does not exist yet
                attr_spec = Sdf.AttributeSpec(
                    prim_spec,
                    "physxParticleBody:springBendStiffness",
                    Sdf.ValueTypeNames.Float,
                )
                attr_spec.default = float(sampled[i])

            print(f"[ENV {i}] spring_bend_stiffness → {sampled[i]:.4e}")

    return {i: sampled[i] for i in range(num_envs)}
# ─────────────────────────────────────────────────────────────────────────────


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, save):
    """Runs the simulation loop."""
    # Extract scene entities
    garment = scene["particle_obj"]
    sphere = scene["sphere1"]
    sphere2 = scene["sphere2"]

    stage = scene.stage

    # Define goals for the arm
    garment_entity_cfg = SceneEntityCfg("particle_obj")
    garment_entity_cfg.resolve(scene)
    sphere_entity_cfg = SceneEntityCfg("sphere1")
    sphere_entity_cfg.resolve(scene)
    sphere_entity_cfg2 = SceneEntityCfg("sphere2")
    sphere_entity_cfg2.resolve(scene)


    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # pos = garment.print_nodal_position_from_sim().detach()

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 1100 == 0:
            # reset time
            count = 0
            """  
            Reset the robot and garment to their initial states.  
            """
            "Garment"
            delete_attachment(scene)
            init_state = garment.data.nodal_pos_w.clone()
            init_vel = garment.data.nodal_vel_w.clone()
            init_vel.zero_()
            garment.write_nodal_pos_to_sim(init_state)
            garment.write_nodal_velocity_to_sim(init_vel)

            # sampled_stiffness = randomize_cloth_stiffness(scene)
            sim.reset()  # PhysX ingests all USD changes in one shot here

            num_fold = 2
            pnp_len = int(900/num_fold)

        if count == 43:
            # min_distance=0.4
            pick_pos, idx1, idx2 = select_two_boundary_picks_batch(
                garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3),
                min_distance=0.4)
            z_offset = torch.zeros(pick_pos.shape, device="cuda")
            z_offset[:, :, 2] = 0.0
            pick_pos = pick_pos + z_offset

            base = torch.tensor(scene.cloner._positions, device="cuda", dtype=torch.float32)
            pick_position = pick_pos[:, :, :3] - base.repeat(2, 1, 1).permute(1, 0, 2)
            pick_position = pick_position
            # pick_position[:, :, 2] =

            new_pick = pick_position + base.repeat(2, 1, 1).permute(1, 0, 2)
            pick_pos[:, :, :3] = new_pick

            change_reset_point(scene, pick_position)
            sphere.reset()
            sphere2.reset()
            sphere.write_root_com_pose_to_sim(pick_pos[:, 0])
            sphere2.write_root_com_pose_to_sim(pick_pos[:, 1])
            # garment.write_nodal_pos_to_sim(state)
            # garment.write_nodal_velocity_to_sim(init_vel)

        if count == 44:
            garment_entity_cfg = SceneEntityCfg("particle_obj")
            garment_entity_cfg.resolve(scene)
            sphere = scene["sphere1"]
            sphere2 = scene["sphere2"]
            stage = scene.stage
            init_state = garment.data.nodal_pos_w
            init_vel = garment.data.nodal_vel_w
            create_attachment(sphere, sphere2, stage)

        elif 50 + pnp_len > count > 50:
            if count == 51:
                # Generate grasping trajectory
                # picks = torch.zeros(scene.num_envs, 2, 7, device="cuda")
                # picks[:, :, -1] = 1
                # picks[:, :, :3] = torch.tensor(scene.cloner._positions, device="cuda", dtype=torch.float32).repeat(2, 1, 1).permute(1, 0, 2) + pick_position[:, :3]
                traj = bimanual_arc_trajectories( garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3), pick_pos[:, :, :3], 0.3, 0.35,  int(pnp_len))
                # traj = bimanual_arc_trajectories(garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3),
                #                                  pick_pos[:, :, :3], 0.15, 0.2, int(pnp_len))

            sphere.write_root_com_pose_to_sim(traj[:, 0, count - 51])
            sphere2.write_root_com_pose_to_sim(traj[:, 1, count - 51])

            if count == 50 + 80:
                num_env = scene.num_envs
                rigid_poses = torch.cat((sphere.data.root_link_pose_w[:, :3].unsqueeze(1), sphere2.data.root_link_pose_w[:, :3].unsqueeze(1)), dim=1)
                state = garment.data.nodal_pos_w
                grasp1 = state.reshape(num_env, -1, 3)[torch.arange(num_env), idx1].unsqueeze(1)
                grasp2 = state.reshape(num_env, -1, 3)[torch.arange(num_env), idx2].unsqueeze(1)
                grasps = torch.cat((grasp1, grasp2), dim=1)

                attach_state = attachment_check(rigid_poses, grasps, threshold=0.1)
                # print(grasp1)
                # print(attach_state)
                # print("attachment check complete.")

        elif count > 50 + pnp_len:
            if count == 50 + pnp_len + 1:
                delete_attachment(scene)
                garment.write_nodal_velocity_to_sim(init_vel)
            # if count == 50 + pnp_len + 2:
                # garment.write_nodal_velocity_to_sim(init_vel)

            if count == 50 + pnp_len + 2:
                init_state = garment.data.nodal_pos_w.clone()
                init_vel = garment.data.nodal_vel_w.clone()
                init_vel.zero_()
                # min_distance = 0.3
                pick_pos, idx1, idx2 = select_two_boundary_picks_batch(
                    init_state.reshape(scene.num_envs, -1, 3),
                    min_distance=0.4)
                z_offset = torch.zeros(pick_pos.shape, device="cuda")
                # z_offset[:, :, 2] = -0.015
                pick_pos = pick_pos + z_offset
                # pick_pos[:, :, 2] = 0

                base = torch.tensor(scene.cloner._positions, device="cuda", dtype=torch.float32)
                pick_position = pick_pos[:, :, :3] - base.repeat(2, 1, 1).permute(1, 0, 2)

                pick_position[:, :, :3] = pick_position[:, :, :3] * 0.8
                # pick_position = pick_position

                new_pick = pick_position.clone() + base.repeat(2, 1, 1).permute(1, 0, 2)
                pick_pos[:, :, :3] = new_pick

            if count == 50 + pnp_len + 3:
                change_reset_point(scene, pick_position)
                sphere.reset()
                sphere2.reset()
                sphere.write_root_com_pose_to_sim(pick_pos[:, 0])
                sphere2.write_root_com_pose_to_sim(pick_pos[:, 1])
                # create_attachment(sphere, sphere2, stage)
                # garment.write_nodal_velocity_to_sim(init_vel)
                # garment.write_nodal_pos_to_sim(init_state)

            if count == 50 + pnp_len + 32:
                init_state = garment.data.nodal_pos_w.clone()
                init_vel = garment.data.nodal_vel_w.clone()
                init_vel.zero_()

            if count == 50 + pnp_len + 33:
                garment.write_nodal_velocity_to_sim(torch.zeros_like(init_vel))
                garment.write_nodal_pos_to_sim(init_state)
                create_attachment(sphere, sphere2, stage)

            if 50 + pnp_len + 52 < count < 50 + pnp_len + 149:
                garment.write_nodal_velocity_to_sim(init_vel)

            if count == 50 + pnp_len + 149:
                traj = bimanual_arc_trajectories( garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3), pick_pos[:, :, :3], 0.3, 0.3,  int(pnp_len))
                # traj = bimanual_arc_trajectories(garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3),
                #                                  pick_pos[:, :, :3], 0.15, 0.15, int(pnp_len))

            if count > 50 + pnp_len + 150:
                sphere.write_root_com_pose_to_sim(traj[:, 0, count - (50 + pnp_len + 151)])
                sphere2.write_root_com_pose_to_sim(traj[:, 1, count - (50 + pnp_len + 151)])

            if count == 50 + pnp_len + 50 + 80:
                num_env = scene.num_envs
                rigid_poses = torch.cat((sphere.data.root_link_pose_w[:, :3].unsqueeze(1),sphere2.data.root_link_pose_w[:, :3].unsqueeze(1)), dim=1)
                state = garment.data.nodal_pos_w
                grasp1 = state.reshape(num_env, -1, 3)[torch.arange(num_env), idx1].unsqueeze(1)
                grasp2 = state.reshape(num_env, -1, 3)[torch.arange(num_env), idx2].unsqueeze(1)
                grasps = torch.cat((grasp1, grasp2), dim=1)

                attach_state = attachment_check(rigid_poses, grasps, threshold=0.05)
                # print(grasp1)
                # print(attach_state)
                # print("attachment check complete.")

        if save == True:
            """Saving the soft object state"""
            if count == 0:
                dt = datetime.datetime.now()
                save_dir = "/home/ubuntu/Github/DexGarmentLab/Env_StandAlone/object_data/shirt"
                folder = save_dir + f"/{dt.year}_{dt.month}_{dt.day}"

                save_time = dt.strftime("%H_%M_%S")
                save_folder = folder + f"/{save_time}"


                object_path = save_folder + f"/object"
                action_path = save_folder + f"/action"
                camera_path = save_folder + f"/camera/pixels"
                intrinsics_path = save_folder + f"/camera/intrinsics"
                segemented_path = save_folder + f"/camera/segment"
                mk_dir(object_path)
                mk_dir(action_path)
                mk_dir(camera_path)
                mk_dir(intrinsics_path)
                mk_dir(segemented_path)

                camera_pos = scene["tiled_camera"].data.pos_w
                camera_quat = scene["tiled_camera"].data.quat_w_ros
                torch.save(camera_pos, save_folder + f"/camera/pose.pt")
                torch.save(camera_quat, save_folder + f"/camera/quat.pt")


            if count > 49:

                if count%10 == 0:
                    now_time = datetime.datetime.now()
                    time = int(now_time.strftime("%Y%m%d%H%M%S%f"))
                    filename = save_folder + f"/object/{time}" + ".pt"
                    position = garment.data.nodal_pos_w
                    position = position.reshape(scene.num_envs, -1, 3)

                    torch.save(position, filename)

                    grasp1 = sphere.data.root_link_pose_w
                    grasp2 = sphere2.data.root_link_pose_w
                    grasp = torch.hstack([grasp1, grasp2])

                    if count == 50 + pnp_len:
                        grasp = mask_actions_torch(attach_state[0], grasp.reshape(len(grasp), 2, -1))
                        grasp = grasp.reshape(len(grasp), -1)
                    elif count == 50 + pnp_len + 140:
                        grasp = mask_actions_torch(attach_state[0], grasp.reshape(len(grasp), 2, -1))
                        grasp = grasp.reshape(len(grasp), -1)

                    filename2 = save_folder + f"/action/{time}" + ".pt"
                    torch.save(grasp, filename2)

                    camera_data = scene["tiled_camera"].data.output["distance_to_image_plane"]
                    segment = scene["tiled_camera"].data.output["instance_id_segmentation_fast"]
                    intrinsics = scene["tiled_camera"].data.intrinsic_matrices

                    filename3 = save_folder + f"/camera/pixels/{time}" + ".pt"
                    filename4 = save_folder + f"/camera/intrinsics/{time}" + ".pt"
                    filename5 = save_folder + f"/camera/segment/{time}" + ".pt"
                    torch.save(camera_data, filename3)
                    torch.save(intrinsics, filename4)
                    torch.save(segment, filename5)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        if count == 1100:
            print("Completed one demonstration instance")

        scene.update(sim_dt)



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.025, device=args_cli.device)
    # sim_cfg = sim_utils.SimulationCfg(dt=0.01, render_interval=4, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    # sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    sim.set_camera_view([1.25, 1.25, 1.25], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = ParticleSceneCfg(num_envs=args_cli.num_envs, env_spacing=3, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, save=True)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

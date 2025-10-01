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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.device = 'cuda:0'

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

from isaacsim.core.utils.stage import get_current_stage

from Data_Gen_Utils import find_grasp_points, mk_dir
from gen_utils.traj_gen import generate_bimanual_arc_from_given_picks
from gen_utils.grasp_selection import select_random_boundary_points_with_quat

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
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=100, dynamic_friction=100)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),


    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.0, 6.0, 3.0), rot=(0, 0, 0.5555702, 0.8314696 ), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )
    # Things that influence the particle / rigid attachment - distance to the rigid sphere at the start of the simulation (translation) - why? I DON'T KNOW
    # Things that influence the particle / rigid attachment - scale - why? I DON'T KNOW

    # particle_obj = ParticleObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Garment",
    #     spawn= sim_utils.SoftUsdFileCfg(
    #         usd_path=f"/home/hengyi/GitHub/DexGarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Top074/TNLC_Top074_obj.usd",
    #         translation=(0, -1.25, 0.05),
    #         # translation=(-0.2, 0.5, 0),
    #         # scale=(0.0185, 0.0185, 0.0185),
    #         scale=(0.0085, 0.0085, 0.0085),
    #         particle_props=sim_utils.ParticleBodyPropertiesCfg(),
    #         # mass_props=sim_utils.MassPropertiesCfg(mass=69),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(),
    #         physics_material=sim_utils.ParticleBodyMaterialCfg(),
    #     ),
    # )

    # /home/hengyi/GitHub/DexGarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Top074/TNLC_Top074_obj.usd

    particle_obj = ParticleObjectCfg(
        prim_path="{ENV_REGEX_NS}/Garment",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5, 0.0001),
            visible=True,
            particle_props=sim_utils.ParticleBodyPropertiesCfg(),
            # collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.ParticleBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.2),
        ),
        init_state=ParticleObjectCfg.InitialStateCfg(),
    )

def create_attachment_1(attachment_prim, stage, custom=False):
    # attachment_path_1
    if custom == "Cloth":
        for i in range(len(attachment_prim.root_physx_view.prim_paths)):
            attachment_path = attachment_prim.root_physx_view.prim_paths[i].replace("Sphere1/geometry/mesh",
                                                                                    "Garment/geometry/Geom/Towel") + "/attachment_1"
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
            attachment.GetActor0Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i].replace("Sphere1/geometry/mesh",
                                                                       "Garment/geometry/Geom/Towel")])
            attachment.GetActor1Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i]])
            att = PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
            att.Apply(attachment.GetPrim())
            _ = att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.02)
    else:
        for i in range(len(attachment_prim.root_physx_view.prim_paths)):
            attachment_path = attachment_prim.root_physx_view.prim_paths[i].replace("Sphere1",
                                                                                    "Garment") + "/attachment_1"
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
            attachment.GetActor0Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i].replace("Sphere1", "Garment")])
            attachment.GetActor1Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i]])
            att = PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
            att.Apply(attachment.GetPrim())
            _ = att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.02)

def create_attachment_2(attachment_prim, stage, custom=False):
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
            attachment_path = attachment_prim.root_physx_view.prim_paths[i].replace("Sphere2",
                                                                                    "Garment") + "/attachment_2"
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
            attachment.GetActor0Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i].replace("Sphere2", "Garment")])
            attachment.GetActor1Rel().SetTargets(
                [attachment_prim.root_physx_view.prim_paths[i]])
            att = PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
            att.Apply(attachment.GetPrim())
            _ = att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.02)

def delete_rigid_prim(scene):
    for i in range(scene.num_envs):
        delete_prim(f"/World/envs/env_{i}/Sphere1")
        delete_prim(f"/World/envs/env_{i}/Sphere2")

def create_rigid_prim(scene, pos1, pos2):
    """Creates a rigid prim in the scene."""
    # Create a sphere prim

    sphere_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sphere1",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            # translation=(-0.2, 0, 0),
            visible=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1e10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos1),
    )

    sphere_cfg2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sphere2",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            # translation=(0.2, 0, 0),
            visible=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1e10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos2),
    )
    #
    particle_obj = ParticleObjectCfg(
        prim_path="{ENV_REGEX_NS}/Garment",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5, 0.0001),
            visible=True,
            particle_props=sim_utils.ParticleBodyPropertiesCfg(),
            # collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.ParticleBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.2),
        ),
        init_state=ParticleObjectCfg.InitialStateCfg(),
    )

    # particle_obj = ParticleObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Garment",
    #     spawn=sim_utils.SoftUsdFileCfg(
    #         usd_path=f"/home/hengyi/GitHub/DexGarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Top074/TNLC_Top074_obj.usd",
    #         translation=(0, -1.25, 0.05),
    #         # translation=(-0.2, 0.5, 0),
    #         # scale=(0.0185, 0.0185, 0.0185),
    #         scale=(0.0085, 0.0085, 0.0085),
    #         particle_props=sim_utils.ParticleBodyPropertiesCfg(),
    #         # mass_props=sim_utils.MassPropertiesCfg(mass=69),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(),
    #         physics_material=sim_utils.ParticleBodyMaterialCfg(),
    #     ),
    # )

    scene.cfg.sphere1 = sphere_cfg
    scene.cfg.sphere2 = sphere_cfg2
    scene.cfg.particle_obj = particle_obj

    delete_prim("/World/defaultGroundPlane")
    delete_prim("/World/Light")
    delete_prim("/World/envs/env_0/Garment")
    delete_prim("/World/envs/env_0/Sphere1")
    delete_prim("/World/envs/env_0/Sphere2")
    delete_prim("/World/envs/env_0/Camera")
    return

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, save):
    """Runs the simulation loop."""
    # Extract scene entities
    garment = scene["particle_obj"]

    stage = scene.stage

    # Define goals for the arm
    garment_entity_cfg = SceneEntityCfg("particle_obj")
    garment_entity_cfg.resolve(scene)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # pos = garment.print_nodal_position_from_sim().detach()

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 1000 == 0:
            # reset time
            count = 0
            """
            Reset the robot and garment to their initial states.
            """
            "Garment"
            init_state = garment.data.nodal_pos_w.clone()
            init_vel = garment.data.nodal_vel_w.clone()
            init_vel.zero_()
            garment.write_nodal_pos_to_sim(init_state)
            garment.write_nodal_velocity_to_sim(init_vel)
            particle_mass = garment.data.particle_mass
            particle_mass = particle_mass.fill_(0.01)
            garment.write_particle_masses_to_sim(particle_mass)
            garment.reset()

            nodal_pose = garment.data.nodal_pos_w.clone()
            nodal_pose = nodal_pose.reshape(scene.num_envs, -1, 3)

            num_fold = random.choice([1, 2, 3, 4])
            num_fold = 2
            pnp_len = int(900/num_fold)
            # grasp_point1, grasp_point2 = find_grasp_points(nodal_pose)

        if count == 3:
            # picks = find_grasp_points(garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3))
            pick_pos = select_random_boundary_points_with_quat(garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3)[0], k=2, boundary_tol=0.01, min_distance=0.2)
            base = torch.tensor(scene.cloner._positions[0], device="cuda", dtype=torch.float32)
            pick_pos = pick_pos[:, :3] - base
            create_rigid_prim(scene, pick_pos[0], pick_pos[1])
            sim.reset()

        if count == 4:
            garment = scene["particle_obj"]
            sphere = scene["sphere1"]
            sphere2 = scene["sphere2"]
            sphere_entity_cfg = SceneEntityCfg("sphere1")
            sphere_entity_cfg.resolve(scene)
            sphere2_entity_cfg = SceneEntityCfg("sphere2")
            sphere2_entity_cfg.resolve(scene)
            stage = scene.stage

            init_kt_state = sphere.data.root_link_pose_w.clone()
            sphere.write_root_pose_to_sim(init_kt_state[:, :7])

            init_kt_state2 = sphere2.data.root_link_pose_w.clone()
            sphere2.write_root_pose_to_sim(init_kt_state2[:, :7])

        if count == 5:
            create_attachment_1(sphere, stage)
            create_attachment_2(sphere2, stage)

        elif 50 + pnp_len > count > 50:
            if count == 51:
                # Generate grasping trajectory
                base_poses = torch.tensor(scene.cloner._positions, device=("cuda"), dtype=torch.float32).repeat(2, 1, 1).permute(1, 0, 2)
                pick_poses = base_poses + pick_pos
                traj, _ = generate_bimanual_arc_from_given_picks(garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3), pick_poses, int(pnp_len), 0.1)

            sphere.write_root_com_pose_to_sim(traj[:, 0, count - 51])
            sphere2.write_root_com_pose_to_sim(traj[:, 1, count - 51])

        elif count > 50 + pnp_len:
            if count == 50 + pnp_len + 1:
                delete_rigid_prim(scene)

            if count == 50 + pnp_len + 15:
                init_state = garment.data.nodal_pos_w.clone()
                init_vel = garment.data.nodal_vel_w.clone()
                pick_pos = select_random_boundary_points_with_quat(
                    garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3)[0], k=2, boundary_tol=0.01,
                    min_distance=0.2)
                base = torch.tensor(scene.cloner._positions[0], device="cuda", dtype=torch.float32)
                pick_pos = pick_pos[:, :3] - base

                # prim_utils.create_prim("/World/envs/env_0/Sphere1", "Xform", translation=[0, 0, 0])
                # prim_utils.create_prim("/World/envs/env_0/Sphere2", "Xform", translation=[0, 0, 0])

                sphere_cfg1 = RigidObjectCfg(
                    prim_path="/World/envs/env_0/Sphere1",
                    spawn=sim_utils.SphereCfg(
                        radius=0.02,
                        # translation=(0.2, 0, 0),
                        visible=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1e10),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        physics_material=sim_utils.RigidBodyMaterialCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, 0]),
                )
                sphere_cfg2 = RigidObjectCfg(
                    prim_path="/World/envs/env_0/Sphere2",
                    spawn=sim_utils.SphereCfg(
                        radius=0.02,
                        # translation=(0.2, 0, 0),
                        visible=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1e10),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        physics_material=sim_utils.RigidBodyMaterialCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, 0]),
                )
                RigidObject(cfg = sphere_cfg1)
                RigidObject(cfg = sphere_cfg2)
                # create_rigid_prim(scene, pick_pos[0], pick_pos[1])
                garment = scene["particle_obj"]
                sphere = scene["sphere1"]
                sphere2 = scene["sphere2"]
                sphere_entity_cfg = SceneEntityCfg("sphere1")
                sphere_entity_cfg.resolve(scene)
                sphere2_entity_cfg = SceneEntityCfg("sphere2")
                sphere2_entity_cfg.resolve(scene)
                stage = scene.stage
                sim.reset()

            if count == 50 + pnp_len + 16:
                garment.write_nodal_pos_to_sim(init_state)
                init_vel.zero_()
                garment.write_nodal_velocity_to_sim(init_vel)

            # if count == 50 + pnp_len + 22:
            #     garment.write_nodal_pos_to_sim(init_state)
            #     init_vel.zero_()
            #     garment.write_nodal_velocity_to_sim(init_vel)

            if count == 50 + pnp_len + 17:
                picks = torch.zeros(scene.num_envs, 2, 7, device="cuda")
                picks[:, :, -1] = 1
                picks[:, :, :3] = torch.tensor(scene.cloner._positions, device="cuda", dtype=torch.float32).repeat(2, 1, 1).permute(1, 0, 2) + pick_pos[:, :3]
                # sphere.write_root_com_pose_to_sim(picks[:, 0])
                # sphere2.write_root_pose_to_sim(picks[:, 1])
                create_attachment_1(sphere, stage)
                create_attachment_2(sphere2, stage)

            # if count == 50 + pnp_len + 48:


            if count == 50 + pnp_len + 49:
                traj, _ = generate_bimanual_arc_from_given_picks(garment.data.nodal_pos_w.clone().reshape(scene.num_envs, -1, 3), picks[:, :, :3], int(pnp_len), 0.1)

            if count > 50 + pnp_len + 50:
                sphere.write_root_com_pose_to_sim(traj[:, 0, count - (50 + pnp_len + 51)])
                sphere2.write_root_com_pose_to_sim(traj[:, 1, count - (50 + pnp_len + 51)])

        if save == True:
            """Saving the soft object state"""
            if count == 0:
                dt = datetime.datetime.now()
                save_dir = "/home/ubuntu/Github/DexGarmentLab/Env_StandAlone/object_data"
                folder = save_dir + f"/{dt.year}_{dt.month}_{dt.day}"

                save_time = dt.strftime("%H_%M_%S")
                save_folder = folder + f"/{save_time}"

                object_path = save_folder + f"/object"
                action_path = save_folder + f"/action"
                mk_dir(object_path)
                mk_dir(action_path)

            if count > 49:

                if count%10 == 0:

                    now_time = datetime.datetime.now()
                    time = int(now_time.strftime("%Y%m%d%H%M%S%f"))
                    filename = save_folder + f"/object/{time}" + ".pt"
                    position = garment.data.nodal_pos_w.clone()
                    position = position.reshape(scene.num_envs, -1, 3)

                    torch.save(position, filename)

                    grasp1 = sphere.data.root_link_pose_w.clone()
                    grasp2 = sphere2.data.root_link_pose_w.clone()
                    grasp = torch.hstack([grasp1, grasp2])

                    filename2 = save_folder + f"/action/{time}" + ".pt"
                    torch.save(grasp, filename2)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers

        scene.update(sim_dt)



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
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
    run_simulator(sim, scene, save=False)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

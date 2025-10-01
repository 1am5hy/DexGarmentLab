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

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import random
import torch
import tqdm

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
from isaaclab.assets import ParticleObject, ParticleObjectCfg, RigidObjectCfg, RigidObject, DeformableObject, DeformableObjectCfg
from pxr import PhysxSchema, Usd, UsdPhysics
##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

@configclass
class ParticleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.init_state.pos = (0.6, 0, 0)
        robot.init_state.joint_pos['panda_joint1'] = 2.8
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

    if args_cli.robot == "franka_panda":
        robot2 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")
        robot2.init_state.pos = (-0.6, 0, 0)
    elif args_cli.robot == "ur10":
        robot2 = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


    particle_obj = ParticleObjectCfg(
        prim_path="{ENV_REGEX_NS}/Garment",
        spawn= sim_utils.SoftUsdFileCfg(
            usd_path=f"/home/hengyi/GitHub/DexGarmentLab/Assets/Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_Top074/TNLC_Top074_obj.usd",
            translation=(0, -0.75, 0),
            scale=(0.0085, 0.0085, 0.0085),
            particle_props=sim_utils.ParticleBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(),
            physics_material=sim_utils.ParticleBodyMaterialCfg(),
        ),
        init_state=ParticleObjectCfg.InitialStateCfg(),
    )


def find_grasp_points(poses):
    """
        Input:
            poses : The nodal poses of the particle object in all envs (shape : (num_envs, num_nodes, 3))
            max_translation : The maximum translation to apply to the sampled pose
    """
    device = poses.device
    num_envs = len(poses)

    gps_left = torch.zeros(num_envs, 7, device=device)
    gps_right = torch.zeros(num_envs, 7, device=device)

    gps_left[:, -2] = 1
    gps_right[:, -3] = 1

    for i in range(num_envs):
        # Find the minimum and maximum y-coordinates
        point_cloud = poses[i].clone()
        upper_y = torch.max(poses[i, :, 1])
        lower_y = torch.max(poses[i, :, 1]) - 0.15

        y_mask = (point_cloud[:, 1] >= lower_y) & (point_cloud[:, 1] <= upper_y)
        filtered_points = point_cloud[y_mask]

        left_x_upper = torch.max(filtered_points[:, 0]) - 0.05
        left_x_lower = torch.max(filtered_points[:, 0]) - 0.3

        right_x_upper = torch.min(filtered_points[:, 0]) + 0.3
        right_x_lower = torch.min(filtered_points[:, 0]) + 0.05


        x_mask_left = (filtered_points[:, 0] >= left_x_lower.clone().detach()) & (filtered_points[:, 0] <= left_x_upper.clone().detach())
        x_mask_right = (filtered_points[:, 0] >= right_x_lower.clone().detach()) & (filtered_points[:, 0] <= right_x_upper.clone().detach())

        grasp_points_left = filtered_points[x_mask_left].clone().detach()
        grasp_points_right = filtered_points[x_mask_right].clone().detach()

        grasp_left_idx = torch.randperm(len(grasp_points_left))[:1]
        grasp_right_idx = torch.randperm(len(grasp_points_right))[:1]

        grasp_left = grasp_points_left[grasp_left_idx]
        grasp_right = grasp_points_right[grasp_right_idx]

        gps_left[i, :3] = grasp_left
        gps_right[i, :3] = grasp_right

    return gps_left, gps_right


def generate_vertical_first_trajectory(current_pose, desired_pose, num_steps, vertical_ratio=0.5):
    """
    Generate a two-phase Cartesian trajectory:
    1. Move vertically to target Z while maintaining current XY
    2. Move horizontally to target XY position

    Args:
        current_pose: [num_envs, 7] - current position and quaternion
        desired_pose: [num_envs, 7] - desired position and quaternion
        num_steps: int - total number of timesteps
        vertical_ratio: float (0-1) - fraction of steps for vertical phase

    Returns:
        [num_envs, num_steps, 7] trajectory tensor
    """
    assert current_pose.shape == desired_pose.shape
    assert current_pose.shape[1] == 7
    assert 0 < vertical_ratio < 1

    num_envs = current_pose.shape[0]
    device = current_pose.device

    # Normalize quaternions
    current_quat = current_pose[:, 3:] / torch.norm(current_pose[:, 3:], dim=1, keepdim=True)
    desired_quat = desired_pose[:, 3:] / torch.norm(desired_pose[:, 3:], dim=1, keepdim=True)

    # Create intermediate waypoint (same XY as start, Z as target)
    waypoint = current_pose.clone()
    waypoint[:, 2] = desired_pose[:, 2]  # Target Z

    # Determine steps for each phase
    num_vertical = int(num_steps * vertical_ratio)
    num_horizontal = num_steps - num_vertical

    # Phase 1: Vertical movement (Z) with initial orientation
    t_vertical = torch.linspace(0, 1, num_vertical, device=device)

    # Position: Constant XY, linear Z
    pos_phase1 = torch.zeros(num_envs, num_vertical, 3, device=device)
    pos_phase1[:, :, :2] = current_pose[:, :2].unsqueeze(1)  # Constant XY
    pos_phase1[:, :, 2] = current_pose[:, 2].unsqueeze(1) + \
                          t_vertical.unsqueeze(0) * \
                          (waypoint[:, 2] - current_pose[:, 2]).unsqueeze(1)

    # Orientation: Constant at initial orientation during vertical phase
    quat_phase1 = current_quat.unsqueeze(1).expand(-1, num_vertical, -1)

    # Phase 2: Horizontal movement (XY) with orientation interpolation
    t_horizontal = torch.linspace(0, 1, num_horizontal, device=device)

    # Position: Linear XY, constant Z
    pos_phase2 = torch.zeros(num_envs, num_horizontal, 3, device=device)
    pos_phase2[:, :, :2] = waypoint[:, :2].unsqueeze(1) + \
                           t_horizontal.unsqueeze(0).unsqueeze(-1) * \
                           (desired_pose[:, :2] - waypoint[:, :2]).unsqueeze(1)
    pos_phase2[:, :, 2] = desired_pose[:, 2].unsqueeze(1)  # Constant Z

    # Orientation: Full slerp during horizontal phase
    quat_phase2 = slerp_quaternions(current_quat, desired_quat, t_horizontal)

    # Combine phases
    positions = torch.cat([pos_phase1, pos_phase2], dim=1)
    quaternions = torch.cat([quat_phase1, quat_phase2], dim=1)

    trajectory = torch.cat([positions, quaternions], dim=2)
    return trajectory


def generate_cartesian_trajectory(current_pose, desired_pose, num_steps):
    """
    Generate Cartesian trajectories for vectorized environments.

    Args:
        current_pose: torch.Tensor of shape [num_envs, 7] [x, y, z, qx, qy, qz, qw]
        desired_pose: torch.Tensor of shape [num_envs, 7] [x, y, z, qx, qy, qz, qw]
        num_steps: int - number of timesteps in the trajectory

    Returns:
        torch.Tensor of shape [num_envs, num_steps, 7] containing the trajectories
    """
    assert current_pose.shape == desired_pose.shape, "Input shapes must match"
    assert current_pose.shape[1] == 7, "Last dimension must be 7 (pos + quat)"

    num_envs = current_pose.shape[0]

    # Normalize input quaternions
    current_quat = current_pose[:, 3:] / torch.norm(current_pose[:, 3:], dim=1, keepdim=True)
    desired_quat = desired_pose[:, 3:] / torch.norm(desired_pose[:, 3:], dim=1, keepdim=True)

    # Create time steps from 0 to 1 [num_steps]
    t = torch.linspace(0, 1, num_steps, device=current_pose.device)

    # Linear interpolation for position [num_envs, num_steps, 3]
    positions = current_pose[:, :3].unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * \
                (desired_pose[:, :3] - current_pose[:, :3]).unsqueeze(1)

    # Spherical linear interpolation for orientation [num_envs, num_steps, 4]
    # Compute dot product between quaternions [num_envs]
    dot = (current_quat * desired_quat).sum(dim=1)

    # Handle cases where we need to flip quaternions
    flip_mask = dot < 0.0
    desired_quat = desired_quat.clone()  # Avoid modifying input
    desired_quat[flip_mask] = -desired_quat[flip_mask]
    dot[flip_mask] = -dot[flip_mask]

    DOT_THRESHOLD = 0.9995
    # Initialize output quaternions
    quaternions = torch.zeros(num_envs, num_steps, 4, device=current_pose.device)

    # Case 1: Quaternions are very close - linear interpolation
    close_mask = dot > DOT_THRESHOLD
    if close_mask.any():
        q0 = current_quat[close_mask]  # [num_close_envs, 4]
        q1 = desired_quat[close_mask]  # [num_close_envs, 4]

        # Linear interpolation [num_close_envs, num_steps, 4]
        interp = q0.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * (q1 - q0).unsqueeze(1)
        quaternions[close_mask] = interp / torch.norm(interp, dim=2, keepdim=True)

    # Case 2: Need proper slerp
    far_mask = ~close_mask
    if far_mask.any():
        q0 = current_quat[far_mask]  # [num_far_envs, 4]
        q1 = desired_quat[far_mask]  # [num_far_envs, 4]
        dot_far = dot[far_mask]  # [num_far_envs]

        theta_0 = torch.acos(dot_far.clamp(-1.0, 1.0))  # [num_far_envs]
        sin_theta_0 = torch.sin(theta_0)  # [num_far_envs]

        theta = theta_0.unsqueeze(1) * t.unsqueeze(0)  # [num_far_envs, num_steps]
        sin_theta = torch.sin(theta)  # [num_far_envs, num_steps]

        s0 = torch.cos(theta) - dot_far.unsqueeze(1) * sin_theta / sin_theta_0.unsqueeze(1)
        s1 = sin_theta / sin_theta_0.unsqueeze(1)

        # Broadcasted slerp [num_far_envs, num_steps, 4]
        quaternions[far_mask] = (s0.unsqueeze(-1) * q0.unsqueeze(1) +
                                 s1.unsqueeze(-1) * q1.unsqueeze(1))

    # Combine positions and quaternions [num_envs, num_steps, 7]
    trajectory = torch.cat([positions, quaternions], dim=2)

    return trajectory

def generate_two_phase_trajectory(current_pose, desired_pose, num_steps, horizontal_ratio=0.7):
    """
    Generate a two-phase Cartesian trajectory:
    1. Move horizontally to target XY while maintaining current height
    2. Move vertically down to target Z

    Args:
        current_pose: [num_envs, 7] - current position and quaternion
        desired_pose: [num_envs, 7] - desired position and quaternion
        num_steps: int - total number of timesteps
        horizontal_ratio: float (0-1) - fraction of steps for horizontal phase

    Returns:
        [num_envs, num_steps, 7] trajectory tensor
    """
    assert current_pose.shape == desired_pose.shape
    assert current_pose.shape[1] == 7
    assert 0 < horizontal_ratio < 1

    num_envs = current_pose.shape[0]
    device = current_pose.device

    # Normalize quaternions
    current_quat = current_pose[:, 3:] / torch.norm(current_pose[:, 3:], dim=1, keepdim=True)
    desired_quat = desired_pose[:, 3:] / torch.norm(desired_pose[:, 3:], dim=1, keepdim=True)

    # Create intermediate waypoint (same Z as start, XY as target)
    waypoint = desired_pose.clone()
    waypoint[:, 2] = current_pose[:, 2]  # Maintain current Z

    # Determine steps for each phase
    num_horizontal = int(num_steps * horizontal_ratio)
    num_vertical = num_steps - num_horizontal

    # Phase 1: Horizontal movement (XY) with orientation interpolation
    t_horizontal = torch.linspace(0, 1, num_horizontal, device=device)

    # Position: Linear XY, constant Z
    pos_phase1 = torch.zeros(num_envs, num_horizontal, 3, device=device)
    pos_phase1[:, :, :2] = current_pose[:, :2].unsqueeze(1) + \
                           t_horizontal.unsqueeze(0).unsqueeze(-1) * \
                           (waypoint[:, :2] - current_pose[:, :2]).unsqueeze(1)
    pos_phase1[:, :, 2] = current_pose[:, 2].unsqueeze(1)  # Constant Z

    # Orientation: Full slerp during horizontal phase
    quat_phase1 = slerp_quaternions(current_quat, desired_quat, t_horizontal)

    # Phase 2: Vertical movement (Z) with constant orientation
    t_vertical = torch.linspace(0, 1, num_vertical, device=device)

    # Position: Constant XY, linear Z
    pos_phase2 = torch.zeros(num_envs, num_vertical, 3, device=device)
    pos_phase2[:, :, :2] = desired_pose[:, :2].unsqueeze(1)  # Constant XY
    pos_phase2[:, :, 2] = waypoint[:, 2].unsqueeze(1) + \
                          t_vertical.unsqueeze(0) * \
                          (desired_pose[:, 2] - waypoint[:, 2]).unsqueeze(1)

    # Orientation: Constant at target orientation
    quat_phase2 = desired_quat.unsqueeze(1).expand(-1, num_vertical, -1)

    # Combine phases
    positions = torch.cat([pos_phase1, pos_phase2], dim=1)
    quaternions = torch.cat([quat_phase1, quat_phase2], dim=1)

    trajectory = torch.cat([positions, quaternions], dim=2)
    return trajectory


def slerp_quaternions(q0, q1, t):
    """Batch spherical linear interpolation for quaternions"""
    # q0: [num_envs, 4], q1: [num_envs, 4], t: [num_steps]
    dot = (q0 * q1).sum(dim=1)  # [num_envs]

    # Flip quaternions where needed
    flip_mask = dot < 0.0
    q1 = q1.clone()
    q1[flip_mask] = -q1[flip_mask]
    dot[flip_mask] = -dot[flip_mask]

    DOT_THRESHOLD = 0.9995
    quaternions = torch.zeros(q0.shape[0], len(t), 4, device=q0.device)

    # Case 1: Nearly parallel - linear interpolation
    close_mask = dot > DOT_THRESHOLD
    if close_mask.any():
        interp = q0[close_mask].unsqueeze(1) + \
                 t.unsqueeze(0).unsqueeze(-1) * \
                 (q1[close_mask] - q0[close_mask]).unsqueeze(1)
        quaternions[close_mask] = interp / torch.norm(interp, dim=2, keepdim=True)

    # Case 2: Proper slerp
    far_mask = ~close_mask
    if far_mask.any():
        q0_far = q0[far_mask]
        q1_far = q1[far_mask]
        dot_far = dot[far_mask]

        theta_0 = torch.acos(dot_far.clamp(-1.0, 1.0))
        sin_theta_0 = torch.sin(theta_0)

        theta = theta_0.unsqueeze(1) * t.unsqueeze(0)
        sin_theta = torch.sin(theta)

        s0 = torch.cos(theta) - dot_far.unsqueeze(1) * sin_theta / sin_theta_0.unsqueeze(1)
        s1 = sin_theta / sin_theta_0.unsqueeze(1)

        quaternions[far_mask] = (s0.unsqueeze(-1) * q0_far.unsqueeze(1) +
                                 s1.unsqueeze(-1) * q1_far.unsqueeze(1))

    return quaternions

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    robot2 = scene["robot2"]
    garment = scene["particle_obj"]

        # print(garment)
        # print(len(garment))

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    diff_ik_cfg2 = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller2 = DifferentialIKController(diff_ik_cfg2, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    ee_marker2 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current2"))
    goal_marker1 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    goal_marker2 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal2"))

    # Define goals for the arm
    ee_goals = [
        [-0.25, 0, 0.7, 0.0, 0.0, 1.0, 0.0],
    ]
    ee_goals2 = [
        [0.25, 0, 0.7, 0.0, 1.0, 0.0, 0.0],
    ]

    ee_goals = torch.tensor(ee_goals, device=sim.device)
    ee_goals2 = torch.tensor(ee_goals2, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    robot_origin = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    robot_origin2 = torch.zeros(scene.num_envs, diff_ik_controller2.action_dim, device=robot2.device)
    robot_origin[:] = ee_goals[current_goal_idx]
    robot_origin2[:] = ee_goals2[current_goal_idx]


    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg2 = SceneEntityCfg("robot2", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_fingers = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"], body_names=["panda_hand"])
        robot_entity_fingers2 = SceneEntityCfg("robot2", joint_names=["panda_finger_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
        robot_entity_cfg2 = SceneEntityCfg("robot2", joint_names=[".*"], body_names=["ee_link"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

    garment_entity_cfg = SceneEntityCfg("particle_obj")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    robot_entity_cfg2.resolve(scene)
    robot_entity_fingers.resolve(scene)
    robot_entity_fingers2.resolve(scene)
    garment_entity_cfg.resolve(scene)


    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        ee_jacobi_idx2 = robot_entity_cfg2.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
        ee_jacobi_idx2 = robot_entity_cfg2.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    grasping_traj_len = 150
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 600 == 0:
            # reset time
            count = 0

            """
            Reset the robot and garment to their initial states.
            """

            "Garment"
            init_state = garment.data.default_nodal_pos.clone()
            garment.write_nodal_state_to_sim(init_state)
            particle_mass = garment.data.particle_mass
            particle_mass = particle_mass.fill_(0.01)
            garment.write_particle_masses_to_sim(particle_mass)
            garment.reset()

            "Robot 1"
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # joint_pose_limits = torch.tensor([0, 0.15], device=robot.device)
            # joint_vel_limits = torch.tensor([0, 5.0], device=robot.device)
            # robot.write_joint_position_limit_to_sim(joint_pose_limits,joint_ids=robot_entity_fingers.joint_ids)
            # robot.write_joint_velocity_limit_to_sim(joint_vel_limits,joint_ids=robot_entity_fingers.joint_ids)
            robot.reset()
            # reset actions
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            ik_commands = robot_origin.clone()
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)

            "Robot 2"
            # reset joint state
            joint_pos2 = robot2.data.default_joint_pos.clone()
            joint_vel2 = robot2.data.default_joint_vel.clone()
            robot2.write_joint_state_to_sim(joint_pos2, joint_vel2)
            # robot2.write_joint_position_limit_to_sim(joint_pose_limits,joint_ids=robot_entity_fingers2.joint_ids)
            # robot2.write_joint_velocity_limit_to_sim(joint_vel_limits, joint_ids=robot_entity_fingers2.joint_ids)
            robot2.reset()
            # reset actions
            joint_pos_des2 = joint_pos2[:, robot_entity_cfg2.joint_ids].clone()
            # reset controller
            ik_commands2 = robot_origin2.clone()
            diff_ik_controller2.reset()
            diff_ik_controller2.set_command(ik_commands2)

        elif count == 150:
            "Garment"
            nodal_pose = garment.data.nodal_pos_w.clone()
            nodal_pose = nodal_pose.reshape(scene.num_envs, -1, 3)
            grasp_point1, grasp_point2 = find_grasp_points(nodal_pose)
            goal_marker1.visualize(grasp_point1[:, 0:3], grasp_point1[:, 3:7])
            goal_marker2.visualize(grasp_point2[:, 0:3], grasp_point2[:, 3:7])

            "Robot 1 Grasp Trajectory"
            cur_pos = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], cur_pos[:, 0:3], cur_pos[:, 3:7]
            )
            ee_pos_b = torch.cat((ee_pos_b, ee_quat_b), dim=1)
            ee_pos_b_des, ee_quat_b_des = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], grasp_point1[:, 0:3], grasp_point1[:, 3:7]
            )
            ee_pos_b_des = torch.cat((ee_pos_b_des, ee_quat_b_des), dim=1)

            grasp1_traj = generate_two_phase_trajectory(ee_pos_b, ee_pos_b_des, grasping_traj_len)

            "Robot 2 Grasp Trajectory"
            cur_pos2 = robot2.data.body_pose_w[:, robot_entity_cfg2.body_ids[0]]
            root_pose_w2 = robot2.data.root_pose_w
            ee_pos_b2, ee_quat_b2 = subtract_frame_transforms(
                root_pose_w2[:, 0:3], root_pose_w2[:, 3:7], cur_pos2[:, 0:3], cur_pos2[:, 3:7]
            )
            ee_pos_b2 = torch.cat((ee_pos_b2, ee_quat_b2), dim=1)

            ee_pos_b_des2, ee_quat_b_des2 = subtract_frame_transforms(
                root_pose_w2[:, 0:3], root_pose_w2[:, 3:7], grasp_point2[:, 0:3], grasp_point2[:, 3:7]
            )
            ee_pos_b_des2 = torch.cat((ee_pos_b_des2, ee_quat_b_des2), dim=1)

            grasp2_traj = generate_two_phase_trajectory(ee_pos_b2, ee_pos_b_des2, grasping_traj_len)

        else:
            """
            Acquire robot joint commands using differential IK controller.
            """

            if 150 + grasping_traj_len > count > 150:
                "Robot 1"
                ik_commands[:] = grasp1_traj[:, count - 151, :]
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)

                "Robot 2"
                ik_commands2[:] = grasp2_traj[:, count - 151, :]
                diff_ik_controller2.reset()
                diff_ik_controller2.set_command(ik_commands2)

            elif count == 150 + grasping_traj_len + 1:
                # Reset the goal to the initial goal
                cur_pos = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
                root_pose_w = robot.data.root_pose_w
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], cur_pos[:, 0:3], cur_pos[:, 3:7]
                )
                ee_pos_b = torch.cat((ee_pos_b, ee_quat_b), dim=1)
                ee_pos_b_des = robot_origin.clone()
                grasp1_traj = generate_vertical_first_trajectory(ee_pos_b, ee_pos_b_des, grasping_traj_len)
                # print()

                cur_pos2 = robot2.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
                root_pose_w2 = robot2.data.root_pose_w
                ee_pos_b2, ee_quat_b2 = subtract_frame_transforms(
                    root_pose_w2[:, 0:3], root_pose_w2[:, 3:7], cur_pos2[:, 0:3], cur_pos2[:, 3:7]
                )
                ee_pos_b2 = torch.cat((ee_pos_b2, ee_quat_b2), dim=1)
                ee_pos_b_des2 = robot_origin2.clone()
                grasp2_traj = generate_vertical_first_trajectory(ee_pos_b2, ee_pos_b_des2, grasping_traj_len)

            elif 150 + grasping_traj_len + grasping_traj_len + 1 > count > 150 + grasping_traj_len + 1:
                "Robot 1"
                ik_commands[:] = grasp1_traj[:, count - 151 - grasping_traj_len - 1, :]
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)

                "Robot 2"
                ik_commands2[:] = grasp2_traj[:, count - 151 - grasping_traj_len - 1, :]
                diff_ik_controller2.reset()
                diff_ik_controller2.set_command(ik_commands2)

            elif count > 150 + grasping_traj_len + 1 + grasping_traj_len:
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)

                "Robot 2"
                diff_ik_controller2.reset()
                diff_ik_controller2.set_command(ik_commands2)

            "Robot 1"
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            # This computes the end effector pose with relation to robot root pose
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            "Robot 2"
            jacobian2 = robot2.root_physx_view.get_jacobians()[:, ee_jacobi_idx2, :, robot_entity_cfg2.joint_ids]
            ee_pose_w2 = robot2.data.body_pose_w[:, robot_entity_cfg2.body_ids[0]]
            root_pose_w2 = robot2.data.root_pose_w
            joint_pos2 = robot2.data.joint_pos[:, robot_entity_cfg2.joint_ids]
            # compute frame in root frame
            ee_pos_b2, ee_quat_b2 = subtract_frame_transforms(
                root_pose_w2[:, 0:3], root_pose_w2[:, 3:7], ee_pose_w2[:, 0:3], ee_pose_w2[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des2 = diff_ik_controller2.compute(ee_pos_b2, ee_quat_b2, jacobian2, joint_pos2)


        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot2.set_joint_position_target(joint_pos_des2, joint_ids=robot_entity_cfg2.joint_ids)
        # Gripper commands - close
        if count < 300:
            gripper_control(robot_entity_fingers, robot, garment, action="Open")
            gripper_control(robot_entity_fingers2, robot2, garment, action="Open")
        elif count == 301:
            gripper_control(robot_entity_fingers, robot, garment, action="Close")
            gripper_control(robot_entity_fingers2, robot2, garment, action="Close")

        #

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_pose_w2 = robot2.data.body_state_w[:, robot_entity_cfg2.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        ee_marker2.visualize(ee_pose_w2[:, 0:3], ee_pose_w2[:, 3:7])


        goal_marker1.visualize(ik_commands[:, 0:3] + robot.data.root_pose_w[:, 0:3], ik_commands[:, 3:7])
        goal_marker2.visualize(ik_commands2[:, 0:3] + robot2.data.root_pose_w[:, 0:3], ik_commands2[:, 3:7])


def gripper_control(robot_entity_fingers, robot, garment, action):
    """Control the gripper of the robot."""
    # This function can be used to control the gripper of the robot.
    robot_fingers_pos = robot.data.joint_pos[:, robot_entity_fingers.joint_ids]
    robot_fingers_vel = robot.data.joint_vel[:, robot_entity_fingers.joint_ids]
    print(robot_fingers_pos)
    gripper_command_target = torch.zeros(args_cli.num_envs, 2, device=robot.device)

    if action == "Close":
        gripper_command_target[:, 0] = robot_fingers_pos[:, 0] - 0.035
        gripper_command_target[:, 1] = robot_fingers_pos[:, 1] - 0.035
        robot.set_joint_position_target(gripper_command_target, joint_ids=robot_entity_fingers.joint_ids)

        # stage = stage_utils.get_current_stage()
        # attachment_path = robot.cfg.prim_path.replace("env_.*/Robot", "env_0/Attachment/attach")
        # attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
        # attachment.GetActor0Rel().SetTargets([robot.cfg.prim_path.replace("env_.*/Robot", "env_0/Garment/geometry/mesh")])
        # attachment.GetActor1Rel().SetTargets([robot.cfg.prim_path.replace("env_.*/Robot", "env_0/Robot/panda_left_finger")])
        # att = PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
        # att.Apply(attachment.GetPrim())
        # _ = att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.02)

    elif action == "Open":

        gripper_command_target[:, 0] = robot_fingers_pos[:, 0] + 2
        gripper_command_target[:, 1] = robot_fingers_pos[:, 1] + 2
        robot.set_joint_position_target(gripper_command_target, joint_ids=robot_entity_fingers.joint_ids)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    # sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    sim.set_camera_view([1.25, 1.25, 1.25], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = ParticleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

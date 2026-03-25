import torch
import os


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


def generate_vertical_first_trajectory(current_pose, desired_pose, num_steps, vertical_ratio=0.80):
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

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class BimanualQuaternionTrajectory:
    def __init__(self, num_timesteps, via_point_spacing, initial_pose_left, initial_pose_right,bbox_min, bbox_max,
                 device='cuda'):
        """
        Generates bimanual trajectories with quaternion orientations.

        Parameters:
        - num_timesteps: Total number of timesteps
        - via_point_spacing: Spacing between control points
        - initial_pose_left: {'position': [x,y,z], 'orientation': [qx,qy,qz,qw]}
        - initial_pose_right: Same format as left
        - device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_timesteps = num_timesteps
        self.via_point_spacing = via_point_spacing
        self.fixed_distance = torch.norm(initial_pose_left[:3] - initial_pose_right[:3])

        # Convert initial poses to tensors with proper shapes
        self.initial_left_pos = initial_pose_left[:3].reshape(1, 3)
        self.initial_right_pos = initial_pose_right[:3].reshape(1, 3)
        self.initial_left_quat = initial_pose_left[3:].reshape(1, 4)
        self.initial_right_quat = initial_pose_right[3:].reshape(1, 4)

        self.bbox_min = torch.tensor(bbox_min, device=self.device)
        self.bbox_max = torch.tensor(bbox_max, device=self.device)

        # Generate full trajectory
        self._generate_full_trajectory()

    # def _initialize_poses(self, initial_left, initial_right):
    #     """Initialize position and orientation tensors."""
    #     # Positions
    #     self.initial_left_pos = torch.tensor(initial_left['position'], dtype=torch.float32, device=self.device).view(1,
    #                                                                                                                  3)
    #     self.initial_right_pos = torch.tensor(initial_right['position'], dtype=torch.float32, device=self.device).view(
    #         1, 3)
    #
    #     # Quaternion orientations (normalized)
    #     self.initial_left_quat = torch.tensor(initial_left['orientation'], dtype=torch.float32,
    #                                           device=self.device).view(1, 4)
    #     self.initial_left_quat = self.initial_left_quat / torch.norm(self.initial_left_quat, dim=1, keepdim=True)
    #
    #     self.initial_right_quat = torch.tensor(initial_right['orientation'], dtype=torch.float32,
    #                                            device=self.device).view(1, 4)
    #     self.initial_right_quat = self.initial_right_quat / torch.norm(self.initial_right_quat, dim=1, keepdim=True)

    def _generate_full_trajectory(self):
        """Generate complete trajectory with quaternion orientations."""
        # Generate position trajectory
        self._generate_position_trajectory()

        # Generate orientation trajectory
        self._generate_orientation_trajectory()

    def _generate_position_trajectory(self):
        """Generate position trajectories for both arms."""
        num_via_points = (self.num_timesteps // self.via_point_spacing) + 2

        # Centroid trajectory
        centroid_via_points = []
        current_pos = (self.initial_left_pos + self.initial_right_pos) / 2
        centroid_via_points.append(current_pos)

        direction = torch.randn(1, 3, device=self.device)
        direction = direction / torch.norm(direction, dim=1, keepdim=True)

        for _ in range(1, num_via_points):
            direction_change = torch.randn(1, 3, device=self.device) * 0.3
            direction = (direction + direction_change).renorm(p=2, dim=1, maxnorm=1)
            current_pos = current_pos + direction * (self.fixed_distance * 0.5)

            # Apply bounding box constraints if they exist
            if self.bbox_min is not None and self.bbox_max is not None:
                current_pos = torch.max(current_pos, self.bbox_min)
                current_pos = torch.min(current_pos, self.bbox_max)

            centroid_via_points.append(current_pos)

        self.centroid_via_points = torch.cat(centroid_via_points, dim=0)

        # Relative vectors
        initial_rel_vec = self.initial_right_pos - (self.initial_left_pos + self.initial_right_pos) / 2
        initial_rel_vec = initial_rel_vec * (self.fixed_distance / 2) / torch.norm(initial_rel_vec, dim=1, keepdim=True)

        rel_vectors = [initial_rel_vec]
        current_rel = initial_rel_vec

        rot_axis = torch.randn(1, 3, device=self.device)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=1, keepdim=True)
        rot_speed = torch.rand(1, 1, device=self.device) * 0.1 - 0.05

        for _ in range(1, num_via_points):
            rot_axis_change = torch.randn(1, 3, device=self.device) * 0.2
            rot_axis = (rot_axis + rot_axis_change).renorm(p=2, dim=1, maxnorm=1)

            angle = rot_speed * self.via_point_spacing
            rot_matrix = self._axis_angle_to_matrix(rot_axis, angle)
            current_rel = torch.bmm(rot_matrix, current_rel.unsqueeze(-1)).squeeze(-1)
            current_rel = current_rel * (self.fixed_distance / 2) / torch.norm(current_rel, dim=1, keepdim=True)
            rel_vectors.append(current_rel)

        self.rel_via_points = torch.cat(rel_vectors, dim=0)

        # Interpolate positions
        self._interpolate_positions()

    def _interpolate_positions(self):
        """Interpolate between via points for smooth position trajectories."""
        via_indices = torch.arange(0, self.num_timesteps, self.via_point_spacing, device=self.device)
        if via_indices[-1] != self.num_timesteps - 1:
            via_indices = torch.cat([via_indices, torch.tensor([self.num_timesteps - 1], device=self.device)])

        all_indices = torch.arange(self.num_timesteps, device=self.device)
        upper_idx = torch.searchsorted(via_indices, all_indices, right=True)
        upper_idx = torch.clamp(upper_idx, 1, len(via_indices) - 1)
        lower_idx = upper_idx - 1

        t0 = via_indices[lower_idx]
        t1 = via_indices[upper_idx]
        alpha = (all_indices - t0) / (t1 - t0).clamp(min=1e-6)
        alpha = alpha.unsqueeze(1)

        # Interpolate centroid
        p0_centroid = self.centroid_via_points[lower_idx]
        p1_centroid = self.centroid_via_points[upper_idx]
        self.centroid_traj = p0_centroid + alpha * (p1_centroid - p0_centroid)

        # Interpolate relative vectors
        p0_rel = self.rel_via_points[lower_idx]
        p1_rel = self.rel_via_points[upper_idx]
        self.rel_traj = p0_rel + alpha * (p1_rel - p0_rel)
        self.rel_traj = self.rel_traj * (self.fixed_distance / 2) / torch.norm(self.rel_traj, dim=1, keepdim=True)

        # Final positions
        self.left_pos = self.centroid_traj - self.rel_traj
        self.right_pos = self.centroid_traj + self.rel_traj

    def _generate_orientation_trajectory(self):
        """Generate orientation trajectories using quaternion SLERP."""
        num_via_points = (self.num_timesteps // self.via_point_spacing) + 2

        # Left arm orientation via points
        left_quat_via = [self.initial_left_quat]
        current_left_quat = self.initial_left_quat

        # Right arm orientation via points
        right_quat_via = [self.initial_right_quat]
        current_right_quat = self.initial_right_quat

        # Generate random rotations for both arms
        for _ in range(1, num_via_points):
            # Left arm rotation change
            left_axis = torch.randn(1, 3, device=self.device)
            left_axis = left_axis / torch.norm(left_axis, dim=1, keepdim=True)
            left_angle = torch.empty(1, 1, device=self.device).uniform_(-0.2, 0.2)
            left_rot_quat = self._axis_angle_to_quaternion(left_axis, left_angle)
            current_left_quat = self._quaternion_multiply(left_rot_quat, current_left_quat)
            left_quat_via.append(current_left_quat)

            # Right arm rotation change (correlated with left)
            right_axis = left_axis + torch.randn(1, 3, device=self.device) * 0.1
            right_axis = right_axis / torch.norm(right_axis, dim=1, keepdim=True)
            right_angle = left_angle * torch.empty(1, 1, device=self.device).uniform_(0.8, 1.2)
            right_rot_quat = self._axis_angle_to_quaternion(right_axis, right_angle)
            current_right_quat = self._quaternion_multiply(right_rot_quat, current_right_quat)
            right_quat_via.append(current_right_quat)

        self.left_quat_via = torch.cat(left_quat_via, dim=0)
        self.right_quat_via = torch.cat(right_quat_via, dim=0)

        # Interpolate orientations using SLERP
        self._interpolate_quaternions()

    def _interpolate_quaternions(self):
        """Interpolate quaternions using spherical linear interpolation."""
        via_indices = torch.arange(0, self.num_timesteps, self.via_point_spacing, device=self.device)
        if via_indices[-1] != self.num_timesteps - 1:
            via_indices = torch.cat([via_indices, torch.tensor([self.num_timesteps - 1], device=self.device)])

        all_indices = torch.arange(self.num_timesteps, device=self.device)
        upper_idx = torch.searchsorted(via_indices, all_indices, right=True)
        upper_idx = torch.clamp(upper_idx, 1, len(via_indices) - 1)
        lower_idx = upper_idx - 1

        t0 = via_indices[lower_idx]
        t1 = via_indices[upper_idx]
        alpha = (all_indices - t0) / (t1 - t0).clamp(min=1e-6)

        # Interpolate left quaternions
        q0_left = self.left_quat_via[lower_idx]
        q1_left = self.left_quat_via[upper_idx]
        self.left_quat = self._quaternion_slerp(q0_left, q1_left, alpha)

        # Interpolate right quaternions
        q0_right = self.right_quat_via[lower_idx]
        q1_right = self.right_quat_via[upper_idx]
        self.right_quat = self._quaternion_slerp(q0_right, q1_right, alpha)

    def _axis_angle_to_quaternion(self, axis, angle):
        """Convert axis-angle to quaternion."""
        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        half_angle = angle / 2
        w = torch.cos(half_angle)
        xyz = torch.sin(half_angle) * axis
        return torch.cat([xyz, w], dim=1)

    def _axis_angle_to_matrix(self, axis, angle):
        """Convert axis-angle to rotation matrix."""
        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        a = torch.cos(angle / 2)
        b = -axis[:, 0] * torch.sin(angle / 2)
        c = -axis[:, 1] * torch.sin(angle / 2)
        d = -axis[:, 2] * torch.sin(angle / 2)

        rot = torch.zeros(axis.size(0), 3, 3, device=self.device)

        rot[:, 0, 0] = a * a + b * b - c * c - d * d
        rot[:, 0, 1] = 2 * (b * c - a * d)
        rot[:, 0, 2] = 2 * (b * d + a * c)

        rot[:, 1, 0] = 2 * (b * c + a * d)
        rot[:, 1, 1] = a * a + c * c - b * b - d * d
        rot[:, 1, 2] = 2 * (c * d - a * b)

        rot[:, 2, 0] = 2 * (b * d - a * c)
        rot[:, 2, 1] = 2 * (c * d + a * b)
        rot[:, 2, 2] = a * a + d * d - b * b - c * c

        return rot

    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1[:, 3], q1[:, 0], q1[:, 1], q1[:, 2]
        w2, x2, y2, z2 = q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([x, y, z, w], dim=1)

    def _quaternion_slerp(self, q0, q1, t):
        """Spherical linear interpolation between quaternions."""
        # Ensure quaternions are normalized
        q0 = q0 / torch.norm(q0, dim=1, keepdim=True)
        q1 = q1 / torch.norm(q1, dim=1, keepdim=True)

        # Compute dot product (cosine of angle between quaternions)
        dot = (q0 * q1).sum(dim=1)

        # If orientations are very similar, use linear interpolation
        linear_mask = dot.abs() > 0.9995
        result = torch.zeros_like(q0)

        # Linear interpolation cases
        if linear_mask.any():
            result[linear_mask] = q0[linear_mask] + t[linear_mask].unsqueeze(1) * (q1[linear_mask] - q0[linear_mask])
            result[linear_mask] = result[linear_mask] / torch.norm(result[linear_mask], dim=1, keepdim=True)

        # SLERP cases
        if not linear_mask.all():
            # Compute angle and perpendicular component
            theta = torch.acos(dot[~linear_mask].abs())
            sin_theta = torch.sin(theta)

            # Compute interpolation weights
            w0 = torch.sin((1 - t[~linear_mask]) * theta) / sin_theta
            w1 = torch.sin(t[~linear_mask] * theta) / sin_theta

            # Flip one quaternion if dot product is negative
            q1_adj = q1[~linear_mask] * torch.sign(dot[~linear_mask]).unsqueeze(1)

            # Interpolate
            result[~linear_mask] = w0.unsqueeze(1) * q0[~linear_mask] + w1.unsqueeze(1) * q1_adj

        return result

    def get_left_poses(self):
        """Get left arm poses (position + quaternion)."""
        return self.left_pos, self.left_quat

    def get_right_poses(self):
        """Get right arm poses (position + quaternion)."""
        return self.right_pos, self.right_quat

    def visualize(self, sample_every=10):
        """Visualize the full 6D trajectory."""
        indices = torch.arange(0, self.num_timesteps, sample_every, device='cpu')

        # Positions
        left_pos = self.left_pos[indices].cpu().numpy()
        right_pos = self.right_pos[indices].cpu().numpy()

        # Quaternion orientations
        left_quat = self.left_quat[indices].cpu().numpy()
        right_quat = self.right_quat[indices].cpu().numpy()

        # Convert quaternions to rotation matrices for visualization
        left_rot = np.array([self._quaternion_to_matrix(q) for q in left_quat])
        right_rot = np.array([self._quaternion_to_matrix(q) for q in right_quat])

        # Create figure
        fig = plt.figure(figsize=(18, 12))

        # 3D Trajectory with Orientation Frames
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(left_pos[:, 0], left_pos[:, 1], left_pos[:, 2], 'b-', label='Left')
        ax1.plot(right_pos[:, 0], right_pos[:, 1], right_pos[:, 2], 'r-', label='Right')

        # Draw orientation frames at sample points
        for i in range(0, len(left_pos), max(1, len(left_pos) // 5)):
            # Left arm frame
            x_dir = left_rot[i, :, 0] * 0.1
            y_dir = left_rot[i, :, 1] * 0.1
            z_dir = left_rot[i, :, 2] * 0.1
            ax1.quiver(left_pos[i, 0], left_pos[i, 1], left_pos[i, 2],
                       x_dir[0], x_dir[1], x_dir[2], color='red')
            ax1.quiver(left_pos[i, 0], left_pos[i, 1], left_pos[i, 2],
                       y_dir[0], y_dir[1], y_dir[2], color='green')
            ax1.quiver(left_pos[i, 0], left_pos[i, 1], left_pos[i, 2],
                       z_dir[0], z_dir[1], z_dir[2], color='blue')

            # Right arm frame
            x_dir = right_rot[i, :, 0] * 0.1
            y_dir = right_rot[i, :, 1] * 0.1
            z_dir = right_rot[i, :, 2] * 0.1
            ax1.quiver(right_pos[i, 0], right_pos[i, 1], right_pos[i, 2],
                       x_dir[0], x_dir[1], x_dir[2], color='red')
            ax1.quiver(right_pos[i, 0], right_pos[i, 1], right_pos[i, 2],
                       y_dir[0], y_dir[1], y_dir[2], color='green')
            ax1.quiver(right_pos[i, 0], right_pos[i, 1], right_pos[i, 2],
                       z_dir[0], z_dir[1], z_dir[2], color='blue')

        ax1.set_title('3D Trajectory with Orientation Frames')
        ax1.legend()

        # Position plots
        times = indices.cpu().numpy()
        ax2 = fig.add_subplot(232)
        ax2.plot(times, left_pos[:, 0], 'r-', label='X')
        ax2.plot(times, left_pos[:, 1], 'g-', label='Y')
        ax2.plot(times, left_pos[:, 2], 'b-', label='Z')
        ax2.set_title('Left Arm Position')
        ax2.legend()

        ax3 = fig.add_subplot(233)
        ax3.plot(times, right_pos[:, 0], 'r-', label='X')
        ax3.plot(times, right_pos[:, 1], 'g-', label='Y')
        ax3.plot(times, right_pos[:, 2], 'b-', label='Z')
        ax3.set_title('Right Arm Position')
        ax3.legend()

        # Quaternion components plots
        ax4 = fig.add_subplot(234)
        ax4.plot(times, left_quat[:, 0], 'r-', label='qx')
        ax4.plot(times, left_quat[:, 1], 'g-', label='qy')
        ax4.plot(times, left_quat[:, 2], 'b-', label='qz')
        ax4.plot(times, left_quat[:, 3], 'k-', label='qw')
        ax4.set_title('Left Arm Quaternion')
        ax4.legend()

        ax5 = fig.add_subplot(235)
        ax5.plot(times, right_quat[:, 0], 'r-', label='qx')
        ax5.plot(times, right_quat[:, 1], 'g-', label='qy')
        ax5.plot(times, right_quat[:, 2], 'b-', label='qz')
        ax5.plot(times, right_quat[:, 3], 'k-', label='qw')
        ax5.set_title('Right Arm Quaternion')
        ax5.legend()

        # Distance plot
        # distances = torch.norm(self.left_pos - self.right_pos, dim=1).cpu().numpy()
        # ax6 = fig.add_subplot(236)
        # ax6.plot(times, distances, 'k-')
        # ax6.axhline(self.fixed_distance, color='r', linestyle='--')
        # ax6.set_title('Distance Between Arms')

        plt.tight_layout()
        plt.show()

    def _quaternion_to_matrix(self, q):
        """Convert quaternion to rotation matrix (numpy version for visualization)."""
        q = q / np.linalg.norm(q)
        x, y, z, w = q[0], q[1], q[2], q[3]

        rot = np.zeros((3, 3))

        rot[0, 0] = 1 - 2 * y * y - 2 * z * z
        rot[0, 1] = 2 * x * y - 2 * z * w
        rot[0, 2] = 2 * x * z + 2 * y * w

        rot[1, 0] = 2 * x * y + 2 * z * w
        rot[1, 1] = 1 - 2 * x * x - 2 * z * z
        rot[1, 2] = 2 * y * z - 2 * x * w

        rot[2, 0] = 2 * x * z - 2 * y * w
        rot[2, 1] = 2 * y * z + 2 * x * w
        rot[2, 2] = 1 - 2 * x * x - 2 * y * y

        return rot

def mk_dir(path_dir):
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir, exist_ok=True)
        return True
    else:
        return False



# Example usage:
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     B, N = 2, 100
#     point_clouds = torch.rand((B, N, 3), device=device)
#     picks = torch.tensor([[0.5, 0.5, 0.0],
#                           [0.6, 0.5, 0.0]], device=device)
#
#     traj, places, assigned_places = generate_single_fold_with_random_places_quat(
#         point_clouds,
#         picks,
#         timesteps=20,
#         arc_height=0.15,
#         seed=42
#     )
#
#     print("Trajectory shape:", traj.shape)  # (B, 2, T, 7)
#     print("First pose [pos+quat]:", traj[0, 0, 0])

# # Example usage
# if __name__ == "__main__":
#     # Parameters
#     num_timesteps = 5000
#     via_point_spacing = 50
#     fixed_distance = 0.8
#
#     # Initial poses with quaternion orientations
#     initial_left = torch.tensor([-0.4, 0, 0, 0, 0, 0, 1], dtype=torch.float32, device="cuda")
#     initial_right = torch.tensor([0.4, 0, 0, 0, 0, 1, 0], dtype=torch.float32, device="cuda")
#
#     # Generate trajectory
#     start_time = time.time()
#     traj = BimanualQuaternionTrajectory(
#         num_timesteps=num_timesteps,
#         via_point_spacing=via_point_spacing,
#         initial_pose_left=initial_left,
#         initial_pose_right=initial_right,
#         device='cuda'
#     )
#     print(f"Trajectory generation time: {time.time() - start_time:.4f} seconds")

    # # Visualize
    # traj.visualize(sample_every=1)
    #
    # # Access full trajectories
    # left_pos, left_quat = traj.get_left_poses()
    # right_pos, right_quat = traj.get_right_poses()
    #
    # # Verify distance constraint
    # distances = torch.norm(left_pos - right_pos, dim=1)
    # print(
    #     f"Distance stats - Min: {distances.min().item():.4f}, Max: {distances.max().item():.4f}, Mean: {distances.mean().item():.4f}")
    #
    # # Example: Get pose at specific timestep
    # t = 1000
    # left_pos_t = left_pos[t].cpu().numpy()
    # left_quat_t = left_quat[t].cpu().numpy()
    # print(f"\nLeft arm at timestep {t}:")
    # print(f"Position: {left_pos_t}")
    # print(f"Quaternion: {left_quat_t} (norm: {np.linalg.norm(left_quat_t):.4f})")



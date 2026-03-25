import numpy as np
import trimesh
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

# Load OBJ
mesh = trimesh.load("/home/ubuntu/Downloads/Tshirt.obj")
vertices = mesh.vertices

# Define a 90° rotation around X
theta = np.deg2rad(90)
Rx = np.array([
    [1, 0,           0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta),  np.cos(theta)]
])

# Apply rotation
rotated_vertices = vertices @ Rx.T

rotated_vertices_t = torch.tensor(rotated_vertices, device="cuda")
rotated_vertices_t = rotated_vertices_t - rotated_vertices_t.mean()
torch.save(rotated_vertices_t, "/home/ubuntu/Github/cuhk_leph_learning/soft_flow/data/shirt_sampled_idx1.pt")

# Plot rotated vertices
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], s=1)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.savefig("/home/ubuntu/Downloads/Tshirt1.png", dpi=300)

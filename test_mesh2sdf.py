# import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# import mesh_to_sdf
import trimesh
import numpy as np
import torch
from pysdf import SDF
import matplotlib.pyplot as plt


# receiver_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result_0827_nosinglepene/smpl_mesh_1.obj"
receiver_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/test_0.obj"
mesh = trimesh.load(receiver_path)

# intruder_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result_0827_nosinglepene/smpl_mesh_0.obj"
intruder_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/test_1.obj"
intruder = trimesh.load(intruder_path)
int_verts = intruder.vertices


sdf_f = SDF(mesh.vertices, mesh.faces)
# query_pts = torch.tensor(int_verts, dtype=torch.float32, device="cuda:0")
sdf = sdf_f(int_verts)
print(np.unique(sdf))

# colors = (np.ones_like(int_verts) * np.abs(sdf[:, None]) * 255).astype(np.uint8)
colors = np.zeros_like(int_verts, dtype=np.uint8)
colors[sdf > 0, :] = np.array([255, 0, 0])
colors[sdf <= 0, :] = np.array([125, 125, 125])
save_mesh = trimesh.Trimesh(vertices=int_verts, faces=intruder.faces, vertex_colors=colors)
save_mesh.export("test.obj")

import os
import torch
import numpy as np
import pickle
from smplx.body_models import SMPL, SMPLX
import trimesh
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds, Meshes
from tqdm import tqdm

def point_mesh_distance(meshes, pcls):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()    # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]    # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = _PointFaceDistance.apply(
        points, points_first_idx, tris, tris_first_idx, max_points, 5e-3
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()    # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()    # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = torch.sqrt(point_to_face) * weights_p
    point_dist = point_to_face.sum() / N

    return point_dist

def calculate_chamfer_p2s(src_mesh, tgt_mesh, num_samples=1000):

    tgt_points = Pointclouds(sample_points_from_meshes(tgt_mesh, num_samples))
    src_points = Pointclouds(sample_points_from_meshes(src_mesh, num_samples))
    p2s_dist = point_mesh_distance(src_mesh, tgt_points) * 100.0
    chamfer_dist = (point_mesh_distance(tgt_mesh, src_points) * 100.0 + p2s_dist) * 0.5

    return chamfer_dist, p2s_dist

def main():
    seq = "hug/hug13"
    root = f"/home/vclab/8T_SSD1/dataset/Hi4D/{seq}/smpl_mesh"
    frames = list(range(11, 111))
    NUM_PERSON = 2

    chamfer_total = 0.0
    p2s_total = 0.0
    lines = ""
    chamfer_total_prev = 0.0
    p2s_total_prev = 0.0
    for frame_name in tqdm(frames):
        # gt_path = "/home/vclab/8T_SSD1/dataset/Hi4D/backhug/backhug02/smpl/000001.npz"
        # pred_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/0921_default/000136/smpl_param.pkl"
        gt_path = os.path.join(root, f"mesh_{frame_name:06d}_gt.obj")
        pred_path = os.path.join(root, f"mesh_{frame_name:06d}.obj")
        prev_path = os.path.join(root, f"mesh_{frame_name:06d}_before.obj")

        mesh_gt = trimesh.load(gt_path)
        mesh_pred = trimesh.load(pred_path)
        mesh_prev = trimesh.load(prev_path)

        mesh_gt = Meshes(verts=[torch.tensor(mesh_gt.vertices, dtype=torch.float32)], faces=[torch.tensor(mesh_gt.faces)])
        mesh_pred = Meshes(verts=[torch.tensor(mesh_pred.vertices, dtype=torch.float32)], faces=[torch.tensor(mesh_pred.faces)])
        mesh_prev = Meshes(verts=[torch.tensor(mesh_prev.vertices, dtype=torch.float32)], faces=[torch.tensor(mesh_prev.faces)])
        chamfer, p2s = calculate_chamfer_p2s(mesh_pred, mesh_gt)
        chamfer_total += chamfer.item()
        p2s_total += p2s.item()
        lines += f"FRAME: {frame_name:03d} CHAMF: {chamfer.item():6.4f} P2S: {p2s.item():6.4f}\n"
        chamfer_prev, p2s_prev = calculate_chamfer_p2s(mesh_prev, mesh_gt)
        chamfer_total_prev += chamfer_prev.item()
        p2s_total_prev += p2s_prev.item()
    with open(os.path.join(root, "evaluation.txt"), "w") as fp:
        fp.write(f"MEAN CHAMF: {chamfer_total / len(frames):6.4f} P2S: {p2s_total / len(frames):6.4f}\n")
        fp.write(f"MEAN PREV CHAMF: {chamfer_total_prev / len(frames):6.4f} P2S: {p2s_total_prev / len(frames):6.4f}\n")
        fp.write(lines)
    print(f"MEAN CHAMF: {chamfer_total / len(frames):6.4f} P2S: {p2s_total / len(frames):6.4f}")
    print(f"MEAN PREV CHAMF: {chamfer_total_prev / len(frames):6.4f} P2S: {p2s_total_prev / len(frames):6.4f}")

if __name__ == "__main__":
    main()
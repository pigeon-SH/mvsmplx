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
    gt_root = "/home/vclab/8T_SSD1/dataset/Hi4D/backhug/backhug02/smpl"
    pred_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/0921_default_openpose"
    frames = list(range(1, 151))
    NUM_PERSON = 2

    smpl_template_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/smplx/models/smpl/SMPL_NEUTRAL.pkl"
    smplx_template_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/smplx/models/smplx/SMPLX_NEUTRAL.npz"
    smpl_gt_model = SMPL(smpl_template_path)
    smpl_pred_model = SMPLX(smplx_template_path, num_pca_comps=12)

    chamfer_total = 0.0
    p2s_total = 0.0
    lines = ""
    for frame_name in tqdm(frames):
        # gt_path = "/home/vclab/8T_SSD1/dataset/Hi4D/backhug/backhug02/smpl/000001.npz"
        # pred_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/0921_default/000136/smpl_param.pkl"
        gt_path = os.path.join(gt_root, f"{frame_name:06d}.npz")
        pred_path = os.path.join(pred_root, f"{frame_name:06d}", "smpl_param.pkl")      

        param_gt = dict(np.load(gt_path))
        with open(pred_path, "rb") as f:
            param_pred = pickle.load(f)
        param_pred.sort(key=lambda x:x['person_id'])

        meshes_gt = []
        meshes_pred = []
        for person_id in range(NUM_PERSON):
            # load gt
            betas = torch.tensor(param_gt['betas'][person_id][None], dtype=torch.float32)
            body_pose = torch.tensor(param_gt['body_pose'][person_id][None], dtype=torch.float32)
            global_orient = torch.tensor(param_gt['global_orient'][person_id][None], dtype=torch.float32)
            transl = torch.tensor(param_gt['transl'][person_id][None], dtype=torch.float32)
            smpl_gt = smpl_gt_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
            
            faces_gt = smpl_gt_model.faces
            verts_gt = smpl_gt.vertices[0]
            mesh_gt = trimesh.Trimesh(vertices=verts_gt, faces=faces_gt)
            meshes_gt.append(mesh_gt)

            # load pred
            betas = torch.tensor(param_pred[person_id]['result']['betas'], dtype=torch.float32)
            global_orient = torch.tensor(param_pred[person_id]['result']['global_orient'], dtype=torch.float32)
            body_pose = torch.tensor(param_pred[person_id]['result']['body_pose'], dtype=torch.float32)[:, 3:]
            left_hand_pose = torch.tensor(param_pred[person_id]['result']['left_hand_pose'], dtype=torch.float32)
            right_hand_pose = torch.tensor(param_pred[person_id]['result']['right_hand_pose'], dtype=torch.float32)
            transl = torch.tensor(param_pred[person_id]['result']['global_body_translation'], dtype=torch.float32)[None]
            expression = torch.tensor(param_pred[person_id]['result']['expression'], dtype=torch.float32)
            jaw_pose = torch.tensor(param_pred[person_id]['result']['jaw_pose'], dtype=torch.float32)
            leye_pose = torch.tensor(param_pred[person_id]['result']['leye_pose'], dtype=torch.float32)
            reye_pose = torch.tensor(param_pred[person_id]['result']['reye_pose'], dtype=torch.float32)
            smpl_pred = smpl_pred_model(betas=betas, global_orient=global_orient, body_pose=body_pose,
                        left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl,
                        expression=expression, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose)
            
            faces_pred = smpl_pred_model.faces
            verts_pred = smpl_pred.vertices[0]
            mesh_pred = trimesh.Trimesh(vertices=verts_pred, faces=faces_pred)
            meshes_pred.append(mesh_pred)

        mesh_gt = meshes_gt[0]
        mesh_pred = meshes_pred[0]
        for i in range(1, NUM_PERSON):
            mesh_gt += meshes_gt[i]
            mesh_pred += meshes_pred[i]

        mesh_gt = Meshes(verts=[torch.tensor(mesh_gt.vertices, dtype=torch.float32)], faces=[torch.tensor(mesh_gt.faces)])
        mesh_pred = Meshes(verts=[torch.tensor(mesh_pred.vertices, dtype=torch.float32)], faces=[torch.tensor(mesh_pred.faces)])
        chamfer, p2s = calculate_chamfer_p2s(mesh_pred, mesh_gt)
        chamfer_total += chamfer.item()
        p2s_total += p2s.item()
        lines += f"FRAME: {frame_name:03d} CHAMF: {chamfer.item():6.4f} P2S: {p2s.item():6.4f}\n"
    with open(os.path.join(pred_root, "evaluation.txt"), "w") as fp:
        fp.write(f"MEAN CHAMF: {chamfer_total / len(frames):6.4f} P2S: {p2s_total / len(frames):6.4f}\n")
        fp.write(lines)
    print(f"MEAN CHAMF: {chamfer_total / len(frames):6.4f} P2S: {p2s_total / len(frames):6.4f}")

if __name__ == "__main__":
    main()
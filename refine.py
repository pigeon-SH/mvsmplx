import os
import cv2
import pickle
import trimesh
import utils
import smplx
from data_parser import read_keypoints, update_mask_with_keypoints
import torch
import numpy as np
from camera import PerspectiveCamera
from pysdf import SDF
from tqdm import tqdm
from smplx.lbs import transform_mat

torch.autograd.set_detect_anomaly(True)

max_persons = 2
data_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data_smplx/Hi4D/talk/talk01/000140"
result_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result_0827_nosinglepene"
cam_param = np.load("/home/vclab/dataset/Hi4D/talk/talk01/cameras/rgb_cameras.npz")
device = torch.device('cuda')

rho = 100
robustifier = utils.GMoF(rho=rho)
joint_mapper = utils.JointMapper(utils.smpl_to_openpose())
# body_models = [smplx.create(gender='neutral', model_path="./smplx/models", model_type='smplx', num_pca_comps=12, joint_mapper=joint_mapper).to(device=device).eval()] * max_persons

fp = open(os.path.join(result_root, "smpl_param.pkl"), "rb")
smpl_params = [None, None]
smpl_param = pickle.load(fp)

# betas = []
# global_orient = []
# body_pose = []
# left_hand_pose = []
# right_hand_pose = []
# transl = []
# expression = []
# jaw_pose = []
# leye_pose = []
# reye_pose = []
body_models = []
for i in range(max_persons):
    # smpl_params[smpl_param[i]['person_id']] = smpl_param[i]['result']
    # betas.append(smpl_param[i]['result']['betas']) # (1, 300)
    # global_orient.append(smpl_param[i]['result']['global_orient']) # (1, 3)
    # body_pose.append(smpl_param[i]['result']['body_pose'][:, 3:]) # (1, 63)
    # left_hand_pose.append(smpl_param[i]['result']['left_hand_pose']) # (1, 12)
    # right_hand_pose.append(smpl_param[i]['result']['right_hand_pose']) # (1, 12)
    # transl.append(smpl_param[i]['result']['global_body_translation'][None]) # (1, 3)
    # expression.append(smpl_param[i]['result']['expression']) # (1, 100)
    # jaw_pose.append(smpl_param[i]['result']['jaw_pose']) # (1, 3)
    # leye_pose.append(smpl_param[i]['result']['leye_pose']) # (1, 3)
    # reye_pose.append(smpl_param[i]['result']['reye_pose']) # (1, 3)
# betas = torch.tensor(np.concatenate(betas, axis=0), dtype=torch.float32, device=device)
# global_orient = torch.tensor(np.concatenate(global_orient, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
# body_pose = torch.tensor(np.concatenate(body_pose, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
# left_hand_pose = torch.tensor(np.concatenate(left_hand_pose, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
# right_hand_pose = torch.tensor(np.concatenate(right_hand_pose, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
# transl = torch.tensor(np.concatenate(transl, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
# expression = torch.tensor(np.concatenate(expression, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
# jaw_pose = torch.tensor(np.concatenate(jaw_pose, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
# leye_pose = torch.tensor(np.concatenate(leye_pose, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
# reye_pose = torch.tensor(np.concatenate(reye_pose, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
    smpl_param[i]['result']['transl'] = smpl_param[i]['result']['global_body_translation'][None]
    smpl_param[i]['result']['body_pose'] = smpl_param[i]['result']['body_pose'][:, 3:]
    model_params = dict(model_path="./smplx/models",
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        dtype=torch.float32,
                        model_type='smplx', 
                        num_pca_comps=12, 
                        **smpl_param[i]['result'])
    body_model = smplx.create(gender="neutral", **model_params).to(device=device)
    body_model.betas.requires_grad_(False)
    body_models.append(body_model)

cam_ids = cam_param['ids']
frame_idx = int(os.path.basename(data_root))

keypoints = []
masks = []
cameras = []
for cam_id in cam_ids:
    fn = f"{frame_idx:06d}_{cam_id:02d}"
    kpts_fn = os.path.join(data_root, "keypoints", fn + "_keypoints.json")
    kpts_tuple = read_keypoints(kpts_fn, max_persons=max_persons)
    keypoint = kpts_tuple.keypoints

    mask_fn = os.path.join(data_root, "mask_gt", fn + ".png")
    mask = cv2.imread(mask_fn).astype(np.float32)
    mask = update_mask_with_keypoints(mask, keypoint)

    cam_idx = np.arange(len(cam_param['ids']))[cam_param['ids'] == cam_id].item()
    K = cam_param['intrinsics'][cam_idx]
    RT = cam_param['extrinsics'][cam_idx]
    camera = PerspectiveCamera().to(device=device)
    camera.focal_length_x = torch.full([1], K[0, 0]).to(device=device)
    camera.focal_length_y = torch.full([1], K[1, 1]).to(device=device)
    camera.center = torch.tensor([K[0, 2], K[1, 2]], dtype=torch.float32).unsqueeze(0).to(device=device)
    camera.rotation.data = torch.from_numpy(RT[:3, :3].astype(np.float32)).unsqueeze(0).to(device=device)
    camera.translation.data = torch.from_numpy(RT[:3, 3].astype(np.float32)).unsqueeze(0).to(device=device)
    camera.rotation = camera.rotation.requires_grad_(False)
    camera.translation = camera.translation.requires_grad_(False)
    
    for person_id in range(max_persons):
        kpts = keypoint[person_id]
        kpts_mask = mask[kpts[:, 1].astype(np.int16), kpts[:, 0].astype(np.int16)]
        keypoint[person_id][kpts_mask != (person_id + 1)] = 0.
    
    keypoint = torch.tensor(keypoint, dtype=torch.float32, device=device)
    mask = torch.tensor(mask, dtype=torch.float32, device=device)
    keypoints.append(keypoint)
    masks.append(mask)
    cameras.append(camera)

joints_to_ign = torch.tensor((1, 9, 12))
joint_weights = torch.ones(keypoints[0].shape[1], dtype=torch.float32, device=device)
joint_weights[joints_to_ign] = 0.

EPOCHS = 20
# params = [global_orient, body_pose, left_hand_pose, right_hand_pose, transl, expression, jaw_pose, leye_pose, reye_pose]
final_params = []
for person_id in range(max_persons):
    body_params = list(body_models[person_id].parameters())
    final_params += list(filter(lambda x: x.requires_grad, body_params))
optimizer = torch.optim.Adam(params=final_params, lr=0.01)
for epoch in range(EPOCHS):
    loss = 0.
    for i in range(len(masks)):
        kpts = keypoints[i]
        mask = masks[i]
        camera = cameras[i]

        camera_mat = transform_mat(camera.rotation, camera.translation.unsqueeze(dim=-1))[0]
        camera_center = torch.inverse(camera_mat)[:3, 3]

        body_model_face = body_model.faces
        body_model_outputs = []
        for i in range(max_persons):
            body_model_outputs.append(body_models[i]())

        for person_id in range(max_persons):
            body_model = body_models[person_id]
            proj_joint = camera(body_model().joints)[0]
            joints_conf = kpts[person_id][..., -1]
            joints_conf[joints_conf < 0.3] = 0.
            joint_diff = robustifier(kpts[person_id][:, :2] - proj_joint)
            joint_loss = joint_diff * joint_weights.unsqueeze(-1) * joints_conf.unsqueeze(-1)
            loss += torch.mean(joint_diff) * 0.001

        intruder_indices = [0, 1]
        receiver_indices = [1, 0]
        verts_colls = []
        for i in range(max_persons):
            int_idx = intruder_indices[i]
            rec_idx = receiver_indices[i]
            # receiver = trimesh.Trimesh(vertices=body_model_outputs[rec_idx].vertices.cpu().detach().numpy()[0], faces=body_model_face.cpu().detach().numpy())
            rec_verts = body_model_outputs[rec_idx].vertices[0]
            int_verts = body_model_outputs[int_idx].vertices[0]
            # sdf = mesh_to_sdf.mesh_to_sdf(receiver, int_verts.cpu().detach().numpy(), surface_point_method='scan', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
            sdf_f = SDF(rec_verts.cpu().detach().numpy(), body_model_face) # receiver.vertices, receiver.faces)
            sdf = sdf_f(int_verts.cpu().detach().numpy())

            sdf_mask = sdf > 0 # < 0
            verts_coll = int_verts[sdf_mask, :]
            verts_colls.append(verts_coll)
        
        pos_dists = []
        neg_dists = []
        for person_id in intruder_indices:
            verts_proj = camera((verts_colls[person_id])[None])[0]
            # mask = torch.tensor(mask, device=body_model_face.device)
            y_idx = torch.clip(verts_proj[..., 1].int(), 0, mask.shape[0]-1)
            x_idx = torch.clip(verts_proj[..., 0].int(), 0, mask.shape[1]-1)
            verts_proj_mask = mask[y_idx, x_idx] == (person_id + 1)
            verts_proj_mask = verts_proj_mask[..., 0]
            pos_dist = ((verts_colls[person_id][verts_proj_mask] - camera_center[None]) ** 2).mean()
            neg_dist = ((verts_colls[person_id][~verts_proj_mask] - camera_center[None]) ** 2).mean()
            pos_dists.append(pos_dist)
            neg_dists.append(neg_dist)

        m = 1.0
        ranking_loss = 0.
        r_loss_0 = m + pos_dists[0] - neg_dists[1]
        r_loss_1 = m + pos_dists[1] - neg_dists[0]
        ranking_loss = ranking_loss + r_loss_0 if r_loss_0 > 0 else ranking_loss
        ranking_loss = ranking_loss + r_loss_1 if r_loss_1 > 0 else ranking_loss
        loss += ranking_loss
    
    if loss == 0:
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("EPOCH", epoch, "LOSS", loss.item())


for person_id in range(max_persons):
    body_model = body_models[person_id].eval()
    body_model_output = body_model()
    verts = body_model_output.vertices.cpu().detach().numpy()[0]
    faces = body_model.faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(f"test_{person_id}.obj")


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
from vposer.model_loader import load_vposer
from prior import create_prior

max_persons = 2
device = torch.device('cuda')
torch.autograd.set_detect_anomaly(True)

def load_prior():
    body_pose_prior = create_prior(prior_type='l2').to(device=device)
    jaw_prior = create_prior(prior_type='l2').to(device=device)
    expr_prior = create_prior(prior_type='l2').to(device=device)
    left_hand_prior = create_prior(prior_type='l2').to(device=device)
    right_hand_prior = create_prior(prior_type='l2').to(device=device)
    shape_prior = create_prior(prior_type='l2').to(device=device)
    angle_prior = create_prior(prior_type='angle').to(device=device)
    return body_pose_prior, jaw_prior, expr_prior, left_hand_prior, right_hand_prior, shape_prior, angle_prior

def load_dataset(data_root, cam_path):
    cam_param = np.load(cam_path)
    cam_ids = cam_param['ids']
    frame_idx = int(os.path.basename(data_root))

    keypoints = []
    masks = []
    cameras = []
    for cam_id in cam_ids:
        fn = f"{frame_idx:06d}_{cam_id:02d}"
        # kpts_fn = os.path.join(data_root, "keypoints", fn + "_keypoints.json")
        # kpts_tuple = read_keypoints(kpts_fn, max_persons=max_persons)
        # keypoint = kpts_tuple.keypoints
        kpts_fn = os.path.join(data_root, "keypoints_masked", fn + "_keypoints.npy")
        keypoint = np.load(kpts_fn)

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
    return keypoints, masks, cameras, joint_weights

def load_body_model(smpl_path=None, vposer_latent_dim=32):
    if smpl_path:
        fp = open(smpl_path, "rb")
        smpl_param = pickle.load(fp)

    joint_mapper = utils.JointMapper(utils.smpl_to_openpose())
    body_models = []
    for i in range(max_persons):
        if smpl_path:
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
        else:
            body_mean_poses = torch.zeros([max_persons, vposer_latent_dim], dtype=torch.float32)
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
                                num_pca_comps=12, )
                                # **smpl_param[i]['result'])
            body_model = smplx.create(gender="neutral", **model_params).to(device=device)
        body_models.append(body_model)
    return body_models

def main():
    data_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data_smplx/Hi4D/talk/talk01/000140"
    result_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/0901_nointer_noself_masked"
    cam_path = "/home/vclab/dataset/Hi4D/talk/talk01/cameras/rgb_cameras.npz"
    vposer_path = "./vposer/models"
    
    rho = 100
    robustifier = utils.GMoF(rho=rho)
    smpl_path = os.path.join(result_root, "smpl_param.pkl")
    # smpl_path = None
    body_models = load_body_model(smpl_path)
    body_model_face = body_models[0].faces
    keypoints, masks, cameras, joint_weights = load_dataset(data_root, cam_path)
    transls = torch.tensor([[0, 0, 0] for _ in range(max_persons)], dtype=torch.float32, device=device, requires_grad=True)

    # vposer
    pose_embeddings = torch.zeros([max_persons, 32], dtype=torch.float32, device=device, requires_grad=True)
    # vposer_ckpt = os.path.expandvars(vposer_ckpt)
    vposer, _ = load_vposer(vposer_path, vp_model='snapshot')
    vposer = vposer.to(device=device)
    vposer.eval()

    # priors
    body_pose_prior, jaw_prior, expr_prior, left_hand_prior, right_hand_prior, shape_prior, angle_prior = load_prior()

    EPOCHS = 10
    final_params = [pose_embeddings, transls]
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

            body_model_outputs = []
            for i in range(max_persons):
                body_pose = vposer.decode(pose_embeddings[person_id], output_type='aa').view(1, -1)
                body_model_outputs.append(body_models[i](body_pose=body_pose, return_full_pose=True))

            for person_id in range(max_persons):
                # joint loss
                proj_joint = camera(body_model_outputs[person_id].joints + transls[person_id])[0]
                joints_conf = kpts[person_id][..., -1]
                # joints_conf = torch.where(joints_conf < 0.3, 0., joints_conf)
                joint_diff = robustifier(kpts[person_id][:, :2] - proj_joint)
                joint_loss = (joint_diff * joint_weights.unsqueeze(-1) * joints_conf.unsqueeze(-1)).mean()
                loss += joint_loss

                # prior loss
                pprior_loss = pose_embeddings[person_id].pow(2).sum()
                shape_loss = torch.sum(shape_prior(body_model_outputs[person_id].betas))
                body_pose = body_model_outputs[person_id].full_pose[:, 3:66]
                angle_prior_loss = torch.sum(angle_prior(body_pose))
                left_hand_prior_loss = torch.sum(left_hand_prior(body_model_outputs[person_id].left_hand_pose))
                right_hand_prior_loss = torch.sum(expr_prior(body_model_outputs[person_id].right_hand_pose))
                expression_loss = torch.sum(left_hand_prior(body_model_outputs[person_id].expression))
                jaw_prior_loss = torch.sum(jaw_prior(body_model_outputs[person_id].jaw_pose))
                prior_loss = (pprior_loss + shape_loss + angle_prior_loss + jaw_prior_loss + expression_loss + left_hand_prior_loss + right_hand_prior_loss)
                
                loss += prior_loss

            intruder_indices = [0, 1]
            receiver_indices = [1, 0]
            verts_colls = []
            for i in range(max_persons):
                int_idx = intruder_indices[i]
                rec_idx = receiver_indices[i]
                rec_verts = body_model_outputs[rec_idx].vertices[0] + transls[rec_idx][None]
                int_verts = body_model_outputs[int_idx].vertices[0] + transls[int_idx][None]
                sdf_f = SDF(rec_verts.cpu().detach().numpy(), body_model_face) # receiver.vertices, receiver.faces)
                sdf = sdf_f(int_verts.cpu().detach().numpy())

                sdf_mask = sdf > 0 # < 0
                verts_coll = int_verts[sdf_mask, :]
                verts_colls.append(verts_coll)
            
            pos_dists = []
            neg_dists = []
            for person_id in intruder_indices:
                verts_proj = camera((verts_colls[person_id] + transls[person_id][None])[None])[0]
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
        print(f"EPOCH {epoch:02d} LOSS: {loss.item():6.4f} JOINT: {joint_loss.item():6.4f} PRIOR: {prior_loss.item():6.4f} RANKING: {ranking_loss:6.4f}")


    verts_total = []
    faces_total = []
    for person_id in range(max_persons):
        body_model = body_models[person_id].eval()
        body_model_output = body_model()
        verts = body_model_output.vertices.cpu().detach().numpy()[0] + transls[person_id].cpu().detach().numpy()
        faces = body_model.faces
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(f"test_{person_id}.obj")
    
        verts_total.append(verts)
        faces_total.append(faces + person_id * len(verts))
    
    vertices = np.concatenate(verts_total, axis=0)
    faces = np.concatenate(faces_total, axis=0)
    out_mesh = trimesh.Trimesh(vertices, faces)
    out_mesh.export("test_total.obj")

if __name__ == "__main__":
    main()

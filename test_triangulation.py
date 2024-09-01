import os
import os.path
import cv2
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from data_parser import read_keypoints, update_mask_with_keypoints
from refine import load_body_model

def triangulate_point(x1, x2, P1, P2):
    """
    x1: First camera pixel coordinates (2x1 numpy array or list)
    x2: Second camera pixel coordinates (2x1 numpy array or list)
    P1: First camera projection matrix (3x4 numpy array)
    P2: Second camera projection matrix (3x4 numpy array)
    
    Returns:
    X: Triangulated 3D point (4x1 numpy array)
    """
    # Create the A matrix as described above
    A = np.array([
        x1[0] * P1[2, :] - P1[0, :],
        x1[1] * P1[2, :] - P1[1, :],
        x2[0] * P2[2, :] - P2[0, :],
        x2[1] * P2[2, :] - P2[1, :]
    ])

    # Compute the SVD of A
    U, S, Vt = np.linalg.svd(A)
    
    # The solution X is the last column of V (or last row of Vt)
    X = Vt[-1]

    # Homogeneous to 3D coordinates by dividing by the last element
    X = X / X[3]
    
    return X[:3]

data_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data_smplx/Hi4D/talk/talk01/000140"
cam_path = "/home/vclab/dataset/Hi4D/talk/talk01/cameras/rgb_cameras.npz"
smpl_path = os.path.join("/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/0901_nointer_noself/smpl_param.pkl")
body_models = load_body_model(smpl_path)

cam_param = np.load(cam_path)
cam_ids = cam_param['ids']
frame_idx = int(os.path.basename(data_root))

keypoint_dict = {}
for cam_id in cam_ids:
    keypoint_dict[cam_id] = []
for person_id in range(2):
    verts = body_models[person_id]().joints.cpu().detach().numpy()[0]
    verts = np.concatenate([verts, np.ones_like(verts[:, 0:1])], axis=1)
    for cam_id in cam_ids:
        fn = f"{frame_idx:06d}_{cam_id:02d}"
        kpts_fn = os.path.join(data_root, "keypoints", fn + "_keypoints.json")
        kpts_tuple = read_keypoints(kpts_fn)
        kpts = kpts_tuple.keypoints[person_id]

        cam_idx = np.arange(len(cam_param['ids']))[cam_param['ids'] == cam_id].item()
        K = cam_param['intrinsics'][cam_idx]
        RT = cam_param['extrinsics'][cam_idx]
        P = np.matmul(K, RT)

        fn = f"{frame_idx:06d}_{cam_id:02d}"
        orig_path = os.path.join(data_root, "color", fn + ".jpg")
        orig_img = cv2.imread(orig_path)
        H, W = orig_img.shape[:2]
        img = orig_img.copy()
        verts_cam = np.matmul(RT, verts.T).T
        verts_cam = verts_cam / verts_cam[:, -1:]
        verts_2d = np.matmul(K, verts_cam.T).T # (N, 3)

        mask_fn = os.path.join(data_root, "mask_gt", fn + ".png")
        mask = cv2.imread(mask_fn).astype(np.float32)
        mask = update_mask_with_keypoints(mask, kpts_tuple.keypoints)
        
        diff = np.abs(kpts[:, :2] - verts_2d[:, :2])
        diff[:, 0] /= W
        diff[:, 1] /= H
        diff = diff.sum(axis=1)
        kpts_mask_smpl = diff > 0.2

        kpts_mask_mask = (mask[kpts[:, 1].astype(np.int16), kpts[:, 0].astype(np.int16)] != (person_id + 1)).any(axis=-1)
        # kpts_masked[(kpts_mask != (person_id + 1)).any(axis=-1), 2] = 0.0

        kpts_mask = np.logical_or(kpts_mask_smpl, kpts_mask_mask)
        kpts_masked = kpts.copy()
        kpts_masked[kpts_mask, 2] = 0.0

        verts_2d[kpts_masked[:, 2] == 0, 2] = 0.8
        kpts_masked[kpts_masked[:, 2] == 0] = verts_2d[kpts_masked[:, 2] == 0]
        keypoint_dict[cam_id].append(kpts_masked)

save_dir = os.path.join(data_root, "keypoints_masked")
os.makedirs(save_dir, exist_ok=True)
for cam_id in cam_ids:
    fn = f"{frame_idx:06d}_{cam_id:02d}"
    keypoint_dict[cam_id] = np.stack(keypoint_dict[cam_id], axis=0)
    save_path = os.path.join(save_dir, fn + "_keypoints.npy")
    np.save(save_path, keypoint_dict[cam_id])



#     color = (0, 0, 255) # B, G, R
#     for pts in verts_2d:
#         cv2.circle(img, (int(pts[0]), int(pts[1])), radius=8, color=color, thickness=-1)
#     color = (0, 255, 0) # B, G, R
#     for pts in kpts:
#         cv2.circle(img, (int(pts[0]), int(pts[1])), radius=8, color=color, thickness=-1)
#     color = (255, 0, 0) # B, G, R
#     for pts in kpts_masked:
#         if pts[2] > 0.3:
#             cv2.circle(img, (int(pts[0]), int(pts[1])), radius=8, color=color, thickness=-1)
#     cv2.imwrite(f"test_{cam_id:02d}.png", img)
# diffs = np.concatenate(diff_list, axis=0)
# plt.plot(np.arange(len(diffs)), diffs)
# plt.savefig("test.png")
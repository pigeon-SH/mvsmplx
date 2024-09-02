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
body_models, _, _ = load_body_model(smpl_path)

cam_param = np.load(cam_path)
cam_ids = cam_param['ids']
frame_idx = int(os.path.basename(data_root))

keypoint_dict = {}
kpts_persons = []
for cam_id in cam_ids:
    keypoint_dict[cam_id] = []
for person_id in range(2):
    # verts = body_models[person_id]().joints.cpu().detach().numpy()[0]
    # verts = np.concatenate([verts, np.ones_like(verts[:, 0:1])], axis=1)
    P_list = []
    RT_list = []
    K_list = []
    kpts_list = []
    for cam_id in cam_ids:
        fn = f"{frame_idx:06d}_{cam_id:02d}"
        kpts_fn = os.path.join(data_root, "keypoints", fn + "_keypoints.json")
        kpts_tuple = read_keypoints(kpts_fn)
        kpts = kpts_tuple.keypoints[person_id]

        cam_idx = np.arange(len(cam_param['ids']))[cam_param['ids'] == cam_id].item()
        K = cam_param['intrinsics'][cam_idx]
        RT = cam_param['extrinsics'][cam_idx]
        P = np.matmul(K, RT)

        P_list.append(P)
        RT_list.append(RT)
        K_list.append(K)
        kpts_list.append(kpts)

        fn = f"{frame_idx:06d}_{cam_id:02d}"
        orig_path = os.path.join(data_root, "color", fn + ".jpg")
        orig_img = cv2.imread(orig_path)
        H, W = orig_img.shape[:2]
    
    kpts = np.stack(kpts_list, axis=0)  # (V, J, 3)
    Ps = np.stack(P_list, axis=0)  # (V, 3, 4)
    kpts_persons.append(kpts)

    joint_list = []
    diff_list = []
    joint = np.array([-1000, -1000, -1000], dtype=np.float32)
    for j in range(kpts.shape[1]):
        scores = kpts[:, j, 2]
        top_idx = np.argsort(-scores)
        for i in range(len(top_idx) - 1):
            indices = top_idx[i:i+2]
            top_kpts = kpts[indices, j, :]
            x1 = top_kpts[0, :2]
            x2 = top_kpts[1, :2]
            top_Ps = Ps[indices]
            P1 = top_Ps[0]
            P2 = top_Ps[1]
            joint = triangulate_point(x1, x2, P1, P2)
            joint_inho = np.concatenate([joint, np.ones_like(joint[..., 0:1])], axis=-1)
            diffs = 0.
            for v in range(len(cam_ids)):
                j_cam = np.matmul(RT_list[v], joint_inho.T).T
                j_cam /= j_cam[..., -1:]
                j_2d = np.matmul(K_list[v], j_cam.T).T
                diff = (np.abs(kpts[v, j, :2] - j_2d[:2]) * (kpts[v, j, 2] != 0))
                diff[0] /= W
                diff[1] /= H
                diff = diff.mean()
                diffs += diff
                diff_list.append(diff)

            if diffs / len(cam_ids) < 0.05:
                break
        joint_list.append(joint)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(diff_list)), diff_list)
    fig.savefig("test.png")
    diff_list = []
    joints = np.stack(joint_list, axis=0)   # (J, 3)
    joints_inho = np.concatenate([joints, np.ones_like(joints[..., 0:1])], axis=-1)
    for v, cam_id in enumerate(cam_ids):
        j_cam = np.matmul(RT_list[v], joints_inho.T).T
        j_cam /= j_cam[..., -1:]
        j_2d = np.matmul(K_list[v], j_cam.T).T
        diff = (np.abs(kpts[v, :, :2] - j_2d[:, :2]) * (kpts[v, :, 2:] != 0))    # (J, 2)
        diff[:, 0] /= W
        diff[:, 1] /= H
        diff = diff.mean(axis=1)
        diff_list.append(diff)
        kpts_mask = diff > 0.05
        kpts_masked = kpts[v].copy()
        kpts_masked[kpts_mask, 2] = 0.0

        keypoint_dict[cam_id].append(kpts_masked)

    diffs = np.concatenate(diff_list, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(diffs)), diffs)
    fig.savefig("test_after.png")

save_dir = os.path.join(data_root, "keypoints_masked")
os.makedirs(save_dir, exist_ok=True)
for idx, cam_id in enumerate(cam_ids):
    fn = f"{frame_idx:06d}_{cam_id:02d}"
    keypoint_dict[cam_id] = np.stack(keypoint_dict[cam_id], axis=0)
    save_path = os.path.join(save_dir, fn + "_keypoints.npy")
    np.save(save_path, keypoint_dict[cam_id])

    orig_path = os.path.join(data_root, "color", fn + ".jpg")
    orig_img = cv2.imread(orig_path)
    H, W = orig_img.shape[:2]
    img = orig_img.copy()
    for person_id in range(2):
        kpts_orig = kpts_persons[person_id][idx]
        kpts_masked = keypoint_dict[cam_id][person_id]

        color = (0, 0, 125) if person_id == 0 else (125, 0, 0)
        for pts in kpts_orig:
            if pts[2] > 0.0:
                cv2.circle(img, (int(pts[0]), int(pts[1])), radius=8, color=color, thickness=-1)
        color = (0, 0, 255) if person_id == 0 else (255, 0, 0)
        for pts in kpts_masked:
            if pts[2] > 0.0:
                cv2.circle(img, (int(pts[0]), int(pts[1])), radius=8, color=color, thickness=-1)
    cv2.imwrite(f"test_{cam_id:02d}.png", img)
diffs = np.concatenate(diff_list, axis=0)
plt.plot(np.arange(len(diffs)), diffs)
plt.savefig("test.png")
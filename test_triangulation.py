import os
import cv2
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from data_parser import read_keypoints
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

person_id = 0
data_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data_smplx/Hi4D/talk/talk01/000140"
# mesh_path = f"/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/0901_nointer_noself/smpl_mesh_{person_id}.obj"
cam_path = "/home/vclab/dataset/Hi4D/talk/talk01/cameras/rgb_cameras.npz"
# data_root = "X:\\8T_SSD1\\extractSMPL\\MultiviewSMPLifyX\\data_smplx\\Hi4D\\talk\\talk01\\000140"
# mesh_path = f"X:\\8T_SSD1\\extractSMPL\\MultiviewSMPLifyX\\result\\0901_nointer_noself\\smpl_mesh_{person_id}.obj"
# cam_path = "X:\\dataset\\Hi4D\\talk\\talk01\\cameras\\rgb_cameras.npz"0
smpl_path = os.path.join("/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/0901_nointer_noself/smpl_param.pkl")
body_models = load_body_model(smpl_path)

cam_param = np.load(cam_path)
cam_ids = cam_param['ids']
frame_idx = int(os.path.basename(data_root))

kpts_list = []
P_list = []
RT_list = []
K_list = []
for cam_id in cam_ids:
    fn = f"{frame_idx:06d}_{cam_id:02d}"
    kpts_fn = os.path.join(data_root, "keypoints", fn + "_keypoints.json")
    kpts_tuple = read_keypoints(kpts_fn)
    kpts = kpts_tuple.keypoints[person_id]
    kpts_list.append(kpts)

    cam_idx = np.arange(len(cam_param['ids']))[cam_param['ids'] == cam_id].item()
    K = cam_param['intrinsics'][cam_idx]
    RT = cam_param['extrinsics'][cam_idx]
    P = np.matmul(K, RT)
    P_list.append(P)
    RT_list.append(RT)
    K_list.append(K)

kpts = np.stack(kpts_list, axis=0)  # (V, J, 3)
Ps = np.stack(P_list, axis=0)  # (V, 3, 4)
joint_list = []
for j in range(kpts.shape[1]):
    scores = kpts[:, j, 2]
    top_idx = np.argsort(-scores)[:2]
    top_kpts = kpts[top_idx, j, :]
    x1 = top_kpts[0, :2]
    x2 = top_kpts[1, :2]
    top_Ps = Ps[top_idx]
    P1 = top_Ps[0]
    P2 = top_Ps[1]
    joint = triangulate_point(x1, x2, P1, P2)
    joint_list.append(joint)
joints = np.stack(joint_list, axis=0)   # (J, 3)

# mesh = trimesh.load(mesh_path)
# verts = mesh.vertices
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(verts[..., 0], verts[..., 1], verts[..., 2])
# ax.scatter(joints[..., 0], joints[..., 1], joints[..., 2])
# # plt.show()
# plt.savefig("test.png")
verts = body_models[person_id]().joints.cpu().detach().numpy()[0]

verts = np.concatenate([verts, np.ones_like(verts[:, 0:1])], axis=1)
joints = np.concatenate([joints, np.ones_like(joints[:, 0:1])], axis=1)
for idx, cam_id in enumerate(cam_ids):
    fn = f"{frame_idx:06d}_{cam_id:02d}"
    orig_path = os.path.join(data_root, "color", fn + ".jpg")
    orig_img = cv2.imread(orig_path)
    img = orig_img.copy()
    # kpts_2d = kpts[idx]
    P = Ps[idx]
    RT = RT_list[idx]
    K = K_list[idx]
    verts_cam = np.matmul(RT, verts.T).T
    verts_cam = verts_cam / verts_cam[:, -1:]
    verts_2d = np.matmul(K, verts_cam.T).T # (N, 3)

    joints_cam = np.matmul(RT, joints.T).T
    joints_cam = joints_cam / joints_cam[:, -1:]
    joints_2d = np.matmul(K, joints_cam.T).T # (N, 3)

    mask = np.abs(verts - joints).sum(axis=-1) < 0.4
    # plt.plot(np.arange(len(mask)), mask)
    # plt.savefig("test.png")
    # print(mask)
    verts_2d = verts_2d[mask]
    joints_2d = joints_2d[mask]
    

    color = (0, 0, 255) # B, G, R
    for pts in verts_2d:
        cv2.circle(img, (int(pts[0]), int(pts[1])), radius=8, color=color, thickness=-1)
    color = (255, 0, 0) # B, G, R
    for pts in joints_2d:
        cv2.circle(img, (int(pts[0]), int(pts[1])), radius=8, color=color, thickness=-1)
    cv2.imwrite(f"test_{cam_id:02d}.png", img)
import os
import cv2
import json
import numpy as np
import pickle

IDENTITY = [np.array([255, 255, 255]), np.array([125, 125, 125])]

frames = list(range(20, 150))
cam_ids = [4, 16, 28, 40, 52, 64, 76, 88]
src_root = "data/Hi4D/talk/talk01"
dst_root = "/home/vclab/dataset/Hi4D/talk/talk01/smpl_pred"
os.makedirs(dst_root, exist_ok=True)

for frame_name in frames:
    dst_path = os.path.join(dst_root, f"{frame_name:06d}.npz")
    dst_param = {}
    betas = []
    global_ori = []
    poses = []
    trans = []
    scale = []
    for human_idx in range(2):
        src_path = os.path.join(src_root, f"{frame_name:06d}", f"smpl_{human_idx}", "smpl_param.pkl")
        with open(src_path, "rb") as fp:
            src_param = pickle.load(fp)
        # print(src_param.keys())
        betas.append(src_param['betas'])    # (1, 10)
        global_ori.append(src_param['global_orient'])   # (1, 3)
        poses.append(src_param['body_pose'][:, 3:])     # (1, 69)
        trans.append(src_param['global_body_translation'][None])  # (1, 3)
        scale.append(src_param['body_scale'][None])     # (1, 1)
    dst_param['betas'] = np.concatenate(betas, axis=0)  # (2, 10)
    dst_param['global_orient'] = np.concatenate(global_ori, axis=0) # (2, 3)
    dst_param['body_pose'] = np.concatenate(poses, axis=0)  # (2, 69)
    dst_param['transl'] = np.concatenate(trans, axis=0) # (2, 3)
    dst_param['scale'] = np.concatenate(scale, axis=0)  # (2, 1)
    np.savez(dst_path, **dst_param)
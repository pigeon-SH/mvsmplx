import numpy as np
from PIL import Image
import json
import os
import cv2

colors = ((0, 0, 255), (255, 0, 0))

# for cam_id in [4, 16, 28, 40, 52, 64, 76, 88]:
for cam_id in [16, 40, 64, 88]:
    mask_path = f"/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data_smplx/Hi4D/talk/talk01/000140/mask/000140_{cam_id:02d}.png"
    kpts_path = f"/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data_smplx/Hi4D/talk/talk01/000140/keypoints/000140_{cam_id:02d}_keypoints.json"

    mask = cv2.imread(mask_path)
    mask[mask == 125] = 250
    mask[mask == 255] = 125
    with open(kpts_path, "r") as fp:
        kpts_data = json.load(fp)['people']

    for kpts_d in kpts_data:
        idx = kpts_d['person_id']
        kpts = np.array(kpts_d['pose_keypoints_2d'], dtype=np.float32).reshape(25, 3)
        for kpt in kpts:
            if kpt[-1] > 0.0:
                cv2.circle(mask, (int(kpt[0]), int(kpt[1])), radius=8, color=colors[idx], thickness=-1)
        kpts = np.array(kpts_d['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[17 : 17 + 51, :]
        for kpt in kpts:
            if kpt[-1] > 0.0:
                cv2.circle(mask, (int(kpt[0]), int(kpt[1])), radius=8, color=colors[idx], thickness=-1)
        kpts = np.array(kpts_d['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
        for kpt in kpts:
            if kpt[-1] > 0.0:
                cv2.circle(mask, (int(kpt[0]), int(kpt[1])), radius=8, color=colors[idx], thickness=-1)
        kpts = np.array(kpts_d['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
        for kpt in kpts:
            if kpt[-1] > 0.0:
                cv2.circle(mask, (int(kpt[0]), int(kpt[1])), radius=8, color=colors[idx], thickness=-1)
    cv2.imwrite(f"test_{cam_id:02d}.png", mask)

    mask = cv2.imread(mask_path)
    mask[mask == 125] = 2
    mask[mask == 255] = 1
    mask_render = mask * 125
    with open(kpts_path, "r") as fp:
        kpts_data = json.load(fp)['people']

    for kpts_d in kpts_data:
        idx = kpts_d['person_id']
        kpts_pose = np.array(kpts_d['pose_keypoints_2d'], dtype=np.float32).reshape(25, 3)
        kpts_face = np.array(kpts_d['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[17 : 17 + 51, :]
        kpts_lhand = np.array(kpts_d['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
        kpts_rhand = np.array(kpts_d['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
        kpts = np.concatenate([kpts_pose, kpts_face, kpts_lhand, kpts_rhand], axis=0)
        kpts_mask = mask[kpts[:, 1].astype(np.int16), kpts[:, 0].astype(np.int16)]
        kpts[kpts_mask != (idx + 1)] = 0.

        for kpt in kpts:
            if kpt[-1] > 0.0:
                cv2.circle(mask_render, (int(kpt[0]), int(kpt[1])), radius=8, color=colors[idx], thickness=-1)
    cv2.imwrite(f"test_{cam_id:02d}_masked.png", mask_render)




# mask_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data/Hi4D/talk/talk01/000020/mask/000020_52.png"
# kpts_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data/Hi4D/talk/talk01/000020/keypoints/000020_52_keypoints.json"

# mask = cv2.imread(mask_path)
# with open(kpts_path, "r") as fp:
#     kpts_data = json.load(fp)['people']

# for idx, kpts in enumerate(kpts_data):
#     kpts = np.array(kpts['pose_keypoints_2d']).reshape(25, 3)
#     for kpt in kpts:
#         if kpt[-1] > 0.3:
#             cv2.circle(mask, (int(kpt[0]), int(kpt[1])), radius=8, color=colors[idx], thickness=-1)
# cv2.imwrite("test_1.png", mask)
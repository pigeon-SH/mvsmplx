import numpy as np
from PIL import Image
import json
import os
import cv2

colors = ((0, 0, 255), (255, 0, 0))

mask_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data/Hi4D/talk/talk01/000020/mask/000020_04.png"
kpts_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data/Hi4D/talk/talk01/000020/keypoints/000020_04_keypoints.json"

mask = cv2.imread(mask_path)
with open(kpts_path, "r") as fp:
    kpts_data = json.load(fp)['people']

for idx, kpts in enumerate(kpts_data):
    kpts = np.array(kpts['pose_keypoints_2d']).reshape(25, 3)
    for kpt in kpts:
        if kpt[-1] > 0.3:
            cv2.circle(mask, (int(kpt[0]), int(kpt[1])), radius=8, color=colors[idx], thickness=-1)
cv2.imwrite("test.png", mask)




mask_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data/Hi4D/talk/talk01/000020/mask/000020_52.png"
kpts_path = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data/Hi4D/talk/talk01/000020/keypoints/000020_52_keypoints.json"

mask = cv2.imread(mask_path)
with open(kpts_path, "r") as fp:
    kpts_data = json.load(fp)['people']

for idx, kpts in enumerate(kpts_data):
    kpts = np.array(kpts['pose_keypoints_2d']).reshape(25, 3)
    for kpt in kpts:
        if kpt[-1] > 0.3:
            cv2.circle(mask, (int(kpt[0]), int(kpt[1])), radius=8, color=colors[idx], thickness=-1)
cv2.imwrite("test_1.png", mask)
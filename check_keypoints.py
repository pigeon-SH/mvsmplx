import os
import cv2
import json
import numpy as np

IDENTITY = [np.array([255, 255, 255]), np.array([125, 125, 125])]

frames = list(range(20, 150))
cam_ids = [4, 16, 28, 40, 52, 64, 76, 88]
root = "data/Hi4D/talk/talk01"
for frame_name in frames:
    for cam_id in cam_ids:
        kpts_path = os.path.join(root, f"{frame_name:06d}", "keypoints", f"{frame_name:06d}_{cam_id:02d}_keypoints.json")
        with open(kpts_path, "r") as fp:
            kpts_data = json.load(fp)
        if len(kpts_data['people']) < 2:
            print(f"FRAME: {frame_name} CAM: {cam_id} PEOPLE: {len(kpts_data['people'])}")
        
        mask_path = os.path.join(root, f"{frame_name:06d}", "mask", f"{frame_name:06d}_{cam_id:02d}.png")
        mask = cv2.imread(mask_path)
        probs = []
        intersects = []
        for human_idx, human_data in enumerate(kpts_data['people']):
            kpts = np.array(human_data['pose_keypoints_2d']).reshape(25, 3)
            kpts = kpts[kpts[:, -1] > 0.3]
            kpts = np.floor(kpts).astype(np.int16)
            rgb = mask[kpts[:, 1], kpts[:, 0]]

            intersect = (rgb == IDENTITY[0]).sum(axis=0)[0]
            probs.append(intersect / len(kpts))
            intersect = (rgb == IDENTITY[1]).sum(axis=0)[0]
            probs.append(intersect / len(kpts))

            intersects.append((rgb == IDENTITY[0]).sum(axis=0)[0])
            intersects.append((rgb == IDENTITY[1]).sum(axis=0)[0])
            intersects.append(len(kpts))
        
        probs_prod = (probs[0] * probs[3], probs[1] * probs[2])
        print(intersects, probs, probs_prod)
        import sys
        sys.exit(-1)

        max_prob = max(probs)
        if len(probs) <= 2:
            if max_prob == probs[0]:
                kpts_data['people'][0]['person_id'] = 0
            else:
                kpts_data['people'][0]['person_id'] = 1
        else:
            if max_prob == probs[0]:
                kpts_data['people'][0]['person_id'] = 0
                kpts_data['people'][1]['person_id'] = 1
            elif max_prob == probs[1]:
                kpts_data['people'][0]['person_id'] = 1
                kpts_data['people'][1]['person_id'] = 0
            elif max_prob == probs[2]:
                kpts_data['people'][0]['person_id'] = 1
                kpts_data['people'][1]['person_id'] = 0
            elif max_prob == probs[3]:
                kpts_data['people'][0]['person_id'] = 0
                kpts_data['people'][1]['person_id'] = 1
        
        with open(kpts_path, "w") as fp:
            json.dump(kpts_data, fp, indent=4)
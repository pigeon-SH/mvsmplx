import os
import cv2
import json
import numpy as np
import itertools

# IDENTITY = [np.array([255, 255, 255]), np.array([125, 125, 125])]
IDENTITY = [np.array([28, 163, 255]), np.array([255, 120, 28])]

frames = list(range(1, 151))
# cam_ids = [4, 16, 28, 40, 52, 64, 76, 88]
cam_ids = [4, 28, 52, 76]
root = "data_smplx/Hi4D/backhug/backhug02"
for frame_name in frames:
    for cam_id in cam_ids:
        kpts_path = os.path.join(root, f"{frame_name:06d}", "keypoints_raw", f"{frame_name:06d}_{cam_id:02d}_keypoints.json")
        with open(kpts_path, "r") as fp:
            kpts_data = json.load(fp)
        if len(kpts_data['people']) == 2:
            print(f"FRAME: {frame_name:3d} CAM: {cam_id:2d} PEOPLE: {len(kpts_data['people'])}")
        
        mask_path = os.path.join(root, f"{frame_name:06d}", "mask_gt", f"{frame_name:06d}_{cam_id:02d}.png")
        mask = cv2.imread(mask_path)
        probs = []
        intersects = []

        if len(kpts_data['people']) == 1:
            human_data = kpts_data['people'][0]
            body_kpts = np.array(human_data['pose_keypoints_2d']).reshape(-1, 3)
            face_kpts = np.array(human_data['face_keypoints_2d']).reshape(-1, 3)
            lhand_kpts = np.array(human_data['hand_left_keypoints_2d']).reshape(-1, 3)
            rhand_kpts = np.array(human_data['hand_right_keypoints_2d']).reshape(-1, 3)
            kpts = np.concatenate([body_kpts, face_kpts, lhand_kpts, rhand_kpts], axis=0)
            kpts = kpts[kpts[:, -1] > 0.31]
            kpts = np.floor(kpts).astype(np.int16)
            rgb = mask[kpts[:, 1], kpts[:, 0]]

            intersect = (rgb == IDENTITY[0]).all(axis=-1).sum()
            prob0 = intersect / len(kpts)
            intersect = (rgb == IDENTITY[1]).all(axis=-1).sum()
            prob1 = intersect / len(kpts)
            if prob0 > prob1:
                kpts_data['people'][0]['person_id'] = 0
            else:
                kpts_data['people'][0]['person_id'] = 1
        else:
            for human_idx, human_data in enumerate(kpts_data['people']):
                body_kpts = np.array(human_data['pose_keypoints_2d']).reshape(-1, 3)
                face_kpts = np.array(human_data['face_keypoints_2d']).reshape(-1, 3)
                lhand_kpts = np.array(human_data['hand_left_keypoints_2d']).reshape(-1, 3)
                rhand_kpts = np.array(human_data['hand_right_keypoints_2d']).reshape(-1, 3)
                kpts = np.concatenate([body_kpts, face_kpts, lhand_kpts, rhand_kpts], axis=0)
                kpts = kpts[kpts[:, -1] > 0.3]
                kpts = np.floor(kpts).astype(np.int16)
                rgb = mask[kpts[:, 1], kpts[:, 0]]

                intersect = (rgb == IDENTITY[0]).all(axis=-1).sum()
                probs.append(intersect / len(kpts))
                intersect = (rgb == IDENTITY[1]).all(axis=-1).sum()
                probs.append(intersect / len(kpts))

                # intersects.append((rgb == IDENTITY[0]).sum(axis=0)[0])
                # intersects.append((rgb == IDENTITY[1]).sum(axis=0)[0])
                # intersects.append(len(kpts))
                kpts_data['people'][human_idx]['person_id'] = -1
            
            if probs[0] * probs[1] == 0:
                if probs[2] > probs[3]:
                    kpts_data['people'][1]['person_id'] = 0
                else:
                    kpts_data['people'][1]['person_id'] = 1
                kpts_data['people'] = [kpts_data['people'][1]]
            elif probs[2] * probs[3] == 0:
                if probs[0] > probs[1]:
                    kpts_data['people'][0]['person_id'] = 0
                else:
                    kpts_data['people'][0]['person_id'] = 1
                kpts_data['people'] = [kpts_data['people'][0]]
            else:
                if probs[0] * probs[3] > probs[1] * probs[2]:
                    kpts_data['people'][0]['person_id'] = 0
                    kpts_data['people'][1]['person_id'] = 1
                else:
                    kpts_data['people'][0]['person_id'] = 1
                    kpts_data['people'][1]['person_id'] = 0
            # num_people = len(kpts_data['people'])
            # num_person_ids = len(IDENTITY)
            # all_combinations = list(itertools.permutations(range(num_people), num_person_ids))

            # max_prob = 0
            # best_combination = None
            # for combination in all_combinations:
            #     current_prob = 1.0
            #     for i, people_id in enumerate(combination):
            #         current_prob *= probs[people_id * num_person_ids + i]
            #     if current_prob > max_prob:
            #         max_prob = current_prob
            #         best_combination = combination
            # if best_combination:
            #     for i, people_id in enumerate(best_combination):
            #         kpts_data['people'][people_id]['person_id'] = i
            
            # # kpts_data['people'] = kpts_data['people'][:2]
            # kpts_data['people'] = [person for person in kpts_data['people'] if person['person_id'] != -1]
            # if len(kpts_data['people']) < 2:
            #     breakpoint()
            #     print()

        save_path = os.path.join(root, f"{frame_name:06d}", "keypoints_openpose", f"{frame_name:06d}_{cam_id:02d}_keypoints.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as fp:
            json.dump(kpts_data, fp, indent=4)
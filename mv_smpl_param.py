import os
import cv2
import json
import numpy as np
import pickle

IDENTITY = [np.array([255, 255, 255]), np.array([125, 125, 125])]

seqs = ["talk/talk22", "backhug/backhug02"] # 
for seq in seqs:
    src_root = os.path.join("/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/1015_temporal_8view", seq)
    # src_root = os.path.join("/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/0925_temporal_single_maskpred_nokptsmask")
    dst_root = os.path.join("/home/vclab/8T_SSD1/dataset/Hi4D", seq, "smpl_pred_8view")
    frame_names = sorted(os.listdir(src_root))
    frame_names = [frame_name for frame_name in frame_names if os.path.isdir(os.path.join(src_root, frame_name))]
    os.makedirs(dst_root, exist_ok=True)

    for frame_name in frame_names:
        dst_path = os.path.join(dst_root, f"{frame_name}.npz")
        dst_param = {}
        betas = []
        global_orients = []
        left_hand_poses = []
        right_hand_poses = []
        jaw_poses = []
        leye_poses = []
        reye_poses = []
        expressions = []
        transls = []
        body_poses = []
        src_path = os.path.join(src_root, frame_name, "smpl_param.pkl")
        with open(src_path, "rb") as fp:
            src_params = pickle.load(fp)
        src_params.sort(key = lambda x: x['person_id'])
        for human_idx in range(len(src_params)):
            betas.append(src_params[human_idx]['result']['betas']) # (1, 300)
            global_orients.append(src_params[human_idx]['result']['global_orient']) # (1, 3)
            left_hand_poses.append(src_params[human_idx]['result']['left_hand_pose']) # (1, 12)
            right_hand_poses.append(src_params[human_idx]['result']['right_hand_pose']) # (1, 12)
            jaw_poses.append(src_params[human_idx]['result']['jaw_pose']) # (1, 3)
            leye_poses.append(src_params[human_idx]['result']['leye_pose']) # (1, 3)
            reye_poses.append(src_params[human_idx]['result']['reye_pose']) # (1, 3)
            expressions.append(src_params[human_idx]['result']['expression']) # (1, 100)
            transls.append(src_params[human_idx]['result']['global_body_translation'][None]) # (1, 3)
            body_poses.append(src_params[human_idx]['result']['body_pose'][:, 3:]) # (1, 63)
        dst_param['betas'] = np.concatenate(betas, axis=0)  # (2, 10)
        dst_param['global_orient'] = np.concatenate(global_orients, axis=0) # (2, 3)
        dst_param['left_hand_pose'] = np.concatenate(left_hand_poses, axis=0) # (2, 12)
        dst_param['right_hand_pose'] = np.concatenate(right_hand_poses, axis=0) # (2, 12)
        dst_param['jaw_pose'] = np.concatenate(jaw_poses, axis=0) # (2, 3)
        dst_param['leye_pose'] = np.concatenate(leye_poses, axis=0) # (2, 3)
        dst_param['reye_pose'] = np.concatenate(reye_poses, axis=0) # (2, 3)
        dst_param['expression'] = np.concatenate(expressions, axis=0) # (2, 100)
        dst_param['transl'] = np.concatenate(transls, axis=0) # (2, 3)
        dst_param['body_pose'] = np.concatenate(body_poses, axis=0) # (2, 63)
        np.savez(dst_path, **dst_param)
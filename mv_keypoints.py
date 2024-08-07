import os
from shutil import copyfile

num_frame = 130
num_human = 2
cam_ids = [4, 16, 28, 40]
keypoint_src_root = "/home/vclab/8T_SSD1/extractSMPL/slahmr/data/Hi4D/talk/talk01/slahmr/track_preds/"
save_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data/Hi4D/talk/talk01/keypoints"

for frame_idx in range(num_frame):
    for human_idx in range(num_human):
        for cam_id in cam_ids:
            save_dir = os.path.join(save_root, f"{human_idx+1}", f"{frame_idx+1:06d}")
            os.makedirs(save_dir, exist_ok=True)
            src_path = os.path.join(keypoint_src_root, str(cam_id), f"{human_idx+1:03d}", f"{frame_idx+1:06d}_keypoints.json")
            dst_path = os.path.join(save_dir, f"{cam_id:02d}_keypoints.json")
            copyfile(src_path, dst_path)
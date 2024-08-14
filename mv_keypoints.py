import os
from shutil import copyfile

# num_frame = 130
# num_human = 2
# cam_ids = [4, 16, 28, 40]
# keypoint_src_root = "/home/vclab/8T_SSD1/extractSMPL/slahmr/data/Hi4D/talk/talk01/slahmr/track_preds/"
# save_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data/Hi4D/talk/talk01/keypoints"

# for frame_idx in range(num_frame):
#     for human_idx in range(num_human):
#         for cam_id in cam_ids:
#             save_dir = os.path.join(save_root, f"{human_idx+1}", f"{frame_idx+1:06d}")
#             os.makedirs(save_dir, exist_ok=True)
#             src_path = os.path.join(keypoint_src_root, str(cam_id), f"{human_idx+1:03d}", f"{frame_idx+1:06d}_keypoints.json")
#             dst_path = os.path.join(save_dir, f"{cam_id:02d}_keypoints.json")
#             copyfile(src_path, dst_path)


num_frame = 130
frames = list(range(20, 150))
num_human = 2
cam_ids = [4, 16, 28, 40, 52, 64, 76, 88]
keypoint_src_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/dd/output"
save_root = "/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data/Hi4D/talk/talk01"

for frame_name in frames:
    for cam_id in cam_ids:
        save_dir = os.path.join(save_root, f"{frame_name:06d}", "keypoints")
        os.makedirs(save_dir, exist_ok=True)
        src_path = os.path.join(keypoint_src_root, f"{frame_name:06d}_{cam_id}_keypoints.json")
        dst_path = os.path.join(save_dir, f"{frame_name:06d}_{cam_id:02d}_keypoints.json")
        copyfile(src_path, dst_path)

img_src_root = "/home/vclab/dataset/Hi4D/talk/talk01/images"
for frame_name in frames:
    for cam_id in cam_ids:
        save_dir = os.path.join(save_root, f"{frame_name:06d}", "color")
        os.makedirs(save_dir, exist_ok=True)
        src_path = os.path.join(img_src_root, str(cam_id), f"{frame_name:06d}.jpg")
        dst_path = os.path.join(save_dir, f"{frame_name:06d}_{cam_id:02d}.jpg")
        copyfile(src_path, dst_path)

        os.makedirs(os.path.join(save_root, f"{frame_name:06d}", "meta"), exist_ok=True)

mask_src_root = "/home/vclab/dataset/Hi4D/talk/talk01/masks"
for frame_name in frames:
    for cam_id in cam_ids:
        save_dir = os.path.join(save_root, f"{frame_name:06d}", "mask")
        os.makedirs(save_dir, exist_ok=True)
        src_path = os.path.join(mask_src_root, str(cam_id), f"{frame_name:06d}.png")
        dst_path = os.path.join(save_dir, f"{frame_name:06d}_{cam_id:02d}.png")
        copyfile(src_path, dst_path)
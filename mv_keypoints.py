import os
from shutil import copyfile
from coco2openpose import convert

frames = list(range(71, 151))
num_human = 2
cam_ids = [4, 28, 52, 76]
# cam_ids = [16, 40, 64, 88]
seq = "backhug/backhug02" # "talk/talk22"
data_root = os.path.join("/home/vclab/8T_SSD1/dataset/Hi4D", seq)
keypoint_src_root = os.path.join(data_root, "kpts2d/sapiens_refined")
# keypoint_src_root = os.path.join(data_root, "kpts2d/openpose")
save_root = os.path.join("/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/data_smplx/Hi4D/", seq)

# for frame_name in frames:
#     for cam_id in cam_ids:
#         save_dir = os.path.join(save_root, f"{frame_name:06d}", "keypoints_coco")
#         os.makedirs(save_dir, exist_ok=True)
#         src_path = os.path.join(keypoint_src_root, str(cam_id), f"{frame_name:06d}.json")
#         dst_path = os.path.join(save_dir, f"{frame_name:06d}_{cam_id:02d}_keypoints.json")
#         copyfile(src_path, dst_path)

for frame_name in frames:
    for cam_id in cam_ids:
        save_dir = os.path.join(save_root, f"{frame_name:06d}", "keypoints_refined")
        os.makedirs(save_dir, exist_ok=True)
        src_path = os.path.join(save_root, f"{frame_name:06d}", "keypoints_coco", f"{frame_name:06d}_{cam_id:02d}_keypoints.json")
        dst_path = os.path.join(save_dir, f"{frame_name:06d}_{cam_id:02d}_keypoints.json")
        convert(dst_path, src_path, ordered=True)

# for frame_name in frames:
#     for cam_id in cam_ids:
#         save_dir = os.path.join(save_root, f"{frame_name:06d}", "keypoints_raw")
#         os.makedirs(save_dir, exist_ok=True)
#         src_path = os.path.join(keypoint_src_root, str(cam_id), f"{frame_name:06d}_keypoints.json")
#         dst_path = os.path.join(save_dir, f"{frame_name:06d}_{cam_id:02d}_keypoints.json")
#         copyfile(src_path, dst_path)

img_src_root = os.path.join(data_root, "images")
for frame_name in frames:
    for cam_id in cam_ids:
        save_dir = os.path.join(save_root, f"{frame_name:06d}", "color")
        os.makedirs(save_dir, exist_ok=True)
        src_path = os.path.join(img_src_root, str(cam_id), f"{frame_name:06d}.jpg")
        dst_path = os.path.join(save_dir, f"{frame_name:06d}_{cam_id:02d}.jpg")
        copyfile(src_path, dst_path)

mask_src_root = os.path.join(data_root, "seg/img_seg_mask")
for frame_name in frames:
    for cam_id in cam_ids:
        save_dir = os.path.join(save_root, f"{frame_name:06d}", "mask_gt")
        os.makedirs(save_dir, exist_ok=True)
        src_path = os.path.join(mask_src_root, str(cam_id), "all", f"{frame_name:06d}.png")
        dst_path = os.path.join(save_dir, f"{frame_name:06d}_{cam_id:02d}.png")
        copyfile(src_path, dst_path)

mask_src_root = os.path.join(data_root, "mask_refined")
for frame_name in frames:
    for cam_id in cam_ids:
        save_dir = os.path.join(save_root, f"{frame_name:06d}", "mask_refined")
        os.makedirs(save_dir, exist_ok=True)
        src_path = os.path.join(mask_src_root, str(cam_id), f"{frame_name:06d}.png")
        dst_path = os.path.join(save_dir, f"{frame_name:06d}_{cam_id:02d}.png")
        copyfile(src_path, dst_path)
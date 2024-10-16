import os
from shutil import copyfile
from coco2openpose import convert

frames = list(range(1, 71))
num_human = 2
cam_ids = [4, 28, 52, 76]
seqs = ["hug/hug13", "talk/talk22"]
for seq in seqs:
    src_root = os.path.join("/home/vclab/8T_SSD1/extractSMPL/MultiviewSMPLifyX/result/0926_temporal_single", seq)
    dst_root = os.path.join("/home/vclab/8T_SSD1/dataset/Hi4D", seq, "smpl_pred")
    frame_names = sorted(os.listdir(src_root))
    for frame_name in frame_names:
        src_path = os.path.join(src_root, frame_name, "smpl_param.pkl")
        dst_path = os.path.join(dst_root, f"{frame_name}.pkl")
        copyfile(src_path, dst_path)
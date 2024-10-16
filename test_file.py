import os
log_root = "result/0924_kptsmask_single_maskpred"
frames = list(range(1, 151, 2))
for frame in frames:
    path = os.path.join(log_root, f"{frame:06d}", "smpl_mesh_total.obj")
    if not os.path.exists(path):
        print(frame)
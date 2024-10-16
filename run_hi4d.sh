GPUIDX=1
seq="talk/talk22"
log="0926_temporal_single"
for frame_idx in {1..71}
do
    formatted_frame=$(printf "%06d" $frame_idx)
    CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/$seq/$formatted_frame --output_folder ./result/$log/$seq/$formatted_frame --keyp_folder keypoints_single --mask_folder mask_pred --temporal_consist True
done

# log="0924_ranking_single_maskpred_nokptsmask"
# for frame_idx in {1..150..2}
# do
#     formatted_frame=$(printf "%06d" $frame_idx)
#     CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/$seq/$formatted_frame --output_folder ./result/$log/$formatted_frame --keyp_folder keypoints_single --mask_folder mask_pred --temporal_consist True --collision_loss True
# done

# log="0924_ranking_single_maskpred"
# for frame_idx in {23..150..2}
# do
#     formatted_frame=$(printf "%06d" $frame_idx)
#     CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/$seq/$formatted_frame --output_folder ./result/$log/$formatted_frame --keyp_folder keypoints_single --mask_folder mask_pred --kpts_filter_mask True --temporal_consist True --collision_loss True
# done

# log="0924_default_single_maskpred"
# for frame_idx in {1..150..2}
# do
#     formatted_frame=$(printf "%06d" $frame_idx)
#     CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/$seq/$formatted_frame --output_folder ./result/$log/$formatted_frame --keyp_folder keypoints_single --mask_folder mask_pred
# done

# log="0924_kptsmask_single_maskpred"
# for frame_idx in {1..150..2}
# do
#     formatted_frame=$(printf "%06d" $frame_idx)
#     CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/$seq/$formatted_frame --output_folder ./result/$log/$formatted_frame --keyp_folder keypoints_single --mask_folder mask_pred --kpts_filter_mask True
# done

# log="0924_temporal_single_maskpred"
# for frame_idx in {1..150..2}
# do
#     formatted_frame=$(printf "%06d" $frame_idx)
#     CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/$seq/$formatted_frame --output_folder ./result/$log/$formatted_frame --keyp_folder keypoints_single --mask_folder mask_pred --kpts_filter_mask True --temporal_consist True
# done

# log="0924_ranking_single_maskpred"
# for frame_idx in {1..150..2}
# do
#     formatted_frame=$(printf "%06d" $frame_idx)
#     CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/$seq/$formatted_frame --output_folder ./result/$log/$formatted_frame --keyp_folder keypoints_single --mask_folder mask_pred --kpts_filter_mask True --temporal_consist True --collision_loss True
# done
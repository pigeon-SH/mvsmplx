GPUIDX=1
seq="backhug/backhug02"
log="0922_ranking_openpose"
for frame_idx in {1..150..2}
do
    formatted_frame=$(printf "%06d" $frame_idx)
    CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/$seq/$formatted_frame --output_folder ./result/$log/$formatted_frame --keyp_folder keypoints_openpose --kpts_filter_mask True --temporal_consist True --collision_loss True
done

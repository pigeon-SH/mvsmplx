GPUIDX=1
seq="backhug/backhug02"
for frame_idx in {1..150}
do
    formatted_frame=$(printf "%06d" $frame_idx)
    CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/$seq/$formatted_frame --output_folder ./result/0921_default_openpose/$formatted_frame --keyp_folder keypoints_openpose
done

GPUIDX=0
# for frame_idx in {20..149}
for frame_idx in {140..145}
do
    formatted_frame=$(printf "%06d" $frame_idx)
    CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smpl.yaml --data_folder ./data/Hi4D/talk/talk01/$formatted_frame --output_folder ./result_interpene_scaled_iter60/$formatted_frame/smpl
done

GPUIDX=0
for frame_idx in {20..149}
do
    formatted_frame=$(printf "%06d" $frame_idx)
    for human_idx in 0 1
    do
        CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smpl.yaml --data_folder ./data/Hi4D/talk/talk01/$formatted_frame --output_folder ./data/Hi4D/talk/talk01/$formatted_frame/smpl_$human_idx --human_idx $human_idx
    done
done

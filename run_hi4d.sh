GPUIDX=0
for frame_idx in 142
# for frame_idx in {140..145}
do
    formatted_frame=$(printf "%06d" $frame_idx)
    CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smplx.yaml --data_folder ./data_smplx/Hi4D/talk/talk01/$formatted_frame --output_folder ./result_xmplx_nointer_8view/$formatted_frame/smpl
done

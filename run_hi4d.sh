GPUIDX=0
for frame_idx in {1..130}
do
    CUDA_VISIBLE_DEVICES=$GPUIDX python main.py --config cfg_files/fit_smpl.yaml --data_folder ./dataset_example/image_data/Hi4D/talk/talk01/$frame_idx --output_folder ./dataset_example/mesh_data/Hi4D/talk/talk01/$frame_idx/smpl
done

# The main_smplx_to_263dim_dump_data.py is the latest implementation of converting SMPL-X to SMPL and 263-dim motion representation.
# The code is refactored to be more modular and efficient from the main_smplx_to_263dim.py.

python main_smplx_to_263dim_dump_data.py \
    --root_dir /work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence \
    --out_dir /work/hdd/bfyo/benk/hhsu2/imu-humans/final_data_per_sequence \
    --dataset_name MotionUnion/humanml \
    --start_idx 100 \
    --end_idx 200 \
    --max_workers 4
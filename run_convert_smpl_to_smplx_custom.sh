
# MobilePoser
python main_smpl_to_smplx_custom.py \
    --pred_data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/MobilePoser/predictions/lingo/global \
    --gt_data_dir /home/haoyuyh3/Downloads/lingo_smplx_files/motion_data/test \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_mobileposer/lingo_global/smplx_gt_cvt \
    --max_frames 60 \
    --mode mobileposer

python main_smpl_to_smplx_custom.py \
    --pred_data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/MobilePoser/predictions/lingo/lw_rp_h \
    --gt_data_dir /home/haoyuyh3/Downloads/lingo_smplx_files/motion_data/test \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_mobileposer/lingo_lw_rp_h/smplx_gt_cvt \
    --max_frames 60 \
    --mode mobileposer


# IMUPoser
python main_smpl_to_smplx_custom.py \
    --pred_data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/predictions/imuposer/lingo_global \
    --gt_data_dir /home/haoyuyh3/Downloads/lingo_smplx_files/motion_data/test \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_imuposer/lingo_global/smplx_gt_cvt \
    --max_frames 60 \
    --mode imuposer

python main_smpl_to_smplx_custom.py \
    --pred_data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/predictions/imuposer/lingo_lw_rp_h \
    --gt_data_dir /home/haoyuyh3/Downloads/lingo_smplx_files/motion_data/test \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_imuposer/lingo_lw_rp_h/smplx_gt_cvt \
    --max_frames 60 \
    --mode imuposer


# BoDiffusion
python main_smpl_to_smplx_custom.py \
    --pred_data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/BoDiffusion/results_100000/BoDiffusion/preds \
    --gt_data_dir /home/haoyuyh3/Downloads/lingo_smplx_files/motion_data/test \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_bodiffusion/lingo/smplx_gt_cvt \
    --max_frames 60 \
    --mode bodiffusion


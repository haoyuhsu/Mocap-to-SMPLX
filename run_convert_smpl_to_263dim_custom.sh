# We only need 'global' IMU config to evaluate text predictions

# IMUPoser
python main_smpl_to_263dim_pred_imuposer.py \
    --data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/predictions/imuposer/lingo_global \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_imuposer/lingo_global/263d

python main_smpl_to_263dim_pred_imuposer.py \
    --data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/predictions/imuposer/humanml_global \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_imuposer/humanml_global/263d


# MobilePoser
python main_smpl_to_263dim_pred_imuposer.py \
    --data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/MobilePoser/predictions/lingo/global \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_mobileposer/lingo_global/263d

python main_smpl_to_263dim_pred_imuposer.py \
    --data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/MobilePoser/predictions/humanml/global \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_mobileposer/humanml_global/263d

python main_smpl_to_263dim_pred_imuposer.py \
    --data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/MobilePoser/predictions/humoto/global \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_mobileposer/humoto_global/263d

python main_smpl_to_263dim_pred_imuposer.py \
    --data_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/MobilePoser/predictions/parahome/global \
    --out_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_mobileposer/parahome_global/263d

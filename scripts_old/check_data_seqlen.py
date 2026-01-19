import numpy as np
import os
from tqdm import tqdm

humanml_data_dir = '/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_263dim'

seq_len_list = []
for f_name in tqdm(os.listdir(humanml_data_dir)):
    if f_name.endswith('.npy'):
        data = np.load(os.path.join(humanml_data_dir, f_name))
        # print(f"{f_name}: shape = {data.shape}, dtype = {data.dtype}")
        seq_len_list.append(data.shape[0])
print("HumanML3D data average sequence length:", np.mean(seq_len_list)) 


humoto_data_dir = '/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humoto_263dim'

seq_len_list = []
for f_name in tqdm(os.listdir(humoto_data_dir)):
    if f_name.endswith('.npy'):
        data = np.load(os.path.join(humoto_data_dir, f_name))
        # print(f"{f_name}: shape = {data.shape}, dtype = {data.dtype}")
        seq_len_list.append(data.shape[0])
print("HUMOTO data average sequence length:", np.mean(seq_len_list))
from common.quaternion import cont6d_to_matrix
import torch
from convert_from_85_to_263 import recover_root_rot_pos, quaternion_to_cont6d
import os
import numpy as np
import pickle
from tqdm import tqdm


# NOTE: this is SMPL format instead of SMPL-X


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)


def extract_smpl_from_motion_vector(data, joints_num=22):
    """
    Extract SMPL-compatible parameters directly from motion vector.
    
    Args:
        data: Motion vector numpy array (seq_len, 263)
        joints_num: Number of joints
        
    Returns:
        Dictionary with SMPL parameters
    """
    # Root rotation from velocity
    r_rot_quat, r_pos = recover_root_rot_pos(torch.from_numpy(data))
    
    # Extract continuous 6D rotation params
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    
    # Get root rotation
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    
    # Get body rotations
    cont6d_params = torch.from_numpy(data[..., start_indx:end_indx])
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    
    # Convert to rotation matrices then to axis-angle
    # Note: You'll need to implement these conversions
    rot_mats = cont6d_to_matrix(cont6d_params)  # (seq_len, 22, 3, 3)
    # axis_angles = matrix_to_axis_angle(rot_mats)  # (seq_len, 22, 3)

    from scipy.spatial.transform import Rotation as R
    axis_angles = R.from_matrix(rot_mats.numpy().reshape(-1, 3, 3)).as_rotvec().reshape(-1, joints_num, 3)
    
    return {
        'global_orient': axis_angles[:, 0],  # (seq_len, 3)
        'body_pose': axis_angles[:, 1:22].reshape(-1, 63),   # (seq_len, 21, 3)
        'transl': r_pos.cpu().numpy()                       # (seq_len, 3)
    }


if __name__ == "__main__":


    root_dir = '/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_263dim'

    seq_names_list = [f for f in sorted(os.listdir(root_dir)) if f.endswith('.npy')]

    out_dir = '/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/_lingo_smpl_85_from_263dim'
    os.makedirs(out_dir, exist_ok=True)

    for seq_name in tqdm(seq_names_list):
        data = np.load(os.path.join(root_dir, seq_name))

        smpl_params = extract_smpl_from_motion_vector(data, joints_num=22)

        out_path = os.path.join(out_dir, seq_name.replace('.npy', '.pkl'))
        with open(out_path, 'wb') as f:
            pickle.dump(smpl_params, f)
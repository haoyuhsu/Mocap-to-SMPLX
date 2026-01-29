import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d


def upsample_motion(input_path, output_path, input_fps=30, output_fps=60):
    """
    Upsample motion data from input_fps to output_fps using interpolation.
    
    Args:
        input_path: Path to input pickle file
        output_path: Path to save upsampled pickle file
        input_fps: Original frame rate (default: 30)
        output_fps: Target frame rate (default: 60)
    """
    # Load motion data
    with open(input_path, 'rb') as f:
        motion_data = pickle.load(f)
    
    # Get number of frames
    n_frames = motion_data['global_orient'].shape[0]
    
    # Calculate the duration of the sequence
    duration = (n_frames - 1) / input_fps  # in seconds
    
    # Calculate number of frames needed for output FPS
    n_upsampled = int(duration * output_fps) + 1
    
    # Create time arrays for interpolation
    t_original = np.arange(n_frames) / input_fps  # time in seconds
    t_upsampled = np.linspace(0, duration, n_upsampled)  # time in seconds
    
    upsampled_data = {}
    
    # === Upsample translation (linear interpolation is fine) ===
    transl = motion_data['transl']  # (N, 3)
    interpolator = interp1d(t_original, transl, axis=0, kind='linear', fill_value='extrapolate')
    upsampled_data['transl'] = interpolator(t_upsampled)
    
    # === Upsample global_orient using SLERP ===
    global_orient = motion_data['global_orient']  # (N, 3) axis-angle
    
    # Convert axis-angle to rotation objects
    rotations = R.from_rotvec(global_orient)
    
    # Create SLERP interpolator
    slerp = Slerp(t_original, rotations)
    
    # Interpolate and convert back to axis-angle
    upsampled_rotations = slerp(t_upsampled)
    upsampled_data['global_orient'] = upsampled_rotations.as_rotvec()
    
    # === Upsample body_pose using SLERP for each joint ===
    body_pose = motion_data['body_pose']  # (N, 23, 3) axis-angle
    n_joints = body_pose.shape[1]
    body_pose_upsampled = np.zeros((n_upsampled, n_joints, 3))
    
    for joint_idx in range(n_joints):
        # Get rotations for this joint across all frames
        joint_rotations = body_pose[:, joint_idx, :]  # (N, 3)
        
        # Convert to rotation objects
        rotations = R.from_rotvec(joint_rotations)
        
        # Create SLERP interpolator for this joint
        slerp = Slerp(t_original, rotations)
        
        # Interpolate and convert back to axis-angle
        upsampled_rotations = slerp(t_upsampled)
        body_pose_upsampled[:, joint_idx, :] = upsampled_rotations.as_rotvec()
    
    upsampled_data['body_pose'] = body_pose_upsampled
    
    # === Betas don't need upsampling (body shape parameters) ===
    upsampled_data['betas'] = motion_data['betas']
    
    # Save upsampled motion
    with open(output_path, 'wb') as f:
        pickle.dump(upsampled_data, f)
    
    print(f"Upsampled motion:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Input: {n_frames} frames at {input_fps} FPS")
    print(f"  Output: {n_upsampled} frames at {output_fps} FPS")
    print(f"  - global_orient: {global_orient.shape} -> {upsampled_data['global_orient'].shape}")
    print(f"  - body_pose: {body_pose.shape} -> {upsampled_data['body_pose'].shape}")
    print(f"  - transl: {transl.shape} -> {upsampled_data['transl'].shape}")


if __name__ == "__main__":

    input_dir = '/home/haoyuyh3/Documents/maxhsu/imu-humans/Mocap-to-SMPLX/_gt_120f_lingo_smpl_from_smplx'          # 30 FPS
    output_dir = '/home/haoyuyh3/Documents/maxhsu/imu-humans/Mocap-to-SMPLX/_gt_240f_60fps_lingo_smpl_from_smplx'   # 60 FPS
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.pkl')])

    for file_name in file_list:
        input_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        upsample_motion(input_file, output_file, input_fps=30, output_fps=60)

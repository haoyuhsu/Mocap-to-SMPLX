"""
Clean implementation for converting raw motion format to motion vectors.
For HumanML3D Dataset processing.
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from common.skeleton import Skeleton
from common.quaternion import (
    qbetween_np, qrot_np, qmul_np, qinv_np, qfix, qrot, qinv,
    quaternion_to_cont6d_np, quaternion_to_cont6d
)
from paramUtil import t2m_raw_offsets, t2m_kinematic_chain


class MotionConverter:
    """Converts raw motion data to motion vector representation."""
    
    def __init__(self, example_data_path, joints_num=22):
        """
        Initialize the converter with skeleton configuration.
        
        Args:
            example_data_path: Path to example motion file for skeleton reference
            joints_num: Number of joints in the skeleton (default: 22)
        """
        self.joints_num = joints_num
        self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        self.kinematic_chain = t2m_kinematic_chain
        
        # Joint indices
        self.l_idx1, self.l_idx2 = 5, 8  # Lower legs
        self.fid_r, self.fid_l = [8, 11], [7, 10]  # Right/Left foot
        self.face_joint_indx = [2, 1, 17, 16]  # Face direction: r_hip, l_hip, sdr_r, sdr_l
        
        # Get target skeleton offsets from example data
        self.tgt_offsets = self._get_target_offsets(example_data_path)
    
    def _get_target_offsets(self, example_path):
        """Extract target skeleton offsets from example data."""
        example_data = np.load(example_path)
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, 'cpu')
        return tgt_skel.get_offsets_joints(example_data[0])
    
    def uniform_skeleton(self, positions):
        """
        Normalize skeleton to uniform scale based on leg length.
        
        Args:
            positions: Joint positions (seq_len, joints_num, 3)
        
        Returns:
            Normalized joint positions
        """
        src_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, 'cpu')
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
        src_offset = src_offset.numpy()
        tgt_offset = self.tgt_offsets.numpy()
        
        # Calculate scale ratio based on leg length
        src_leg_len = np.abs(src_offset[self.l_idx1]).max() + np.abs(src_offset[self.l_idx2]).max()
        tgt_leg_len = np.abs(tgt_offset[self.l_idx1]).max() + np.abs(tgt_offset[self.l_idx2]).max()
        scale_rt = tgt_leg_len / src_leg_len
        
        # Scale root position
        src_root_pos = positions[:, 0]
        tgt_root_pos = src_root_pos * scale_rt
        
        # Inverse and forward kinematics for retargeting
        quat_params = src_skel.inverse_kinematics_np(positions, self.face_joint_indx)
        src_skel.set_offset(self.tgt_offsets)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        
        return new_joints
    
    def detect_foot_contact(self, positions, threshold=0.002):
        """
        Detect foot contact with ground based on velocity threshold.
        
        Args:
            positions: Joint positions (seq_len, joints_num, 3)
            threshold: Velocity threshold for contact detection
        
        Returns:
            feet_l, feet_r: Binary contact masks for left and right feet
        """
        velfactor = np.array([threshold, threshold])
        
        # Left foot velocity
        feet_l_x = (positions[1:, self.fid_l, 0] - positions[:-1, self.fid_l, 0]) ** 2
        feet_l_y = (positions[1:, self.fid_l, 1] - positions[:-1, self.fid_l, 1]) ** 2
        feet_l_z = (positions[1:, self.fid_l, 2] - positions[:-1, self.fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)
        
        # Right foot velocity
        feet_r_x = (positions[1:, self.fid_r, 0] - positions[:-1, self.fid_r, 0]) ** 2
        feet_r_y = (positions[1:, self.fid_r, 1] - positions[:-1, self.fid_r, 1]) ** 2
        feet_r_z = (positions[1:, self.fid_r, 2] - positions[:-1, self.fid_r, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
        
        return feet_l, feet_r
    
    def normalize_pose(self, positions):
        """
        Normalize pose: put on floor, center at origin, face Z+.
        
        Args:
            positions: Joint positions (seq_len, joints_num, 3)
        
        Returns:
            Normalized positions and initial root quaternion
        """
        # Put on floor
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height
        
        # XZ at origin
        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz
        
        # Align to face Z+
        r_hip, l_hip, sdr_r, sdr_l = self.face_joint_indx
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
        
        # Calculate forward direction
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
        
        # Rotation to align with Z+
        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
        
        positions = qrot_np(root_quat_init, positions)
        
        return positions
    
    def compute_cont6d_params(self, positions):
        """
        Compute continuous 6D rotation parameters and velocities.
        
        Args:
            positions: Joint positions (seq_len, joints_num, 3)
        
        Returns:
            cont_6d_params: Continuous 6D rotation parameters
            r_velocity: Root angular velocity
            velocity: Root linear velocity
            r_rot: Root rotations
        """
        skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
        
        # Inverse kinematics to get quaternions
        quat_params = skel.inverse_kinematics_np(positions, self.face_joint_indx, smooth_forward=True)
        
        # Convert to continuous 6D representation
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        
        # Root rotation
        r_rot = quat_params[:, 0].copy()
        
        # Root linear velocity
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        
        # Root angular velocity
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        
        return cont_6d_params, r_velocity, velocity, r_rot
    
    def compute_local_positions(self, positions, r_rot):
        """
        Compute rotation-invariant local positions.
        
        Args:
            positions: Joint positions (seq_len, joints_num, 3)
            r_rot: Root rotations (seq_len, 4)
        
        Returns:
            Local positions relative to root
        """
        # Center at root
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        
        # Rotate to face Z+
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        
        return positions
    
    def convert_to_motion_vector(self, positions, feet_threshold=0.002):
        """
        Convert raw joint positions to motion vector representation.
        
        Args:
            positions: Raw joint positions (seq_len, joints_num, 3)
            feet_threshold: Threshold for foot contact detection
        
        Returns:
            motion_vector: Concatenated motion representation (seq_len, feature_dim)
            global_positions: Global joint positions after normalization
            local_positions: Local joint positions
            l_velocity: Local velocity
        """
        # Step 1: Uniform skeleton
        positions = self.uniform_skeleton(positions)
        
        # Step 2: Normalize pose
        positions = self.normalize_pose(positions)
        global_positions = positions.copy()
        
        # Step 3: Detect foot contacts
        feet_l, feet_r = self.detect_foot_contact(positions, feet_threshold)
        
        # Step 4: Compute continuous 6D parameters and velocities
        cont_6d_params, r_velocity, velocity, r_rot = self.compute_cont6d_params(positions)
        
        # Step 5: Compute local positions
        positions = self.compute_local_positions(positions, r_rot)
        
        # Step 6: Build motion vector components
        # Root height
        root_y = positions[:, 0, 1:2]
        
        # Root rotation velocity (y-axis) and linear velocity (xz plane)
        r_velocity_y = np.arcsin(r_velocity[:, 2:3])
        l_velocity = velocity[:, [0, 2]]
        root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)
        
        # Joint rotations (continuous 6D, excluding root)
        rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
        
        # Joint positions (rotation-invariant, excluding root)
        ric_data = positions[:, 1:].reshape(len(positions), -1)
        
        # Local velocities
        local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
        local_vel = local_vel.reshape(len(local_vel), -1)
        
        # Concatenate all features
        motion_vector = np.concatenate([
            root_data,
            ric_data[:-1],
            rot_data[:-1],
            local_vel,
            feet_l,
            feet_r
        ], axis=-1)
        
        return motion_vector, global_positions, positions, l_velocity


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def process_dataset(data_dir, save_dir_joints, save_dir_vecs, example_id="000021"):
    """
    Process entire motion dataset.
    
    Args:
        data_dir: Directory containing raw motion files
        save_dir_joints: Output directory for joint positions
        save_dir_vecs: Output directory for motion vectors
        example_id: ID of example file for skeleton reference
    """
    from paramUtil import recover_from_ric
    
    # Setup directories
    data_dir = Path(data_dir)
    save_dir_joints = Path(save_dir_joints)
    save_dir_vecs = Path(save_dir_vecs)
    save_dir_joints.mkdir(parents=True, exist_ok=True)
    save_dir_vecs.mkdir(parents=True, exist_ok=True)
    
    # Initialize converter
    example_path = data_dir / f"{example_id}.npy"
    converter = MotionConverter(example_path, joints_num=22)
    
    # Process all files
    source_list = sorted(data_dir.glob("*.npy"))
    frame_num = 0
    
    for source_file in tqdm(source_list, desc="Processing motion files"):
        try:
            # Load data
            source_data = np.load(source_file)[:, :converter.joints_num]
            
            # Convert to motion vector
            data, ground_positions, positions, l_velocity = converter.convert_to_motion_vector(
                source_data, feet_threshold=0.002
            )
            
            # Recover joint positions from motion vector
            rec_ric_data = recover_from_ric(
                torch.from_numpy(data).unsqueeze(0).float(), 
                converter.joints_num
            )
            
            # Save results
            np.save(save_dir_joints / source_file.name, rec_ric_data.squeeze().numpy())
            np.save(save_dir_vecs / source_file.name, data)
            
            frame_num += data.shape[0]
            
        except Exception as e:
            print(f"Error processing {source_file.name}: {e}")
    
    print(f'Total clips: {len(source_list)}, Frames: {frame_num}, '
          f'Duration: {frame_num / 20 / 60:.2f}m')


if __name__ == "__main__":

    from smplx import SMPL
    smpl_model = SMPL(model_path='/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smpl/SMPL_NEUTRAL.pkl')
    smpl_model = smpl_model.to('cuda')  # Move model to CUDA
    default_smpl_output = smpl_model()
    target_skeleton_offsets = default_smpl_output.joints[:, :22].detach().cpu().numpy()  # (22, 3)
    np.save('./smpl_offsets.npy', target_skeleton_offsets)

    converter = MotionConverter('./smpl_offsets.npy', joints_num=22)


    source_data = np.load('../smpl_example.npy').astype(np.float32)
    global_orient_smpl = source_data[:, :3]
    body_pose_smpl = source_data[:, 3:66].reshape(len(source_data), 21, 3)
    transl_smpl = source_data[:, 72:75]
    betas_smpl = source_data[:, 75:85]
    with torch.no_grad():
        smpl_output = smpl_model(
            body_pose=torch.from_numpy(body_pose_smpl).to('cuda'),
            global_orient=torch.from_numpy(np.concatenate([global_orient_smpl[:,None,:], np.zeros((len(global_orient_smpl), 2, 3), dtype=np.float32)], axis=1)).to('cuda'),
            transl=torch.from_numpy(transl_smpl).to('cuda'),
            betas=torch.from_numpy(betas_smpl).to('cuda'),
        )
    source_data = smpl_output.joints[:, :22].detach().cpu().numpy()


    data, ground_positions, positions, l_velocity = converter.convert_to_motion_vector(
        source_data, feet_threshold=0.002
    )
    
    # Recover joint positions from motion vector
    rec_ric_data = recover_from_ric(
        torch.from_numpy(data).unsqueeze(0).float(), 
        converter.joints_num
    )



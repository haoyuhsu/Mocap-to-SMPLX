"""
Pipeline: SMPL → SMPL-X and SMPL → 263D representation for DIP-IMU Dataset

This script converts SMPL motion sequences from the DIP-IMU dataset
to SMPL-X format and/or 263-dimensional motion representation.

The DIP-IMU dataset is already in Y-up convention (same as AMASS).
Target FPS: 30 (downsampled from 60 FPS)
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from smplx import SMPL
from human_body_prior.body_model.body_model import BodyModel
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
from smplifyx.optimize import optimize_pose_smplx

# Add path to conversion utilities
sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/Mocap-to-SMPLX/HumanML3D')
from convert_from_85_to_263 import MotionConverter


class SMPLToSMPLXAnd263Pipeline:
    """Complete pipeline for SMPL → SMPL-X and/or SMPL → 263D motion representation."""
    
    def __init__(self, 
                 smpl_model=None,
                 smplx_model=None,
                 rest_pelvis_smpl=None,
                 rest_pelvis_smplx=None,
                 skeleton_path='./smpl_offsets_temp.npy',
                 device='cuda'):
        """
        Initialize the pipeline.
        
        Args:
            smpl_model: SMPL model instance (smplx)
            smplx_model: SMPL-X model instance (BodyModel)
            rest_pelvis_smpl: Rest pelvis position for SMPL
            rest_pelvis_smplx: Rest pelvis position for SMPL-X
            skeleton_path: Path to skeleton offsets for 263D conversion
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        
        # Load models
        self.smpl_model = smpl_model
        self.rest_pelvis_smpl = rest_pelvis_smpl
        
        self.smplx_model = smplx_model
        self.rest_pelvis_smplx = rest_pelvis_smplx
        
        # Initialize motion converter with SMPL skeleton
        self.motion_converter = MotionConverter(skeleton_path, joints_num=22)
    
    def smpl_to_joints(self, smpl_params):
        """
        Convert SMPL parameters to joint positions.
        
        Args:
            smpl_params: Dictionary with SMPL parameters
        
        Returns:
            joints: Joint positions (nf, 22, 3)
        """
        nf = smpl_params['transl'].shape[0]
        
        # Parse SMPL pose parameters
        global_orient = smpl_params['global_orient']  # (nf, 3)
        body_pose = smpl_params['body_pose']          # (nf, 69)
        transl = smpl_params['transl']                 # (nf, 3) pelvis location
        
        # Handle betas - might be (10,) or (nf, 10)
        if len(smpl_params['betas'].shape) == 1:
            betas = np.tile(smpl_params['betas'], (nf, 1))
        else:
            betas = smpl_params['betas']
        
        # Ensure all parameters are tensors on device
        global_orient = torch.from_numpy(global_orient).float().to(self.device)
        body_pose = torch.from_numpy(body_pose).float().to(self.device)
        transl = torch.from_numpy(transl).float().to(self.device)
        betas = torch.from_numpy(betas).float().to(self.device)
        
        # Get joint positions from SMPL
        with torch.no_grad():
            smpl_output = self.smpl_model(
                global_orient=global_orient[:, None, :],
                body_pose=body_pose.reshape(nf, 23, 3),
                transl=transl - torch.from_numpy(self.rest_pelvis_smpl).float().to(self.device),
                betas=betas
            )
        joints = smpl_output.joints[:, :22, :].detach().cpu().numpy()
        
        return joints
        
    def smpl_to_smplx(self, smpl_params):
        """
        Convert SMPL parameters to SMPL-X parameters using optimization.
        
        Args:
            smpl_params: Dictionary with keys:
                - 'pose': (nf, 72) - SMPL pose parameters
                - 'trans': (nf, 3) - global translation
                - 'betas': (10,) or (nf, 10) - shape parameters
        
        Returns:
            smplx_params: Dictionary with SMPL-X parameters
        """
        nf = smpl_params['transl'].shape[0]
        
        # Parse SMPL pose parameters
        global_orient = smpl_params['global_orient']  # (nf, 3)
        body_pose = smpl_params['body_pose']          # (nf, 69)
        transl = smpl_params['transl']                 # (nf, 3) pelvis location
        
        # Handle betas - might be (10,) or (nf, 10)
        if len(smpl_params['betas'].shape) == 1:
            betas = np.tile(smpl_params['betas'], (nf, 1))
        else:
            betas = smpl_params['betas']
        
        # Ensure all parameters are tensors on device
        global_orient = torch.from_numpy(global_orient).float().to(self.device)
        body_pose = torch.from_numpy(body_pose).float().to(self.device)
        transl = torch.from_numpy(transl).float().to(self.device)
        betas = torch.from_numpy(betas).float().to(self.device)
        
        # Get target joint positions from SMPL
        with torch.no_grad():
            smpl_output = self.smpl_model(
                global_orient=global_orient[:, None, :],
                body_pose=body_pose.reshape(nf, 23, 3),
                transl=transl - torch.from_numpy(self.rest_pelvis_smpl).float().to(self.device),
                betas=betas
            )
        target_joints = smpl_output.joints[:, :22, :].detach().cpu().numpy()

        # Add confidence (1.0 for all joints)
        joint_positions = np.concatenate([target_joints, np.ones((nf, 22, 1))], axis=-1)
        # Set NaN joints to 0 confidence
        nan_joints = np.isnan(joint_positions).any(axis=-1)
        joint_positions[nan_joints] = 0.0
        joint_positions = torch.tensor(joint_positions, dtype=torch.float32, device=self.device)
        
        # Initialize SMPL-X parameters
        params = {
            'global_orient': global_orient.cpu().numpy(),
            'transl': target_joints[:, 0, :],
            'body_pose': np.zeros((nf, 63), dtype=np.float32),  # SMPL-X has 21 body joints (63 params)
            'betas': np.zeros((nf, 10), dtype=np.float32)
        }

        for key in params.keys():
            params[key] = torch.tensor(params[key], dtype=torch.float32, device=self.device)
            params[key].requires_grad = True
        
        # Optimize root translation and orientation
        params = optimize_pose_smplx(params, self.smplx_model, joint_positions, OPT_RT=True)
        
        # Optimize body poses
        params = optimize_pose_smplx(params, self.smplx_model, joint_positions, 
                                     OPT_RT=True, OPT_POSE=True)
        
        # Convert to numpy
        smplx_params = {k: v.detach().cpu().numpy() for k, v in params.items()}
        smplx_params['transl'] += self.rest_pelvis_smplx  # add back to get pelvis location
        
        return smplx_params
    
    def joints_to_263(self, joints):
        """
        Convert joint positions to 263D motion representation.
        
        Args:
            joints: Joint positions (nf, 22, 3)
        
        Returns:
            motion_263: 263D motion representation (nf, 263)
        """
        data, ground_positions, positions, l_velocity = self.motion_converter.convert_to_motion_vector(
            joints, feet_threshold=0.002
        )
        return data
    
    def convert(self, smpl_params):
        """
        Full pipeline: SMPL → 263D representation (and optionally SMPL-X).
        
        Args:
            smpl_params: Dictionary with SMPL parameters
        
        Returns:
            motion_263: 263D motion representation
            smplx_params: SMPL-X parameters
        """
        print("Converting SMPL to 263D representation...")
        joints = self.smpl_to_joints(smpl_params)
        motion_263 = self.joints_to_263(joints)
    
        print("Converting SMPL to SMPL-X...")
        smplx_params = self.smpl_to_smplx(smpl_params)
        
        return motion_263, smplx_params


def load_dip_imu_data(pkl_file_path, target_fps=30):
    """
    Load DIP-IMU data from a single .pkl file (following process_dipimu logic).
    
    Args:
        pkl_file_path: Path to a single DIP-IMU .pkl file
        target_fps: Target FPS for downsampling (default: 30)
    
    Returns:
        Dictionary containing:
            - 'global_orient': (nf, 3) - global orientation (axis-angle)
            - 'body_pose': (nf, 69) - body pose (axis-angle)
            - 'transl': (nf, 3) - global translation (zeros for DIP)
            - 'betas': (10,) - shape parameters (default ones)
            - 'imu_acc': (nf, 6, 3) - IMU accelerations
            - 'imu_ori': (nf, 6, 3, 3) - IMU orientations
        Or None if file cannot be processed
    """
    imu_mask = [7, 8, 9, 10, 0, 2]  # left wrist, right wrist, left thigh, right thigh, head, pelvis
    
    # Enable downsampling from 60 FPS to target_fps
    step = max(1, round(60 / target_fps))

    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
    ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
    pose = torch.from_numpy(data['gt']).float()  # (nf, 72) axis-angle
    
    # Fill nan with nearest neighbors
    for _ in range(4):
        acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
        ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
        acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
        ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])
    
    # Downsample
    acc = acc[6:-6:step].contiguous()
    ori = ori[6:-6:step].contiguous()
    pose = pose[6:-6:step].contiguous()
    
    shape = torch.ones((10))  # Default shape for DIP
    tran = torch.zeros(pose.shape[0], 3)  # DIP-IMU does not contain translations
    
    # Check for NaN values
    if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
        # Split pose into global_orient and body_pose for consistency with pipeline
        global_orient = pose[:, :3].numpy()    # (nf, 3)
        body_pose = pose[:, 3:72].numpy()      # (nf, 69)
        
        return {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'transl': tran.numpy(),
            'betas': shape.numpy(),
            'imu_acc': acc.numpy(),
            'imu_ori': ori.numpy()
        }
    else:
        print(f"  -> Skipped due to NaN values")
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert SMPL parameters to SMPL-X and 263D motion representation.")
    parser.add_argument('--data_dir', type=str, default='/home/haoyuyh3/Documents/maxhsu/imu-humans/data/DIP_IMU', 
                       help='Root directory of DIP-IMU dataset')
    parser.add_argument('--out_dir', type=str, default='./output_dip_smplx_263',
                       help='Output directory for converted results')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get SMPL and SMPL-X models and attributes
    smplx_model_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz'
    smpl_model_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smpl/SMPL_NEUTRAL.pkl'
    
    # Load SMPL model
    print("Loading SMPL model...")
    smpl_model = SMPL(model_path=smpl_model_path).to(device)
    smpl_model.eval()
    for p in smpl_model.parameters():
        p.requires_grad = False
    default_smpl_output = smpl_model()
    rest_pelvis_smpl = default_smpl_output.joints[0, 0].detach().cpu().numpy()
    
    # Load SMPL-X model
    print("Loading SMPL-X model...")
    smplx_model = BodyModel(
        bm_fname=smplx_model_path,
        num_betas=10,
        model_type='smplx'
    ).to(device)
    default_smplx_output = smplx_model()
    rest_pelvis_smplx = default_smplx_output.Jtr[0, 0].detach().cpu().numpy()
    
    # Create skeleton reference for 263D conversion
    skeleton_path = './smpl_offsets_temp.npy'
    if not os.path.exists(skeleton_path):
        target_skeleton_offsets = default_smpl_output.joints[:, :22].detach().cpu().numpy()
        np.save(skeleton_path, target_skeleton_offsets)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = SMPLToSMPLXAnd263Pipeline(
        smpl_model=smpl_model,
        smplx_model=smplx_model,
        rest_pelvis_smpl=rest_pelvis_smpl,
        rest_pelvis_smplx=rest_pelvis_smplx,
        skeleton_path=skeleton_path,
        device=device
    )
    
    print(f"Loading DIP-IMU dataset from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    
    # Get all subject directories
    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('s_')])
    
    total_processed = 0
    total_skipped = 0
    
    # Process each subject
    for subject_dir in subject_dirs:
        print(f"\nProcessing subject: {subject_dir.name}")
        
        # Create output directory with same structure
        out_subject_dir = Path(args.out_dir) / subject_dir.name
        os.makedirs(out_subject_dir, exist_ok=True)
        
        # Get all pkl files for this subject
        pkl_files = sorted(list(subject_dir.glob('*.pkl')))
        
        for pkl_file in tqdm(pkl_files, desc=f"Converting {subject_dir.name}"):
            
            # Load and preprocess data
            seq_data = load_dip_imu_data(pkl_file, target_fps=30)
            
            if seq_data is None:
                total_skipped += 1
                continue
            
            smpl_params = {
                'global_orient': seq_data['global_orient'],
                'body_pose': seq_data['body_pose'],
                'transl': seq_data['transl'],
                'betas': seq_data['betas']
            }
            
            # Convert to 263D (and SMPL-X)
            motion_263, smplx_params = pipeline.convert(smpl_params)
            
            # Save results with same filename
            output_file = out_subject_dir / pkl_file.name
            result_dict = {
                'motion_263': motion_263.astype(np.float32),
                'smplx_params': smplx_params,
                'orig_smpl_params': smpl_params,
                'imu_acc': seq_data['imu_acc'].astype(np.float32),
                'imu_ori': seq_data['imu_ori'].astype(np.float32)
            }
            
            with open(output_file, 'wb') as f:
                pickle.dump(result_dict, f)
            
            total_processed += 1
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Processed: {total_processed} sequences")
    print(f"Skipped: {total_skipped} sequences")
    print(f"Results saved to: {args.out_dir}")
    print("="*50)
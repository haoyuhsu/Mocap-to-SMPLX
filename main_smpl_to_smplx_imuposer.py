"""
Pipeline: SMPL → SMPL-X and SMPL → 263D representation

This script converts SMPL motion sequences from the imuposer dataset
to SMPL-X format and/or 263-dimensional motion representation.

The dataset uses Z-up convention, which is converted to Y-up convention.
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


def load_smpl_imu_from_pkl(file_path):
    """
    Load SMPL parameters and IMU data from .pkl file (imuposer dataset format).
    Converts from Z-up to Y-up coordinate system.
    
    Expected format from dataset:
    - 'pose': (nf, 72) - SMPL pose parameters (Z-up)
    - 'trans': (nf, 3) - global translation (Z-up)
    - 'betas': (300,) - shape parameters
    - 'imu': (nf, 60) - IMU data: acc (15) + ori (45) (Z-up) for 5 devices
        - 'devices': 0: left wrist, 1: right wrist, 2: left front pocket, 3: right front pocket, 4: head
    
    Returns:
        data_dict: Dictionary with converted SMPL parameters and IMU data (Y-up)
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to numpy if needed
    pose_zup = data['pose'].cpu().numpy() if torch.is_tensor(data['pose']) else data['pose']
    trans_zup = data['trans'].cpu().numpy() if torch.is_tensor(data['trans']) else data['trans']
    betas = data['betas'].cpu().numpy() if torch.is_tensor(data['betas']) else data['betas']
    imu_data = data['imu'].cpu().numpy() if torch.is_tensor(data['imu']) else data['imu']
    
    global_orient_zup = pose_zup[:, :3]
    body_pose = pose_zup[:, 3:72]

    ##### Convert from Z-up to Y-up #####
    # Rotation matrix from Z-up to Y-up
    # R_zup_to_yup = np.array([
    #     [1,  0,  0],
    #     [0,  0,  1],
    #     [0, -1,  0]
    # ], dtype=np.float32)
    R_zup_to_yup = np.array([    # adapted from MobilePoser process.py
        [-1, 0, 0],
        [0, 0, 1], 
        [0, 1, 0.]
    ], dtype=np.float32)

    transl_yup = (R_zup_to_yup @ (trans_zup + rest_pelvis_smpl).T).T   # get pelvis location
    global_orient_yup = R.from_matrix(
        np.einsum('ij,njk->nik', R_zup_to_yup, R.from_rotvec(global_orient_zup).as_matrix())
    ).as_rotvec()

    nf = imu_data.shape[0]
    imu_yup = imu_data.copy()

    # Convert accelerations (first 15 values: 5 IMUs * 3)
    acc = imu_data[:, :15].reshape(nf, 5, 3)
    acc_yup = np.einsum('ij,nkj->nki', R_zup_to_yup, acc)
    imu_yup[:, :15] = acc_yup.reshape(nf, 15)

    # Convert orientations (last 45 values: 5 IMUs * 9)
    ori = imu_data[:, 15:].reshape(nf, 5, 3, 3)
    ori_yup = np.einsum('ij,nkjl->nkil', R_zup_to_yup, ori)
    imu_yup[:, 15:] = ori_yup.reshape(nf, 45)
    
    return {
        'global_orient': global_orient_yup,
        'transl': transl_yup,  # now 'transl' is pelvis location
        'body_pose': body_pose,
        'betas': betas[:10] if betas.shape[0] >= 10 else betas,  # use first 10 betas
        'imu': imu_yup
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert SMPL parameters to SMPL-X and 263D motion representation.")
    parser.add_argument('--data_dir', type=str, default='./imuposer_dataset', 
                       help='Root directory of imuposer dataset')
    parser.add_argument('--out_dir', type=str, default='./output_smplx_263',
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
    
    print("Loading IMUPoser dataset...")
    data_dir = Path(args.data_dir)

    # Process all participants (P1-P10)
    participant_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('P')])
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process each participant
    for participant_dir in participant_dirs:
            
        print(f"\nProcessing participant: {participant_dir.name}")
        
        out_participant_dir = Path(args.out_dir) / participant_dir.name
        os.makedirs(out_participant_dir, exist_ok=True)
        
        pkl_files = sorted(list(participant_dir.glob('*.pkl')))
        
        for pkl_file in tqdm(pkl_files, desc=f"Converting {participant_dir.name}"):

            # Load sequence data and convert to Y-up
            seq_data = load_smpl_imu_from_pkl(pkl_file)

            smpl_params = {
                'global_orient': seq_data['global_orient'],
                'body_pose': seq_data['body_pose'],
                'transl': seq_data['transl'],  # pelvis location
                'betas': seq_data['betas'],
            }
            
            # Convert to 263D (and optionally SMPL-X)
            motion_263, smplx_params = pipeline.convert(smpl_params)
            
            # Save results
            output_file = out_participant_dir / pkl_file.name
            result_dict = {
                'motion_263': motion_263.astype(np.float32),
                'smplx_params': smplx_params,
                'orig_smpl_params': smpl_params,
                'imu_data': seq_data['imu'].astype(np.float32)
            }
            
            with open(output_file, 'wb') as f:
                pickle.dump(result_dict, f)
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Results saved to: {args.out_dir}")
    print("="*50)
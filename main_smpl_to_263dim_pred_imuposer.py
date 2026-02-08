"""
Convert IMUPoser predictions to 263-dimensional motion representation.

This script loads predictions from IMUPoser (in SMPL format) and converts them
to 263D representation for motion metric evaluation.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from smplx import SMPL
import argparse
import os
import sys

# Add path to conversion utilities
sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/Mocap-to-SMPLX/HumanML3D')
from convert_from_85_to_263 import MotionConverter


class SMPLTo263Pipeline:
    """Pipeline for SMPL → 263D motion representation."""
    
    def __init__(self, 
                 smpl_model=None,
                 rest_pelvis_smpl=None,
                 skeleton_path='./smpl_offsets_temp.npy',
                 device='cuda'):
        """
        Initialize the pipeline.
        
        Args:
            smpl_model: SMPL model instance (smplx)
            rest_pelvis_smpl: Rest pelvis position for SMPL
            skeleton_path: Path to skeleton offsets for 263D conversion
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        
        # Load models
        self.smpl_model = smpl_model
        self.rest_pelvis_smpl = rest_pelvis_smpl
        
        # Initialize motion converter with SMPL skeleton
        self.motion_converter = MotionConverter(skeleton_path, joints_num=22)
    
    def smpl_to_joints(self, smpl_params):
        """
        Convert SMPL parameters to joint positions.
        
        Args:
            smpl_params: Dictionary with SMPL parameters
                - 'global_orient': (nf, 3) - global orientation (axis-angle)
                - 'body_pose': (nf, 23, 3) or (nf, 69) - body pose (axis-angle)
                - 'transl': (nf, 3) - global translation
                - 'betas': (nf, 10) or (10,) - shape parameters
        
        Returns:
            joints: Joint positions (nf, 22, 3)
        """
        global_orient = smpl_params['orient']  # (nf, 3)
        body_pose = smpl_params['pose']        # (nf, 23, 3)
        transl = smpl_params['transl']         # (nf, 3)
        betas = smpl_params['betas']           # (nf, 10) or (10,)
        
        nf = global_orient.shape[0]
        
        # Handle betas - might be (10,) or (nf, 10)
        if len(betas.shape) == 1:
            betas = np.tile(betas, (nf, 1))
        
        # Ensure all parameters are tensors on device
        global_orient = torch.from_numpy(global_orient).float().to(self.device)
        body_pose = torch.from_numpy(body_pose).float().to(self.device)
        transl = torch.from_numpy(transl).float().to(self.device)
        betas = torch.from_numpy(betas).float().to(self.device)
        
        # Reshape body_pose if needed: (nf, 23, 3) or (nf, 69)
        if len(body_pose.shape) == 2:
            # Already (nf, 69) or needs reshape
            if body_pose.shape[1] == 69:
                body_pose = body_pose.reshape(nf, 23, 3)
            elif body_pose.shape[1] == 23 * 3:
                body_pose = body_pose.reshape(nf, 23, 3)
        
        # Get joint positions from SMPL
        with torch.no_grad():
            smpl_output = self.smpl_model(
                global_orient=global_orient[:, None, :],
                body_pose=body_pose,
                transl=transl - torch.from_numpy(self.rest_pelvis_smpl).float().to(self.device),
                betas=betas
            )
        joints = smpl_output.joints[:, :22, :].detach().cpu().numpy()
        
        return joints
    
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
        Full pipeline: SMPL → 263D representation.
        
        Args:
            smpl_params: Dictionary with SMPL parameters
        
        Returns:
            motion_263: 263D motion representation
        """
        joints = self.smpl_to_joints(smpl_params)
        motion_263 = self.joints_to_263(joints)
        
        return motion_263


def load_imuposer_predictions(pred_dir):
    """
    Load IMUPoser predictions from directory.
    
    Args:
        pred_dir: Path to directory containing prediction pickle files
    
    Returns:
        List of dictionaries containing predictions and ground truth
    """
    pred_dir = Path(pred_dir)
    pkl_files = sorted(list(pred_dir.glob('sample_*.pkl')))
    
    predictions = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        predictions.append(data)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Convert IMUPoser predictions to 263D motion representation.")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing IMUPoser predictions (e.g., ../../predictions/imuposer/humanml_global)')
    parser.add_argument('--out_dir', type=str, default='./output_imuposer_263',
                       help='Output directory for converted results')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load SMPL model
    smpl_model_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smpl/SMPL_NEUTRAL.pkl'
    
    print("Loading SMPL model...")
    smpl_model = SMPL(model_path=smpl_model_path).to(device)
    smpl_model.eval()
    for p in smpl_model.parameters():
        p.requires_grad = False
    default_smpl_output = smpl_model()
    rest_pelvis_smpl = default_smpl_output.joints[0, 0].detach().cpu().numpy()
    
    # Create skeleton reference for 263D conversion
    skeleton_path = './smpl_offsets_temp.npy'
    if not os.path.exists(skeleton_path):
        target_skeleton_offsets = default_smpl_output.joints[:, :22].detach().cpu().numpy()
        np.save(skeleton_path, target_skeleton_offsets)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = SMPLTo263Pipeline(
        smpl_model=smpl_model,
        rest_pelvis_smpl=rest_pelvis_smpl,
        skeleton_path=skeleton_path,
        device=device
    )
    
    print(f"Loading predictions from {args.data_dir}...")
    predictions = load_imuposer_predictions(args.data_dir)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Converting {len(predictions)} sequences to 263D representation...")
    
    for idx, pred_data in enumerate(tqdm(predictions, desc="Converting")):
        pred_smpl = pred_data['pred']
        
        # Convert predictions to 263D
        pred_motion_263 = pipeline.convert(pred_smpl)
        
        # Save results
        output_file = out_dir / f"sample_{idx:05d}.npy"        
        with open(output_file, 'wb') as f:
            np.save(f, pred_motion_263)
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Processed: {len(predictions)} sequences")
    print(f"Results saved to: {out_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
"""
Pipeline: SMPL-X → SMPL → 263D representation
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from smplx import SMPL, SMPLX
from human_body_prior.body_model.body_model import BodyModel
from smplifyx.optimize import optimize_pose_smpl
import argparse
import os
import pickle

# Import from convert_from_85_to_263.py
import sys
sys.path.append('/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/HumanML3D')
from convert_from_85_to_263 import MotionConverter


class SMPLXTo263Pipeline:
    """Complete pipeline for SMPL-X to 263D motion representation."""
    
    def __init__(self, 
                 smplx_model_path='/projects/benk/hhsu2/imu-humans/body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz',
                 smpl_model_path='/projects/benk/hhsu2/imu-humans/body_models/human_model_files/smpl/SMPL_NEUTRAL.pkl',
                 device='cuda'):
        """
        Initialize the pipeline.
        
        Args:
            smplx_model_path: Path to SMPL-X model
            smpl_model_path: Path to SMPL model
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        
        # Load SMPL-X model
        self.smplx_model = BodyModel(
            bm_fname=smplx_model_path,
            num_betas=10,
            model_type='smplx'
        ).to(device)
        self.smplx_model.eval()
        for p in self.smplx_model.parameters():
            p.requires_grad = False
        
        # Get SMPL-X rest pelvis
        default_smplx_output = self.smplx_model()
        self.rest_pelvis_smplx = default_smplx_output.Jtr[0, 0]
        
        # Load SMPL model
        self.smpl_model = SMPL(model_path=smpl_model_path).to(device)
        
        # Get SMPL rest pelvis
        default_smpl_output = self.smpl_model()
        self.rest_pelvis_smpl = default_smpl_output.joints[0, 0].detach().cpu().numpy()
        
        # Initialize motion converter with SMPL skeleton
        target_skeleton_offsets = default_smpl_output.joints[:, :22].detach().cpu().numpy()
        skeleton_path = './smpl_offsets_temp.npy'
        np.save(skeleton_path, target_skeleton_offsets)
        self.motion_converter = MotionConverter(skeleton_path, joints_num=22)
        
    def smplx_to_smpl(self, smplx_params):
        """
        Convert SMPL-X parameters to SMPL parameters using optimization.
        
        Args:
            smplx_params: Dictionary with keys:
                - 'global_orient': (nf, 3)
                - 'body_pose': (nf, 63) or (nf, 21, 3)
                - 'transl': (nf, 3)
                - 'betas': (nf, 10)
        
        Returns:
            smpl_params: Dictionary with SMPL parameters in same format
        """
        nf = smplx_params['global_orient'].shape[0]
        nj = 22
        
        # Ensure all parameters are tensors on device
        for key in smplx_params.keys():
            if not isinstance(smplx_params[key], torch.Tensor):
                smplx_params[key] = torch.from_numpy(smplx_params[key]).float().to(self.device)
            else:
                smplx_params[key] = smplx_params[key].to(self.device)
        
        # Reshape body_pose if needed
        if len(smplx_params['body_pose'].shape) == 3:
            body_pose = smplx_params['body_pose'].reshape(nf, -1)
        else:
            body_pose = smplx_params['body_pose']
        
        # Get target joint positions from SMPL-X
        dummy_betas = torch.zeros((nf, 10), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            smplx_output = self.smplx_model(
                pose_body=body_pose,
                root_orient=smplx_params['global_orient'],
                trans=smplx_params['transl'] - self.rest_pelvis_smplx,
                betas=dummy_betas
            )
        target_joints = smplx_output.Jtr[:, :22, :].detach().cpu().numpy()
        
        # Add confidence (1.0 for all joints)
        joint_positions = np.concatenate([target_joints, np.ones((nf, nj, 1))], axis=-1)
        # Set NaN joints to 0 confidence
        nan_joints = np.isnan(joint_positions).any(axis=-1)
        joint_positions[nan_joints] = 0.0
        joint_positions = torch.tensor(joint_positions, dtype=torch.float32, device=self.device)
        
        # Initialize SMPL parameters
        params = {
            'global_orient': smplx_params['global_orient'].cpu().numpy(),
            'transl': target_joints[:, 0, :],
            'body_pose': np.zeros((nf, 23, 3)),
            'betas': np.zeros((nf, 10)),
        }
        for key in params.keys():
            params[key] = torch.tensor(params[key], dtype=torch.float32, device=self.device)
        
        # Optimize root translation and orientation
        params = optimize_pose_smpl(params, self.smpl_model, joint_positions, OPT_RT=True)
        
        # Optimize body poses
        params = optimize_pose_smpl(params, self.smpl_model, joint_positions, 
                                    OPT_RT=True, OPT_POSE=True)
        
        # Convert to numpy and adjust translation
        smpl_params = {k: v.cpu().numpy() for k, v in params.items()}
        smpl_params['transl'] = smpl_params['transl'] + self.rest_pelvis_smpl
        
        return smpl_params
    
    def smpl_to_joints(self, smpl_params):
        """
        Convert SMPL parameters to joint positions.
        
        Args:
            smpl_params: Dictionary with SMPL parameters
        
        Returns:
            joints: Joint positions (nf, 22, 3)
        """
        nf = smpl_params['global_orient'].shape[0]
        
        with torch.no_grad():
            smpl_output = self.smpl_model(
                body_pose=torch.from_numpy(smpl_params['body_pose']).to(self.device),
                global_orient=torch.from_numpy(smpl_params['global_orient'][:, None, :]).to(self.device),
                transl=torch.from_numpy(smpl_params['transl'] - self.rest_pelvis_smpl).to(self.device),
                betas=torch.from_numpy(smpl_params['betas']).to(self.device),
            )
        joints = smpl_output.joints[:, :22].detach().cpu().numpy()
        
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
    
    def convert(self, smplx_params):
        """
        Full pipeline: SMPL-X → SMPL → 263D representation.
        
        Args:
            smplx_params: Dictionary with SMPL-X parameters
        
        Returns:
            motion_263: 263D motion representation
            smpl_params: Intermediate SMPL parameters
        """
        print("Converting SMPL-X to SMPL...")
        smpl_params = self.smplx_to_smpl(smplx_params)
        
        print("Converting SMPL to joint positions...")
        joints = self.smpl_to_joints(smpl_params)
        
        print("Converting joints to 263D representation...")
        motion_263 = self.joints_to_263(joints)
        
        return motion_263, smpl_params


def load_smplx_from_npy(file_path):
    """
    Load SMPL-X parameters from .npy file.
    
    Expected format: (nf, 85)
    - [:, :3]: global_orient
    - [:, 3:66]: body_pose (21 joints × 3)
    - [:, 72:75]: transl
    - [:, 75:85]: betas
    """
    data = np.load(file_path).astype(np.float32)
    
    smplx_params = {
        'global_orient': data[:, :3],
        'body_pose': data[:, 3:66],
        'transl': data[:, 72:75],
        'betas': data[:, 75:85],
    }
    
    return smplx_params


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert SMPL-X parameters to 263D motion representation.")
    parser.add_argument('--root_dir', type=str, help='Root directory for input and output data.')
    parser.add_argument('--out_dir_263dim', type=str, help='Output directory for 263D results.')
    parser.add_argument('--out_dir_smpl', type=str, help='Output directory for SMPL results.')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing sequences.')
    parser.add_argument('--end_idx', type=int, default=-1, help='End index for processing sequences.')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads.')
    args = parser.parse_args()


    # Hard fixed
    args.root_dir = '/work/hdd/bczy/tcheng1/exp_test_lingo_6imu_2000frame/viz_test_generate_number_merged'
    args.out_dir_263dim = '/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_263dim'
    args.out_dir_smpl = '/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_smpl'


    os.makedirs(args.out_dir_263dim, exist_ok=True)
    os.makedirs(args.out_dir_smpl, exist_ok=True)

    print("Root dir:", args.root_dir)
    print("Output dir 263D:", args.out_dir_263dim)
    print("Output dir SMPL:", args.out_dir_smpl)

    seq_names = [f for f in os.listdir(args.root_dir) if f.endswith('.npy')]

    # id_xxxxx_step_0.npy
    seq_names = sorted(seq_names, key=lambda x: int(x.split('_')[1]))

    if args.end_idx < 0:
        args.end_idx = len(seq_names)
    seq_names = seq_names[args.start_idx:args.end_idx]


    print(f'Start index: {args.start_idx}, End index: {args.end_idx}, Total sequences to process: {len(seq_names)}')


    def process_sequence(seq_name, args):
        """Process a single sequence file."""
        input_file = os.path.join(args.root_dir, seq_name)
        output_file_263dim = os.path.join(args.out_dir_263dim, seq_name)
        output_file_smpl = os.path.join(args.out_dir_smpl, seq_name[:-4] + '.pkl')

        if os.path.exists(output_file_263dim):
            return f"Output for {seq_name} already exists. Skipping..."

        try:
            # Create pipeline instance per thread to avoid GPU conflicts
            pipeline = SMPLXTo263Pipeline()
            
            # Load and process data
            data = np.load(input_file, allow_pickle=True).item()
            pred_data = data['pred']
            smplx_params = {
                'global_orient': pred_data['orient'],
                'body_pose': pred_data['pose'],   # (N, 63)
                'transl': pred_data['transl'],
                'betas': np.zeros((pred_data['orient'].shape[0], 10)),
            }
            
            # Convert to 263D representation
            motion_263, smpl_params = pipeline.convert(smplx_params)
            
            # Save 263-dim results
            np.save(output_file_263dim, motion_263)
        
            # Save SMPL parameters (global_orient (N, 3), body_pose (N, 23, 3), transl (N, 3), betas (N, 10))
            with open(output_file_smpl, 'wb') as f:
                pickle.dump(smpl_params, f)

            return f"Successfully processed {seq_name}"

            
        except Exception as e:
            return f"Error processing {seq_name}: {str(e)}"
        
    
    # Use ThreadPoolExecutor for I/O bound tasks with GPU operations
    # Use ProcessPoolExecutor if you want true parallelism (but need to handle GPU carefully)
    import concurrent.futures
    import threading
    from functools import partial
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_seq = {executor.submit(process_sequence, seq_name, args): seq_name 
                        for seq_name in seq_names}
        
        # Process completed tasks with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_seq), 
                            total=len(seq_names), desc="Processing sequences"):
            seq_name = future_to_seq[future]
            try:
                result = future.result()
                if "Error" in result or "Skipping" in result:
                    print(result)
            except Exception as exc:
                print(f'{seq_name} generated an exception: {exc}')


    
    # Save SMPL parameters in 85-dim format (Note that 263-dim will shorter than original dim by 1)
    # nf = motion_263.shape[0]
    # smpl_85 = np.zeros((nf, 85))
    # smpl_85[:, :3] = smpl_params['global_orient']
    # smpl_85[:, 3:66] = smpl_params['body_pose'][:, 1:22, :].reshape(nf, 63)
    # smpl_85[:, 72:75] = smpl_params['transl']
    # smpl_85[:, 75:85] = smpl_params['betas']
    # np.save(output_dir / 'smpl_85.npy', smpl_85)
    
    # print(f"Results saved to {output_dir}")
    # print(f"Motion 263 shape: {motion_263.shape}")
    # print(f"SMPL 85 shape: {smpl_85.shape}")
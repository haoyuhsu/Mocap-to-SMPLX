"""
Pipeline: SMPL-X → SMPL → 263D representation
Modified to process ParaHome and Humoto datasets
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
import json
import concurrent.futures
import threading
from functools import partial

# Import from convert_from_85_to_263.py
import sys
sys.path.append('/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/HumanML3D')
from convert_from_85_to_263 import MotionConverter


class SMPLXTo263Pipeline:
    """Complete pipeline for SMPL-X to 263D motion representation."""
    
    def __init__(self, 
                 smplx_model=None,
                 smpl_model=None,
                 rest_pelvis_smplx=None,
                 rest_pelvis_smpl=None,
                 skeleton_path='./smpl_offsets_temp.npy',
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
        self.smplx_model = smplx_model
        self.rest_pelvis_smplx = rest_pelvis_smplx
        
        # Load SMPL model
        self.smpl_model = smpl_model
        self.rest_pelvis_smpl = rest_pelvis_smpl
        
        # Initialize motion converter with SMPL skeleton
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
        smpl_params = self.smplx_to_smpl(smplx_params)
        joints = self.smpl_to_joints(smpl_params)
        motion_263 = self.joints_to_263(joints)
        
        return motion_263, smpl_params


def process_humoto_sequence(
    seq_name_tuple,
    args,
    pipeline,
):
    """Process a single Humoto sequence file."""
    sample_name, split = seq_name_tuple
    sample_idx = str(sample_name).zfill(7)
    
    input_file = os.path.join(args.root_dir, 'all', f'{sample_idx}.pkl')
    output_file = os.path.join(
        args.motion_output_dir_train if split == 'train' else 
        args.motion_output_dir_val if split == 'val' else 
        args.motion_output_dir_test,
        f'{sample_idx}.pkl'
    )

    if os.path.exists(output_file):
        return f"Output for {sample_idx} already exists. Skipping..."

    try:
        # Load Humoto data
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract motion_smpl: (T, 85) for Humoto
        motion_data_smpl85 = data['motion_smpl']  # (T, 85) in SMPL-X format
        texts = data['text']  # list of texts
        
        # Build SMPL-X params (treating SMPL as SMPL-X for conversion)
        smplx_params = {
            'global_orient': motion_data_smpl85[:, 0:3].copy(),
            'body_pose': motion_data_smpl85[:, 3:66].copy(),   # (N, 63)
            'transl': motion_data_smpl85[:, 72:75].copy(),
            'betas': np.zeros((motion_data_smpl85.shape[0], 10)),
        }
        
        # Convert to 263D representation
        motion_263, smpl_params = pipeline.convert(smplx_params)
        
        # Save results
        with open(output_file, 'wb') as f:
            pickle.dump({
                'motion_263': motion_263.astype(np.float32),
                'smpl_params': smpl_params,
                'motion_data_smpl85': motion_data_smpl85.astype(np.float32),
                'texts': texts
            }, f)

        return f"Successfully processed {sample_idx}"
        
    except Exception as e:
        return f"Error processing {sample_idx}: {str(e)}"


def process_parahome_sequence(
    seq_name_tuple,
    args,
    pipeline,
):
    """Process a single ParaHome sequence file."""
    sample_name, split = seq_name_tuple
    sample_idx = sample_name.split('.')[0]
    
    # imu_traj_path = os.path.join(args.root_dir, 'imu_traj', f'{sample_idx}.npy')
    motion_smpl_path = os.path.join(args.root_dir, 'motions_smpl85', f'{sample_idx}.npy')
    text_path = os.path.join(args.root_dir, 'text_annotations', f'{sample_idx}.json')
    output_file = os.path.join(
        args.motion_output_dir_train if split == 'train' else 
        args.motion_output_dir_val if split == 'val' else 
        args.motion_output_dir_test,
        f'{sample_idx}.pkl'
    )

    if os.path.exists(output_file):
        return f"Output for {sample_idx} already exists. Skipping..."

    try:
        # Load ParaHome data
        # imu_traj = np.load(imu_traj_path, allow_pickle=True)  # (T, 6, 6)
        motion_data_smpl85 = np.load(motion_smpl_path, allow_pickle=True)  # (T, 85) in SMPL-X format
        with open(text_path, 'r') as f:
            text_dict = json.load(f)  # dictionary: 'start end': 'text'
        
        # Build SMPL-X params
        smplx_params = {
            'global_orient': motion_data_smpl85[:, 0:3].copy(),
            'body_pose': motion_data_smpl85[:, 3:66].copy(),   # (N, 63)
            'transl': motion_data_smpl85[:, 72:75].copy(),
            'betas': np.zeros((motion_data_smpl85.shape[0], 10)),
        }
        
        # Convert to 263D representation
        motion_263, smpl_params = pipeline.convert(smplx_params)
        
        # Save results
        with open(output_file, 'wb') as f:
            pickle.dump({
                'motion_263': motion_263.astype(np.float32),
                'smpl_params': smpl_params,
                'motion_data_smpl85': motion_data_smpl85.astype(np.float32),
                'texts': text_dict,
                # 'imu_traj': imu_traj,
            }, f)

        return f"Successfully processed {sample_idx}"
        
    except Exception as e:
        return f"Error processing {sample_idx}: {str(e)}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert SMPL-X parameters to 263D motion representation for ParaHome and Humoto datasets.")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['humoto', 'parahome'], 
                        help='Dataset type to process (humoto or parahome)')
    parser.add_argument('--root_dir', type=str, required=True, 
                        help='Root directory of the dataset (e.g., /scratch/bfyo/tcheng1/dataset_process/humoto_data)')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for converted results')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing sequences.')
    parser.add_argument('--end_idx', type=int, default=-1, help='End index for processing sequences.')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads.')
    args = parser.parse_args()

    # Get SMPL and SMPL-X models and attributes
    smplx_model_path = '/projects/benk/hhsu2/imu-humans/body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz'
    smpl_model_path = '/projects/benk/hhsu2/imu-humans/body_models/human_model_files/smpl/SMPL_NEUTRAL.pkl'

    smplx_model = BodyModel(
        bm_fname=smplx_model_path,
        num_betas=10,
        model_type='smplx'
    ).to('cuda')
    smplx_model.eval()
    for p in smplx_model.parameters():
        p.requires_grad = False
    default_smplx_output = smplx_model()
    rest_pelvis_smplx = default_smplx_output.Jtr[0, 0]
        
    smpl_model = SMPL(model_path=smpl_model_path).to('cuda')
    default_smpl_output = smpl_model()
    rest_pelvis_smpl = default_smpl_output.joints[0, 0].detach().cpu().numpy()

    # Initialize pipeline
    skeleton_path = '/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/smpl_offsets_temp.npy'
    pipeline = SMPLXTo263Pipeline(
        smplx_model=smplx_model,
        smpl_model=smpl_model,
        rest_pelvis_smplx=rest_pelvis_smplx,
        rest_pelvis_smpl=rest_pelvis_smpl,
        skeleton_path=skeleton_path,
        device='cuda'
    )

    print(f"Dataset type: {args.dataset_type}")
    print(f"Root dir: {args.root_dir}")
    print(f"Output dir: {args.output_dir}")

    # Create output directories
    motion_output_dir = os.path.join(args.output_dir, 'motion_data')
    os.makedirs(motion_output_dir, exist_ok=True)

    motion_output_dir_train = os.path.join(motion_output_dir, 'train')
    motion_output_dir_val = os.path.join(motion_output_dir, 'val')
    motion_output_dir_test = os.path.join(motion_output_dir, 'test')
    os.makedirs(motion_output_dir_train, exist_ok=True)
    os.makedirs(motion_output_dir_val, exist_ok=True)
    os.makedirs(motion_output_dir_test, exist_ok=True)

    args.motion_output_dir_train = motion_output_dir_train
    args.motion_output_dir_val = motion_output_dir_val
    args.motion_output_dir_test = motion_output_dir_test

    # Load dataset split indices and organize by split
    all_seq_names = []
    
    if args.dataset_type == 'humoto':
        # Humoto: load train_indices.npy, val_indices.npy, test_indices.npy
        train_split_path = os.path.join(args.root_dir, 'train_indices.npy')
        test_split_path = os.path.join(args.root_dir, 'test_indices.npy')
        val_split_path = os.path.join(args.root_dir, 'val_indices.npy')
        
        train_data_list = np.load(train_split_path).tolist()
        all_seq_names.extend([(sample_name, 'train') for sample_name in train_data_list])
        
        val_data_list = np.load(val_split_path).tolist()
        all_seq_names.extend([(sample_name, 'val') for sample_name in val_data_list])
        
        test_data_list = np.load(test_split_path).tolist()
        all_seq_names.extend([(sample_name, 'test') for sample_name in test_data_list])
        
        process_fn = partial(process_humoto_sequence, args=args, pipeline=pipeline)
        
    elif args.dataset_type == 'parahome':
        # ParaHome: load train_split.npy, val_split.npy, test_split.npy
        train_split_path = os.path.join(args.root_dir, 'train_split.npy')
        test_split_path = os.path.join(args.root_dir, 'test_split.npy')
        val_split_path = os.path.join(args.root_dir, 'val_split.npy')
        
        train_data_list = np.load(train_split_path).tolist()
        all_seq_names.extend([(sample_name, 'train') for sample_name in train_data_list])
    
        val_data_list = np.load(val_split_path).tolist()
        all_seq_names.extend([(sample_name, 'val') for sample_name in val_data_list])
    
        test_data_list = np.load(test_split_path).tolist()
        all_seq_names.extend([(sample_name, 'test') for sample_name in test_data_list])
        
        process_fn = partial(process_parahome_sequence, args=args, pipeline=pipeline)

    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    # Sort by sample name
    all_seq_names = sorted(all_seq_names, key=lambda x: x[0])
    
    # Count splits
    train_count = sum(1 for _, split in all_seq_names if split == 'train')
    val_count = sum(1 for _, split in all_seq_names if split == 'val')
    test_count = sum(1 for _, split in all_seq_names if split == 'test')
    print(f"Total train/val/test sequences: {train_count}/{val_count}/{test_count}")

    # Apply start/end index filtering
    if args.end_idx < 0:
        args.end_idx = len(all_seq_names)
    seq_names = all_seq_names[args.start_idx:args.end_idx]

    print(f'Start index: {args.start_idx}, End index: {args.end_idx}, Total sequences to process: {len(seq_names)}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_seq = {executor.submit(process_fn, seq_name_tuple): seq_name_tuple
                        for seq_name_tuple in seq_names}
        
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

    print(f"\nProcessing complete! Results saved to {args.output_dir}")
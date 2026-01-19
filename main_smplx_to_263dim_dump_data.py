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
    parser.add_argument('--out_dir', type=str, help='Output directory for converted results.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Sub-dataset name to process')
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

    root_dir = args.root_dir
    out_dir = args.out_dir
    dataset_name = args.dataset_name

    print("Root dir:", root_dir)
    print("Output dir:", out_dir)
    print("Dataset name:", dataset_name)

    motion_data_dir = os.path.join(root_dir, 'motion_data')   # per-sequence
    motion_output_dir = os.path.join(out_dir, 'motion_data')  # per-sequence
    os.makedirs(motion_output_dir, exist_ok=True)

    motion_output_dir_train = os.path.join(motion_output_dir, 'train')
    motion_output_dir_val = os.path.join(motion_output_dir, 'val')
    motion_output_dir_test = os.path.join(motion_output_dir, 'test')
    os.makedirs(motion_output_dir_train, exist_ok=True)
    os.makedirs(motion_output_dir_val, exist_ok=True)
    os.makedirs(motion_output_dir_test, exist_ok=True)

    args.motion_data_dir = motion_data_dir
    args.motion_output_dir_train = motion_output_dir_train
    args.motion_output_dir_val = motion_output_dir_val
    args.motion_output_dir_test = motion_output_dir_test

    # Get all ids of each split (train/val/test) from each task (t2m/tokenizer)
    train_id_list, val_id_list, test_id_list = [], [], []
    for task in ['t2m', 'tokenizer']:
        task_train_txt = os.path.join(root_dir, 'splits', f'{task}_train.txt')
        task_train_f_list = [line.strip() for line in open(task_train_txt).readlines()]
        train_id_list.extend(task_train_f_list)
        task_val_txt = os.path.join(root_dir, 'splits', f'{task}_val.txt')
        task_val_f_list = [line.strip() for line in open(task_val_txt).readlines()]
        val_id_list.extend(task_val_f_list)
        task_test_txt = os.path.join(root_dir, 'splits', f'{task}_test.txt')
        task_test_f_list = [line.strip() for line in open(task_test_txt).readlines()]
        test_id_list.extend(task_test_f_list)
    train_id_list = sorted(set(train_id_list))
    val_id_list = sorted(set(val_id_list))
    test_id_list = sorted(set(test_id_list))
    print(f"Total train/val/test sequences: {len(train_id_list)}/{len(val_id_list)}/{len(test_id_list)}")

    all_seq_names = []
    all_seq_names.extend([(id, 'train') for id in train_id_list if id.startswith(dataset_name)])
    all_seq_names.extend([(id, 'val') for id in val_id_list if id.startswith(dataset_name)])
    all_seq_names.extend([(id, 'test') for id in test_id_list if id.startswith(dataset_name)])

    all_seq_names = sorted(all_seq_names, key=lambda x: x[0])

    if args.end_idx < 0:
        args.end_idx = len(all_seq_names)
    seq_names = all_seq_names[args.start_idx:args.end_idx]

    print(f'Start index: {args.start_idx}, End index: {args.end_idx}, Total sequences to process: {len(seq_names)}')


    def process_sequence(
        seq_name_tuple,
        args,
        pipeline,
    ):
        """Process a single sequence file."""
        seq_name, split = seq_name_tuple

        input_file = os.path.join(args.motion_data_dir, split, seq_name.replace('/', '_') + '.pkl')
        output_file = os.path.join(
            args.motion_output_dir_train if split == 'train' else 
            args.motion_output_dir_val if split == 'val' else 
            args.motion_output_dir_test,
            seq_name.replace('/', '_') + '.pkl'
        )

        if os.path.exists(output_file):
            return f"Output for {seq_name} already exists. Skipping..."

        try:            
            # Load and process data
            with open(input_file, 'rb') as f:
                data = pickle.load(f)

            motion_data_smpl85 = data['motion_data_smpl85']  # (N, 85)
            texts = data['texts']

            smplx_params = {
                'global_orient': motion_data_smpl85[:, :3],
                'body_pose': motion_data_smpl85[:, 3:66],   # (N, 63)
                'transl': motion_data_smpl85[:, 72:75],
                'betas': np.zeros((motion_data_smpl85.shape[0], 10)),
            }
            
            # Convert to 263D representation
            motion_263, smpl_params = pipeline.convert(smplx_params)
            
            # Save results
            # - 263-dim motion representation
            # - SMPL parameters (global_orient (N, 3), body_pose (N, 23, 3), transl (N, 3), betas (N, 10))
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'motion_263': motion_263.astype(np.float32),
                    'smpl_params': smpl_params,
                    'motion_data_smpl85': motion_data_smpl85.astype(np.float32),
                    'texts': texts,
                }, f)

            return f"Successfully processed {seq_name}"

            
        except Exception as e:
            return f"Error processing {seq_name}: {str(e)}"

    
    process_fn = partial(process_sequence, args=args, pipeline=pipeline)
        

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
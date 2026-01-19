"""
This script consists of two parts:
1. Load the precomputed 3D positions from .npy file
    Notes: .npy has 64 joints of Optitrack.
2. do the SMPLify(-X) optimization and get SMPL(-X) parameters
"""

import argparse
import sys
import time
import os

import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import torch
import h5py
from smplx import SMPLXLayer

from smplifyx.optimize import *
from utils.io import write_smplx
from utils.mapping import (SELECTED_JOINTS, AMASS_TO_SMPLX,
                                     JointMapper, 
                                     create_amass_joint_data)
from utils.torch_utils import *
# from utils.visualize_smplx import visualize_smplx_model


def load_joint_positions(npy):
    joint_positions=np.load(npy) # (tf,64,3)
    # ignore the numb joints
    joint_positions=joint_positions[:,SELECTED_JOINTS] # (tf,tj,3)
    
    return joint_positions

# def parse_shape_pkl(shape_pkl):
#     # load pre-computed shape parameters
#     pkl=joblib.load(shape_pkl)
#     gender=pkl['gender']
#     betas=pkl['betas']
#     return gender,betas

def smplifyx(joint_positions):
    nf=joint_positions.shape[0]
    nj=joint_positions.shape[1]

    # gender,betas=parse_shape_pkl(args.shape_pkl)
    # betas=betas.repeat(nf,axis=0)

    gender = 'neutral'
    betas = np.zeros((nf, 10))

    model_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smplx'

    # create body models
    body_models=SMPLXLayer(model_path=model_path,num_betas=10,gender=gender,
                           joint_mapper=JointMapper(AMASS_TO_SMPLX),flat_hand_mean=True).to('cuda')

    # create params to be optimized,
    params={
        'body_pose':np.zeros((nf,21,3)),
        'lhand_pose':np.zeros((nf,15,3)),
        'rhand_pose':np.zeros((nf,15,3)),
        'jaw_pose':np.zeros((nf,3)),
        'leye_pose':np.zeros((nf,3)),
        'reye_pose':np.zeros((nf,3)),
        'betas':betas,
        'expression':np.zeros((nf,10)),
        'global_orient':np.zeros((nf,3)),
        'transl':np.zeros((nf,3)),
    }
    params=init_params(params,body_models,nf)
    # add confidence 1.0
    joint_positions=np.concatenate([joint_positions,np.ones((nf,nj,1))],axis=-1)
    # however, make the confidence of NaN to 0.0
    nan_joints=np.isnan(joint_positions).any(axis=-1)
    joint_positions[nan_joints]=0.0
    # convert joint_positions to tensor
    joint_positions=torch.tensor(joint_positions,dtype=torch.float32,device='cuda')

    # SMPLify-X optimization
    start=time.time()
    params=multi_stage_optimize(params,body_models,joint_positions)
    end=time.time()

    print("------------------Fitting cost %d s!--------------------"%(end-start))

    return params


if __name__=='__main__':

    # parser=argparse.ArgumentParser()
    # parser.add_argument('--npy',type=str,default='./test_data/P2.npy',
    #                     help='.npy file path that contains 64 joints of Optitrack')
    # parser.add_argument('--shape_pkl',type=str,default='./test_data/P2.pkl',
    #                     help='Pre-computed shape parameters')
    # parser.add_argument('--save_path',type=str,default='./test_data/P2_smplx.npz',
    #                     help='save the SMPLX parameters')
    # parser.add_argument('--vis_smplx',action='store_true',
    #                     help='visualize the results')
    # parser.add_argument('--vis_kp3d',action='store_true',
    #                     help='visualize the 3D joints positions')
    
    # args=parser.parse_args()

    # print('-----------------Parsing %s!-----------------'%(args.npy))
    # load 3D joints positions
    # joint_positions=load_joint_positions(args.npy)   # (tf, tj, 3) --> in this case (824, 64, 3)

    # pd_file_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/virtualhome/virtualhome/Output/test_no_image_recording=True_skip_animation=False/0/pd_test_no_image_recording=True_skip_animation=False.txt'
    # pd_file_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/virtualhome_2025_11_08/virtualhome/Output/test_no_image_recording=True_skip_animation=False_v2/0/pd_test_no_image_recording=True_skip_animation=False_v2.txt'
    # pd_file_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/Pose2Room/datasets/virtualhome_22_classes/recording/3/0/0/Female1/script/0/pd_script.txt'

    # pd_file_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/Pose2Room/datasets/virtualhome_22_classes/recording/0/0/0/Female1/script/0/pd_script.txt'
    # with open(pd_file_path, 'r') as f:
    #     lines = f.readlines()
    # joint_names = lines[0].strip().split() # First line contains column names
    # data_lines = lines[1:]                 # Remaining lines contain data
    # data = []
    # for line in data_lines:
    #     values = line.strip().split()[1:]  # Skip the first column (frame index)
    #     data.append([float(v) for v in values])
    # data_array = np.array(data)
    # data_array = data_array.reshape(data_array.shape[0], -1, 3)  # Reshape to (tf, tj, 3)

    # valid_pose = data_array.sum(-1).sum(0) != 0
    # valid_joint_names = [name for i, name in enumerate(joint_names) if valid_pose[i]]

    from human_body_prior.body_model.body_model import BodyModel
    smplx_model = BodyModel(bm_fname='/home/haoyuyh3/Documents/maxhsu/imu-humans/related_works/motion/MotionMillion-Codes/body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz', num_betas=10, model_type='smplx')
    smplx_model.eval()
    for p in smplx_model.parameters():
        p.requires_grad = False
    smplx_model.cuda()
    default_smplx_output = smplx_model()
    rest_pelvis = default_smplx_output.Jtr[0, 0].detach().cpu().numpy()


    datasets_dir = '/home/haoyuyh3/Documents/maxhsu/imu-humans/Pose2Room/datasets/virtualhome_22_classes'
    samples_dir = os.path.join(datasets_dir, 'samples')
    output_smplx_dir = os.path.join(datasets_dir, 'samples_smpl85')
    os.makedirs(output_smplx_dir, exist_ok=True)

    valid_joint_names = ['Hips', 'LeftUpperLeg', 'RightUpperLeg', 'LeftLowerLeg', 'RightLowerLeg', 'LeftFoot', 'RightFoot', 'Spine', 'Chest', 'Neck', 'Head', 'LeftShoulder', 'RightShoulder', 'LeftUpperArm', 'RightUpperArm', 'LeftLowerArm', 'RightLowerArm', 'LeftHand', 'RightHand', 'LeftToes', 'RightToes', 'LeftThumbProximal', 'LeftThumbIntermediate', 'LeftThumbDistal', 'LeftIndexProximal', 'LeftIndexIntermediate', 'LeftIndexDistal', 'LeftMiddleProximal', 'LeftMiddleIntermediate', 'LeftMiddleDistal', 'RightThumbProximal', 'RightThumbIntermediate', 'RightThumbDistal', 'RightIndexProximal', 'RightIndexIntermediate', 'RightIndexDistal', 'RightMiddleProximal', 'RightMiddleIntermediate', 'RightMiddleDistal']
    
    for sample_file in tqdm(sorted(os.listdir(samples_dir))):
        
        if not sample_file.endswith('.hdf5'):
            continue
        
        sample_path = os.path.join(samples_dir, sample_file)

        with h5py.File(sample_path, 'r') as f:
            # print("Keys:", list(f.keys()))
            # Keys: ['object_nodes', 'room_bbox', 'skeleton_joint_votes', 'skeleton_joints']
            data_array = f['skeleton_joints'][:]  # (tf, tj, 3)

        assert data_array.shape[1] == len(valid_joint_names), "Mismatch between data array joints and valid joint names"

        data_array = create_amass_joint_data(data_array, valid_joint_names)

        smpl_params = smplifyx(data_array)

        num_frames = data_array.shape[0]
        smpl85_data = np.zeros((num_frames, 85)).astype(np.float32)
        smpl85_data[:, :3] = smpl_params['global_orient'].cpu().numpy()                        # (num_frames, 3)
        smpl85_data[:, 3:66] = smpl_params['body_pose'].cpu().numpy().reshape(num_frames, -1)  # (num_frames, 63)
        smpl85_data[:, 72:75] = smpl_params['transl'].cpu().numpy() + rest_pelvis              # (num_frames, 3)

        np.save(os.path.join(output_smplx_dir, sample_file.replace('.hdf5', '.npy')), smpl85_data)

        # print('-----------------Total %s frames!-----------------'%(data_array.shape[0]))
        # print('-----------------SMPLify-X!-----------------')
        # # do smplify
        # smpl_params = smplifyx(data_array)
        # print('-----------------Done!-----------------')


        # print('-----------------Visualization using Rerun!-----------------')

        # import sys
        # sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/imu_synthesis')
        # from viewer import Viewer
        # from visualize_imu import log_smpl_85

        # viewer = Viewer()
        # log_smpl_85(viewer, smpl85_data, seq_idx=0, label='default', color_idx=0)
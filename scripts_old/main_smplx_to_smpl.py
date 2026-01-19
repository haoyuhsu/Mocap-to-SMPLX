"""
Convert SMPLX parameters to SMPL parameters using optimization.
"""

import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from utils.torch_utils import *
from smplifyx.optimize import multi_stage_optimize, optimize_pose_smpl, require_grad

from smplx import SMPL, SMPLX


if __name__=='__main__':
    
    # Convert SMPLX to SMPL
    print('-----------------Converting SMPLX to SMPL!-----------------')

    example_file_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/ParaHome/ParaHome/motions_smpl85/s1.npy'
    motion_smpl85 = np.load(example_file_path)

    smplx_params = {
        'global_orient': torch.from_numpy(motion_smpl85[:, :3]).float().cuda(),
        'body_pose': torch.from_numpy(motion_smpl85[:, 3:66]).float().cuda(),
        'transl': torch.from_numpy(motion_smpl85[:, 72:75]).float().cuda(),
        'betas': torch.from_numpy(motion_smpl85[:, 75:85]).float().cuda(),
    }

    device = 'cuda'    
    nf = smplx_params['global_orient'].shape[0]
    nj = 22
    
    # Load SMPLX model to get target joint positions
    smplx_model = BodyModel(
        bm_fname='/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz',
        num_betas=10,
        model_type='smplx'
    ).to(device)
    smplx_model.eval()
    for p in smplx_model.parameters():
        p.requires_grad = False
    default_smplx_output = smplx_model()
    rest_pelvis = default_smplx_output.Jtr[0, 0]

    
    # Get target joint positions from SMPLX
    dummy_betas = torch.zeros((nf, 10), dtype=torch.float32).cuda()
    with torch.no_grad():
        smplx_output = smplx_model(
            pose_body=smplx_params['body_pose'],
            root_orient=smplx_params['global_orient'],
            trans=smplx_params['transl'] - rest_pelvis,
            betas=dummy_betas
        )
    target_joints = smplx_output.Jtr[:, :22, :].detach().cpu().numpy()  # (nf, 22, 3)

    # add confidence 1.0
    joint_positions=np.concatenate([target_joints,np.ones((nf,nj,1))],axis=-1)
    # however, make the confidence of NaN to 0.0
    nan_joints=np.isnan(joint_positions).any(axis=-1)
    joint_positions[nan_joints]=0.0
    # convert joint_positions to tensor
    joint_positions=torch.tensor(joint_positions,dtype=torch.float32,device='cuda')


    # Load SMPL model for optimization
    smpl_model = SMPL(model_path='/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smpl/SMPL_NEUTRAL.pkl').to(device)
    
    # create params to be optimized,
    params={
        'global_orient':smplx_params['global_orient'].cpu().numpy(),
        'transl':target_joints[:, 0, :],
        'body_pose':np.zeros((nf,23,3)),
        'betas':np.zeros((nf,10)),
    }
    for key in params.keys():
        params[key]=torch.tensor(params[key],dtype=torch.float32,
                                device=torch.device('cuda'))
    
    # optimize RT
    params=optimize_pose_smpl(params,smpl_model,joint_positions,
                         OPT_RT=True)

    # optimize body poses
    params=optimize_pose_smpl(params,smpl_model,joint_positions,
                         OPT_RT=True,OPT_POSE=True)
    
    print('-----------------Done!-----------------')


    print('-----------------Visualizing the result!-----------------')
    import rerun as rr
    from tqdm import tqdm
    import trimesh
    from scipy.spatial.transform import Rotation as R
    import sys
    sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/imu_synthesis')
    from viewer import Viewer
    viewer = Viewer()
    dt = 1.0 / viewer.sample_fps

    default_smpl_output = smpl_model()
    rest_pelvis_smpl = default_smpl_output.joints[0, 0].detach().cpu().numpy()
    print('Rest pelvis for SMPL:', rest_pelvis_smpl)

    # Extract optimized parameters
    global_orient_smpl = params['global_orient'].cpu().numpy()
    body_pose_smpl = params['body_pose'].cpu().numpy()
    transl_smpl = params['transl'].cpu().numpy()
    betas_smpl = params['betas'].cpu().numpy()

    # Fix the pelvis translation
    transl_smpl = transl_smpl + rest_pelvis_smpl


    # Save
    motion_smpl85_from_smpl = np.zeros((nf, 85))
    motion_smpl85_from_smpl[:, :3] = global_orient_smpl
    motion_smpl85_from_smpl[:, 3:66] = body_pose_smpl[:, 1:22, :].reshape(nf, 63)
    motion_smpl85_from_smpl[:, 72:75] = transl_smpl
    motion_smpl85_from_smpl[:, 75:85] = betas_smpl
    np.save('./smpl_example.npy', motion_smpl85_from_smpl)


    seq_len, seq_idx, label, color_idx = nf, 0, 'smplx_to_smpl', 2

    # Log trajectories (SMPL)
    viewer.log_trajectory(transl_smpl, seq_idx, label=label, color_idx=color_idx)


    # Log pelvis orientations (SMPL)
    for frame_idx in tqdm(range(seq_len), desc="Logging orientations", dynamic_ncols=True):
        rr.set_time_sequence("frames", frame_idx)
        rr.set_time_seconds("sensor_time", frame_idx * dt)
        viewer._log_orientations(R.from_rotvec(global_orient_smpl[frame_idx]).as_matrix(), transl_smpl[frame_idx], seq_idx, label=label)


    with torch.no_grad():
        smpl_output = smpl_model(
            body_pose=torch.from_numpy(body_pose_smpl).to('cuda'),
            global_orient=torch.from_numpy(global_orient_smpl[:,None,:]).to('cuda'),
            transl=torch.from_numpy(transl_smpl - rest_pelvis_smpl).to('cuda'),
            betas=torch.from_numpy(betas_smpl).to('cuda'),
        )
        vertices = smpl_output.vertices.detach().cpu().numpy()
    faces = smpl_model.faces.astype(np.int32)
    smpl_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False) for i in range(seq_len)]

    # Log SMPL meshes
    for frame_idx in tqdm(range(seq_len), desc="Logging SMPL-X meshes", dynamic_ncols=True):
        rr.set_time_sequence("frames", frame_idx)
        rr.set_time_seconds("sensor_time", frame_idx * dt)
        viewer.log_smplx_mesh(smpl_meshes[frame_idx], seq_idx, label=label)





    # # Generate SMPL-X meshes
    # # smplx_model = BodyModel(
    # #     bm_fname='/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz',
    # #     num_betas=10,
    # #     model_type='smplx'
    # # ).to('cuda')
    # # smplx_model.eval()
    # # for p in smplx_model.parameters():
    # #     p.requires_grad = False
    # # default_smplx_output = smplx_model()
    # # rest_pelvis = default_smplx_output.Jtr[0, 0].detach().cpu().numpy()

    # smpl_model = BodyModel(
    #     bm_fname='/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/smplh/neutral/model.npz',
    #     num_betas=10,
    #     model_type='smplh'
    # ).to('cuda')
    # smpl_model.eval()
    # default_smpl_output = smpl_model()
    # rest_pelvis = default_smpl_output.Jtr[0, 0].detach().cpu().numpy()

    # traj_corrected = transl - rest_pelvis

    # with torch.no_grad():
    #     smplx_output = smpl_model(
    #         pose_body=torch.from_numpy(body_pose).cuda(), 
    #         root_orient=torch.from_numpy(global_orient).cuda(),
    #         trans=torch.from_numpy(traj_corrected).cuda(),
    #         betas=torch.zeros((seq_len, 10), dtype=torch.float32).cuda()
    #     )
    #     vertices = smplx_output.v.detach().cpu().numpy()
    # faces = smpl_model.f.detach().cpu().numpy()
    # original_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False) for i in range(seq_len)]

    # # Log SMPL-X meshes
    # for frame_idx in tqdm(range(seq_len), desc="Logging SMPL-X meshes", dynamic_ncols=True):
    #     rr.set_time_sequence("frames", frame_idx)
    #     rr.set_time_seconds("sensor_time", frame_idx * dt)
    #     viewer.log_smplx_mesh(original_meshes[frame_idx], seq_idx, label=label)

import pickle
import os
import numpy as np
from smplx import SMPL, SMPLX
import torch
from tqdm import tqdm
import trimesh
import rerun as rr
from scipy.spatial.transform import Rotation as R
from smplx import SMPL
from human_body_prior.body_model.body_model import BodyModel

import sys
sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/imu_synthesis/')
from viewer import Viewer
from get_imu_readings import simulate_imu_readings
from parametric_model import SimplifiedSMPLX

sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/MobilePoser')
from mobileposer.articulate.model import ParametricModel



##### Configuration #####

device = 'cuda'
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



##### DIP-IMU dataset #####
processed_dipimu_data_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/data/DIP_IMU_processed/s_09/03_b.pkl'
with open(processed_dipimu_data_path, 'rb') as f:
    motion_data = pickle.load(f)

imu_acc = motion_data['imu_acc']    # (N, 6, 3)
imu_ori = motion_data['imu_ori']    # (N, 6, 3, 3)

# SMPLX
smplx_params = motion_data['smplx_params']
smplx_global_orient = smplx_params['global_orient']  # (N, 3)
smplx_body_pose = smplx_params['body_pose']          # (N, 63)
smplx_betas = smplx_params['betas']                  # (N, 10)    
smplx_transl = smplx_params['transl']                # (N, 3)

# SMPL
smpl_params = motion_data['orig_smpl_params']
smpl_global_orient = smpl_params['global_orient']    # (N, 3)
smpl_body_pose = smpl_params['body_pose']            # (N, 69)
smpl_betas = smpl_params['betas']                    # (10,)
smpl_transl = smpl_params['transl']                  # (N, 3)


##### Truncate sequence for faster testing #####
max_seq_len = 100
smplx_global_orient = smplx_global_orient[:max_seq_len]
smplx_body_pose = smplx_body_pose[:max_seq_len]
smplx_betas = smplx_betas[:max_seq_len]
smplx_transl = smplx_transl[:max_seq_len]
smpl_global_orient = smpl_global_orient[:max_seq_len]
smpl_body_pose = smpl_body_pose[:max_seq_len]
smpl_transl = smpl_transl[:max_seq_len]


##### Synthesized IMU trajectory of SMPL #####
# left wrist, right wrist, left thigh, right thigh, head, pelvis
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
body_model = ParametricModel('/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smpl/SMPL_MALE.pkl', device=device)

smpl_all_pose = np.concatenate([smpl_global_orient, smpl_body_pose], axis=1)  # (N, 72)
p = torch.from_numpy(R.from_rotvec(smpl_all_pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)).float().to(device)
shape = None   # consider apply shape
tran = torch.from_numpy(smpl_transl).float().to(device)   # tran is root position!

with torch.no_grad():
    grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)   # tran is root position!

imu_vertices = vert[:, vi_mask].detach().cpu().numpy()
# imu_joints_rot = grot[:, ji_mask].detach().cpu().numpy()

imu_joints_rot = imu_ori    # use real-world sensor orientation

##### Get SMPL meshes #####
seq_len = smpl_global_orient.shape[0]

with torch.no_grad():
    smpl_output = smpl_model(
        betas = torch.zeros((smpl_global_orient.shape[0], 10), dtype=torch.float32).to(device),
        global_orient = torch.tensor(smpl_global_orient[:, None, :], dtype=torch.float32).to(device),
        body_pose = torch.tensor(smpl_body_pose.reshape(seq_len, 23, 3), dtype=torch.float32).to(device),
        transl = torch.tensor(smpl_transl - rest_pelvis_smpl, dtype=torch.float32).to(device)
    )
vertices = smpl_output.vertices.detach().cpu().numpy()  # (N, 6890, 3)
faces = smpl_model.faces

original_smpl_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False) for i in range(seq_len)]


##### Visualization of SMPL #####

viewer = Viewer()
fps = 30.0
dt = 1.0 / fps
seq_len = smpl_global_orient.shape[0]

# Log SMPL meshes
for frame_idx in tqdm(range(seq_len), desc="Logging SMPL meshes", dynamic_ncols=True):
    rr.set_time_sequence("frames", frame_idx)
    rr.set_time_seconds("sensor_time", frame_idx * dt)
    viewer.log_smplx_mesh(original_smpl_meshes[frame_idx], 0, label='example_motion_smpl')

# Log IMU orientations
IMU_device_names = ['left_wrist', 'right_wrist', 'left_thigh', 'right_thigh', 'head', 'pelvis']

for imu_idx in range(len(vi_mask)):
    print(f"Logging IMU trajectory for {IMU_device_names[imu_idx]} at vertex {vi_mask[imu_idx]}, joint {ji_mask[imu_idx]}...")
    
    # Log IMU trajectories
    # viewer.log_trajectory(imu_vertices[:, imu_idx], 0, label='smpl' + f'_{IMU_device_names[imu_idx]}', color_idx=imu_idx+1)
    
    # Log IMU orientations
    for frame_idx in tqdm(range(seq_len), desc="Logging IMU orientations", dynamic_ncols=True):
        rr.set_time_sequence("frames", frame_idx)
        rr.set_time_seconds("sensor_time", frame_idx * dt)
        transl = imu_vertices[frame_idx, imu_idx]
        orient = imu_joints_rot[frame_idx, imu_idx]
        viewer._log_orientations(orient, transl, 0, label='smpl' + f'_{IMU_device_names[imu_idx]}', arrow_scale=0.1)


# breakpoint()
################################################################################


# ##### Synthesized IMU trajectory of SMPL-X #####
# vi_mask = torch.tensor([4133, 6877, 229, 940, 4576, 7313])
# ji_mask = torch.tensor([1, 2, 15, 15, 18, 19])

# simplified_smplx_model = SimplifiedSMPLX(
#     model_path='/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smplx',
#     gender='neutral',
#     num_betas=10,
#     vert_mask=vi_mask.tolist(),
#     device='cuda',
# )

# smplx_output = simplified_smplx_model(pose=torch.zeros((1, 66), device='cuda'))
# rest_pelvis = smplx_output['joints_pos'][0, 0].detach().cpu().numpy()
# traj_corrected = smplx_transl - rest_pelvis

# with torch.no_grad():
#     smplx_output = simplified_smplx_model(
#         pose=torch.from_numpy(np.concatenate([smplx_global_orient, smplx_body_pose], axis=1)).cuda(),
#         betas=torch.from_numpy(smplx_betas).cuda(),
#         transl=torch.from_numpy(traj_corrected).cuda(),
#     )
#     imu_vertices = smplx_output['vertices'].detach().cpu().numpy()
#     imu_joints_rot = smplx_output['body_joints_rot'][:, ji_mask].detach().cpu().numpy()


# ##### Synthesized IMU readings #####
# p = torch.tensor(imu_vertices).cuda()
# R = torch.tensor(imu_joints_rot).cuda()

# a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
#     p, R, fps=30,
#     noise_raw_traj=False,
#     noise_syn_imu=False,
#     noise_est_orient=False,
#     skip_ESKF=True
# )


# ##### Get SMPL-X meshes #####
# seq_len = smplx_global_orient.shape[0]

# with torch.no_grad():
#     smplx_output = smplx_model(
#         pose_body=torch.tensor(smplx_body_pose, dtype=torch.float32).to(device),   # (N, 63)
#         root_orient=torch.tensor(smplx_global_orient, dtype=torch.float32).to(device),    # (N, 3)
#         trans=torch.tensor(smplx_transl - rest_pelvis_smplx, dtype=torch.float32).to(device),    # (N, 3)
#         betas=torch.zeros((smplx_global_orient.shape[0], 10), dtype=torch.float32).to(device)    # (N, 10)
#     )
# vertices = smplx_output.v.detach().cpu().numpy()
# faces = smplx_model.f.detach().cpu().numpy()

# original_smplx_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False) for i in range(seq_len)]


# ##### Visualization of SMPL-X #####

# viewer = Viewer()
# fps = 30.0
# dt = 1.0 / fps
# seq_len = smplx_global_orient.shape[0]

# # Log SMPL-X meshes
# for frame_idx in tqdm(range(seq_len), desc="Logging SMPL-X meshes", dynamic_ncols=True):
#     rr.set_time_sequence("frames", frame_idx)
#     rr.set_time_seconds("sensor_time", frame_idx * dt)
#     viewer.log_smplx_mesh(original_smplx_meshes[frame_idx], 0, label='example_motion_smplx')

# # Log IMU orientations
# IMU_device_names = ['left_hip', 'right_hip', 'left_ear', 'right_ear', 'left_elbow', 'right_elbow']

# for imu_idx in range(len(vi_mask)):
#     print(f"Logging IMU trajectory for {IMU_device_names[imu_idx]} at vertex {vi_mask[imu_idx]}, joint {ji_mask[imu_idx]}...")
    
#     # Log IMU trajectories
#     # viewer.log_trajectory(imu_vertices[:, imu_idx], 0, label='smplx' + f'_{IMU_device_names[imu_idx]}', color_idx=imu_idx+1)
    
#     # Log IMU orientations
#     for frame_idx in tqdm(range(seq_len), desc="Logging IMU orientations", dynamic_ncols=True):
#         rr.set_time_sequence("frames", frame_idx)
#         rr.set_time_seconds("sensor_time", frame_idx * dt)
#         transl = imu_vertices[frame_idx, imu_idx]
#         orient = imu_joints_rot[frame_idx, imu_idx]
#         viewer._log_orientations(orient, transl, 0, label='smplx' + f'_{IMU_device_names[imu_idx]}', arrow_scale=0.1)



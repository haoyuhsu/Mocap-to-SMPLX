import pickle
import os
import numpy as np
from smplx import SMPL, SMPLX
import torch
from tqdm import tqdm
import trimesh
import rerun as rr
import sys
sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/imu_synthesis/')
from viewer import Viewer
from scipy.spatial.transform import Rotation as R
from smplx import SMPL
from human_body_prior.body_model.body_model import BodyModel


device = 'cuda'


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


# Z-up to Y-up
# R_zup_to_yup = np.array([
#     [1,  0,  0],
#     [0,  0,  1],
#     [0, -1,  0]
# ], dtype=np.float32)

R_zup_to_yup = np.array([
    [-1, 0, 0],
    [0, 0, 1], 
    [0, 1, 0.]
], dtype=np.float32)




#################################################################
# Visualize the original SMPL motion of DIP-IMU dataset
#################################################################

# dip_imu_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/data/DIP_IMU'

# subject = 's_09'  # test subject (s_09 and s_10 are test subjects)
# motion_file = '01_a.pkl'  # example motion file

# example_dip_path = os.path.join(dip_imu_path, subject, motion_file)

# with open(example_dip_path, 'rb') as f:
#     dip_data = pickle.load(f, encoding='latin1')

# pose = dip_data['gt']  # (N, 72) axis-angle in original DIP format
# imu_acc = dip_data['imu_acc']  # (N, 17, 3) - accelerations from 17 IMUs
# imu_ori = dip_data['imu_ori']  # (N, 17, 3, 3) - orientations from 17 IMUs


# fps = 60.0
# TARGET_FPS = 30  # downsample to 30 FPS like in process.py
# step = max(1, round(fps / TARGET_FPS))


# # Downsample (skip first and last 6 frames like in process.py)
# pose = pose[6:-6:step]  # (N', 72)

# # Handle NaN values (fill with nearest neighbors)
# pose_tensor = torch.from_numpy(pose).float()
# for _ in range(4):
#     pose_tensor[1:].masked_scatter_(torch.isnan(pose_tensor[1:]), pose_tensor[:-1][torch.isnan(pose_tensor[1:])])
#     pose_tensor[:-1].masked_scatter_(torch.isnan(pose_tensor[:-1]), pose_tensor[1:][torch.isnan(pose_tensor[:-1])])

# if torch.isnan(pose_tensor).sum() == 0:
#     pose_aa = pose_tensor.view(-1, 24, 3).numpy().astype(np.float32)
    
#     # Extract global orientation and body pose
#     global_orient = pose_aa[:, 0, :]  # (N, 3) - root joint
#     body_pose = pose_aa[:, 1:, :]     # (N, 23, 3) - remaining joints
    
#     # DIP-IMU doesn't have translation, so we use zero translation
#     transl = np.zeros((pose_aa.shape[0], 3), dtype=np.float32) + rest_pelvis_smpl  # (N, 3)

#     # visualization
#     viewer = Viewer()
#     dt = 1.0 / TARGET_FPS
#     seq_len = global_orient.shape[0]
    
#     # Generate SMPL meshes
#     with torch.no_grad():
#         smpl_output = smpl_model(
#             betas=torch.zeros((seq_len, 10), dtype=torch.float32).to(device),
#             global_orient=torch.tensor(global_orient[:, None, :], dtype=torch.float32).to(device),
#             body_pose=torch.tensor(body_pose, dtype=torch.float32).to(device),
#             transl=torch.tensor(transl - rest_pelvis_smpl, dtype=torch.float32).to(device)
#         )
#     vertices = smpl_output.vertices.detach().cpu().numpy()  # (N, 6890, 3)
#     faces = smpl_model.faces
    
#     dip_smpl_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False) for i in range(seq_len)]
    
#     # Log SMPL meshes to rerun viewer
#     print("Logging DIP-IMU SMPL meshes to viewer...")
#     for frame_idx in tqdm(range(seq_len), desc="Logging DIP-IMU SMPL meshes", dynamic_ncols=True):
#         rr.set_time_sequence("frames", frame_idx)
#         rr.set_time_seconds("sensor_time", frame_idx * dt)
#         viewer.log_smplx_mesh(dip_smpl_meshes[frame_idx], 0, label='dip_imu_motion')
    
#     print(f"Visualization complete! Sequence: {subject}/{motion_file}")
# else:
#     print(f"Sequence {subject}/{motion_file} contains too many NaN values, skipped.")




##################################################################
# Visualize the converted SMPL-X motion and original SMPL motion of DIP-IMU dataset
##################################################################

processed_dipimu_data_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/data/DIP_IMU_processed/s_09/03_b.pkl'
with open(processed_dipimu_data_path, 'rb') as f:
    motion_data = pickle.load(f)

smplx_params = motion_data['smplx_params']
smpl_params = motion_data['orig_smpl_params']

smpl_global_orient = smpl_params['global_orient'].astype(np.float32)  # (N, 3) axis-angle
smpl_body_pose = smpl_params['body_pose'].astype(np.float32)  # (N, 69) axis-angle
smpl_transl = smpl_params['transl'].astype(np.float32)  # (N, 3)

smplx_global_orient = smplx_params['global_orient'].astype(np.float32)  # (N, 3) axis-angle
smplx_body_pose = smplx_params['body_pose'].astype(np.float32)  # (N, 63) axis-angle
smplx_transl = smplx_params['transl'].astype(np.float32)  # (N, 3)


viewer = Viewer()
fps = 60.0
dt = 1.0 / fps
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

# Log SMPL meshes
for frame_idx in tqdm(range(seq_len), desc="Logging SMPL meshes", dynamic_ncols=True):
    rr.set_time_sequence("frames", frame_idx)
    rr.set_time_seconds("sensor_time", frame_idx * dt)
    viewer.log_smplx_mesh(original_smpl_meshes[frame_idx], 0, label='example_motion')

with torch.no_grad():
    smplx_output = smplx_model(
        pose_body=torch.tensor(smplx_body_pose, dtype=torch.float32).to(device),   # (N, 63)
        root_orient=torch.tensor(smplx_global_orient, dtype=torch.float32).to(device),    # (N, 3)
        trans=torch.tensor(smplx_transl - rest_pelvis_smplx, dtype=torch.float32).to(device),    # (N, 3)
        betas=torch.zeros((smplx_global_orient.shape[0], 10), dtype=torch.float32).to(device)    # (N, 10)
    )
vertices = smplx_output.v.detach().cpu().numpy()
faces = smplx_model.f.detach().cpu().numpy()

original_smplx_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False) for i in range(seq_len)]

# Log SMPL-X meshes
for frame_idx in tqdm(range(seq_len), desc="Logging SMPL-X meshes", dynamic_ncols=True):
    rr.set_time_sequence("frames", frame_idx)
    rr.set_time_seconds("sensor_time", frame_idx * dt)
    viewer.log_smplx_mesh(original_smplx_meshes[frame_idx], 0, label='example_motion_smplx')




#################################################################
# Visualize the original SMPL motion of IMUPoser dataset
#################################################################

# example_imuposer_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/data/imuposer_dataset/P4/7. Basketball.pkl'

# with open(example_imuposer_path, 'rb') as f:
#     motion_data = pickle.load(f)

# pose = motion_data['pose']  # (N, 72) axis-angle

# global_orient = pose[:, :3]  # (N, 3) axis-angle
# body_pose = pose[:, 3:72].reshape(-1, 23, 3)  # (N, 23, 3) axis-angle
# transl = motion_data['trans']  # (N, 3)

# global_orient = global_orient.numpy().astype(np.float32)
# body_pose = body_pose.numpy().astype(np.float32)
# transl = transl.numpy().astype(np.float32)

# # Convert translation
# transl_yup = (R_zup_to_yup @ (transl + rest_pelvis_smpl).T).T   # get pelvis location

# # Convert global orientation (first 3 params)
# global_orient_yup = R.from_matrix(
#     np.einsum('ij,njk->nik', R_zup_to_yup, R.from_rotvec(global_orient).as_matrix())
# ).as_rotvec()

# global_orient = global_orient_yup
# transl = transl_yup

# viewer = Viewer()
# fps = 60.0
# dt = 1.0 / fps
# seq_len = global_orient.shape[0]

# with torch.no_grad():
#     smpl_output = smpl_model(
#         betas = torch.zeros((global_orient.shape[0], 10), dtype=torch.float32).to(device),
#         global_orient = torch.tensor(global_orient[:, None, :], dtype=torch.float32).to(device),
#         body_pose = torch.tensor(body_pose.reshape(seq_len, 23, 3), dtype=torch.float32).to(device),
#         transl = torch.tensor(transl - rest_pelvis_smpl, dtype=torch.float32).to(device)
#     )
# vertices = smpl_output.vertices.detach().cpu().numpy()  # (N, 6890, 3)
# faces = smpl_model.faces

# original_smpl_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False) for i in range(seq_len)]

# # Log SMPL meshes
# for frame_idx in tqdm(range(seq_len), desc="Logging SMPL meshes", dynamic_ncols=True):
#     rr.set_time_sequence("frames", frame_idx)
#     rr.set_time_seconds("sensor_time", frame_idx * dt)
#     viewer.log_smplx_mesh(original_smpl_meshes[frame_idx], 0, label='example_motion')




##################################################################
# Visualize the converted SMPL-X motion and original SMPL motion of IMUPoser dataset
##################################################################

# processed_imuposer_data_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/data/imuposer_dataset_processed/P4/7. Basketball.pkl'
# with open(processed_imuposer_data_path, 'rb') as f:
#     motion_data = pickle.load(f)

# smplx_params = motion_data['smplx_params']
# smpl_params = motion_data['orig_smpl_params']

# smpl_global_orient = smpl_params['global_orient'].astype(np.float32)  # (N, 3) axis-angle
# smpl_body_pose = smpl_params['body_pose'].astype(np.float32)  # (N, 69) axis-angle
# smpl_transl = smpl_params['transl'].astype(np.float32)  # (N, 3)

# smplx_global_orient = smplx_params['global_orient'].astype(np.float32)  # (N, 3) axis-angle
# smplx_body_pose = smplx_params['body_pose'].astype(np.float32)  # (N, 63) axis-angle
# smplx_transl = smplx_params['transl'].astype(np.float32)  # (N, 3)


# viewer = Viewer()
# fps = 60.0
# dt = 1.0 / fps
# seq_len = smpl_global_orient.shape[0]

# with torch.no_grad():
#     smpl_output = smpl_model(
#         betas = torch.zeros((smpl_global_orient.shape[0], 10), dtype=torch.float32).to(device),
#         global_orient = torch.tensor(smpl_global_orient[:, None, :], dtype=torch.float32).to(device),
#         body_pose = torch.tensor(smpl_body_pose.reshape(seq_len, 23, 3), dtype=torch.float32).to(device),
#         transl = torch.tensor(smpl_transl - rest_pelvis_smpl, dtype=torch.float32).to(device)
#     )
# vertices = smpl_output.vertices.detach().cpu().numpy()  # (N, 6890, 3)
# faces = smpl_model.faces

# original_smpl_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False) for i in range(seq_len)]

# # Log SMPL meshes
# for frame_idx in tqdm(range(seq_len), desc="Logging SMPL meshes", dynamic_ncols=True):
#     rr.set_time_sequence("frames", frame_idx)
#     rr.set_time_seconds("sensor_time", frame_idx * dt)
#     viewer.log_smplx_mesh(original_smpl_meshes[frame_idx], 0, label='example_motion')

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

# # Log SMPL-X meshes
# for frame_idx in tqdm(range(seq_len), desc="Logging SMPL-X meshes", dynamic_ncols=True):
#     rr.set_time_sequence("frames", frame_idx)
#     rr.set_time_seconds("sensor_time", frame_idx * dt)
#     viewer.log_smplx_mesh(original_smplx_meshes[frame_idx], 0, label='example_motion_smplx')
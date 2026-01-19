import numpy as np
import torch.nn as nn
import torch

# OPTITRACK_SKEL=[
#     'Hips',
#     'RightUpLeg','RightLeg','RightFoot','RightToeBase',# 'RightToeBase_Nub',
#     'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase',# 'LeftToeBase_Nub',
#     'Spine','Spine_1',
#     'RightShoulder','RightArm','RightForeArm','RightHand',
#         'RightHandPinky1','RightHandPinky2','RightHandPinky3','RightHandPinky3_Nub',
#         'RightHandRing1','RightHandRing2','RightHandRing3','RightHandRing3_Nub',
#         'RightHandMiddle1','RightHandMiddle2','RightHandMiddle3','RightHandMiddle3_Nub',
#         'RightHandIndex1','RightHandIndex2','RightHandIndex3','RightHandIndex3_Nub',
#         'RightHandThumb1','RightHandThumb2','RightHandThumb3','RightHandThumb3_Nub',
#     'LeftShoulder','LeftArm','LeftForeArm','LeftHand',
#         'LeftHandPinky1','LeftHandPinky2','LeftHandPinky3','LeftHandPinky3_Nub',
#         'LeftHandRing1','LeftHandRing2','LeftHandRing3','LeftHandRing3_Nub',
#         'LeftHandMiddle1','LeftHandMiddle2','LeftHandMiddle3','LeftHandMiddle3_Nub',
#         'LeftHandIndex1','LeftHandIndex2','LeftHandIndex3','LeftHandIndex3_Nub',
#         'LeftHandThumb1','LeftHandThumb2','LeftHandThumb3','LeftHandThumb3_Nub',
#     'Neck','Head',# 'Head_Nub'
# ] # 64 , 61 used,xxx_Nub is ignored hand_Nub can not be ignored

# SELECTED_JOINTS=np.concatenate(
#     [range(0,5),range(6,10),range(11,63)]
# )

# OPTITRACK_BODY=np.concatenate(
#     [range(0,15),range(35,39),range(59,61)]
# )
# OPTITRACK_HAND=np.concatenate(
#     [range(15,35),range(39,59)]
# )

# OPTITRACK_TO_SMPLX=np.array([
#     0,
#     2,5,8,11,
#     1,4,7,10,
#     3,9,
#     14,17,19,21,
#         46,47,48,75,
#         49,50,51,74,
#         43,44,45,73,
#         40,41,42,72,
#         52,53,54,71,
#     13,16,18,20,
#         31,32,33,70,
#         34,35,36,69,
#         28,29,30,68,
#         25,26,27,67,
#         37,38,39,66,
#     12,15
# ])


AMASS_SKEL=[
    'MidHip',
    'LHip', 'RHip',
    'spine1',
    'LKnee', 'RKnee',
    'spine2',
    'LAnkle', 'RAnkle',
    'spine3',
    'LFoot', 'RFoot',
    'Neck',
    'LCollar', 'Rcollar',
    'Head',
    'LShoulder', 'RShoulder',
    'LElbow', 'RElbow',
    'LWrist', 'RWrist',
]
SELECTED_JOINTS=np.concatenate(
    [range(0,22)]
)
AMASS_BODY=np.concatenate(
    [range(0,22)]
)
AMASS_TO_SMPLX=np.array(
    range(0,22)
)


VIRTUALHOME_SKEL=[
    'Hips',
    'LeftUpperLeg', 'RightUpperLeg', 'LeftLowerLeg', 'RightLowerLeg', 'LeftFoot', 'RightFoot',
    'Spine', 'Chest', 
    'Neck', 'Head', 
    'LeftShoulder', 'RightShoulder', 'LeftUpperArm', 'RightUpperArm', 'LeftLowerArm', 'RightLowerArm', 
    'LeftHand', 'RightHand', 'LeftToes', 'RightToes', 
    'LeftEye', 'RightEye', 'Jaw', 
    'LeftThumbProximal', 'LeftThumbIntermediate', 'LeftThumbDistal', 'LeftIndexProximal', 'LeftIndexIntermediate', 'LeftIndexDistal', 'LeftMiddleProximal', 'LeftMiddleIntermediate', 'LeftMiddleDistal', 'LeftRingProximal', 'LeftRingIntermediate', 'LeftRingDistal', 'LeftLittleProximal', 'LeftLittleIntermediate', 'LeftLittleDistal', 
    'RightThumbProximal', 'RightThumbIntermediate', 'RightThumbDistal', 'RightIndexProximal', 'RightIndexIntermediate', 'RightIndexDistal', 'RightMiddleProximal', 'RightMiddleIntermediate', 'RightMiddleDistal', 'RightRingProximal', 'RightRingIntermediate', 'RightRingDistal', 'RightLittleProximal', 'RightLittleIntermediate', 'RightLittleDistal', 
    'UpperChest', 
    'LastBone'
]

# AMASS_IDX_TO_VIRTUALHOME_JOINT_NAME_MAPPING = {
#     # AMASS idx: VirtualHome joint name
#     0:  'Hips',              # MidHip
#     1:  'LeftUpperLeg',      # LHip
#     2:  'RightUpperLeg',     # RHip
#     3:  'Spine',             # spine1
#     4:  'LeftLowerLeg',      # LKnee
#     5:  'RightLowerLeg',     # RKnee
#     6:  'Chest',             # spine2
#     7:  'LeftFoot',          # LAnkle
#     8:  'RightFoot',         # RAnkle
#     9:  'Chest',             # spine3
#     10: 'LeftToes',          # LFoot
#     11: 'RightToes',         # RFoot
#     12: 'Neck',              # Neck
#     13: 'LeftShoulder',      # LCollar
#     14: 'RightShoulder',     # Rcollar
#     15: 'Head',              # Head
#     16: 'LeftUpperArm',      # LShoulder
#     17: 'RightUpperArm',     # RShoulder
#     18: 'LeftLowerArm',      # LElbow
#     19: 'RightLowerArm',     # RElbow
#     20: 'LeftHand',          # LWrist
#     21: 'RightHand',         # RWrist
# }

AMASS_IDX_TO_VIRTUALHOME_JOINT_NAME_MAPPING = {
    # AMASS idx: VirtualHome joint name
    0:  'Hips',              # MidHip
    1:  'RightUpperLeg',     # LHip
    2:  'LeftUpperLeg',      # RHip
    3:  'Spine',             # spine1
    4:  'RightLowerLeg',     # LKnee
    5:  'LeftLowerLeg',      # RKnee
    6:  'Chest',             # spine2
    7:  'RightFoot',         # LAnkle
    8:  'LeftFoot',          # RAnkle
    9:  'Chest',             # spine3
    10: 'RightToes',         # LFoot
    11: 'LeftToes',          # RFoot
    12: 'Neck',              # Neck
    13: 'RightShoulder',     # LCollar
    14: 'LeftShoulder',      # Rcollar
    15: 'Head',              # Head
    16: 'RightUpperArm',     # LShoulder
    17: 'LeftUpperArm',      # RShoulder
    18: 'RightLowerArm',     # LElbow
    19: 'LeftLowerArm',      # RElbow
    20: 'RightHand',         # LWrist
    21: 'LeftHand',          # RWrist
}

def create_amass_joint_data(data_array, joint_names):
    """
    Convert VirtualHome joint data (56 joints) to AMASS format (22 joints).
    
    Args:
        data_array: (num_frames, 55*3) array of joint positions
        joint_names: List of VirtualHome joint names
    
    Returns:
        amass_data: (num_frames, 22, 3) array in AMASS format
    """
    num_frames = data_array.shape[0]
    amass_data = np.zeros((num_frames, 22, 3))
    
    # Create name to index mapping for VirtualHome joints
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    
    print(f"\nVirtualHome joints found: {len(name_to_idx)}")
    
    # Fill AMASS data using the mapping
    for amass_idx, vh_joint_name in AMASS_IDX_TO_VIRTUALHOME_JOINT_NAME_MAPPING.items():
        if vh_joint_name in name_to_idx:
            vh_idx = name_to_idx[vh_joint_name]
            amass_data[:, amass_idx, :] = data_array[:, vh_idx, :]
        else:
            print(f"Warning: VirtualHome joint '{vh_joint_name}' not found in data")
    
    return amass_data



class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)
        
if __name__=='__main__':
    print(SELECTED_JOINTS)
    print(len(SELECTED_JOINTS))
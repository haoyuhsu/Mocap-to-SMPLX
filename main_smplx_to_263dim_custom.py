"""
Convert IMUPoser SMPL-X predictions to 263-dimensional motion representation.

Two modes:
  1. Retarget (default): SMPL-X → SMPL (optimization) → joints → 263D
  2. Direct (--direct):   SMPL-X → joints → 263D  (skips SMPL retargeting)

The direct mode preserves full SMPL-X motion fidelity by extracting the
first 22 body joints directly from the SMPL-X forward pass.
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

sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/Mocap-to-SMPLX/HumanML3D')
from convert_from_85_to_263 import MotionConverter
from main_smplx_to_263dim_dump_data import SMPLXTo263Pipeline


class SMPLXTo263DirectPipeline:
    """Convert SMPL-X parameters directly to 263D without retargeting to SMPL.

    This avoids the optimisation-based SMPL-X → SMPL step, so the 263D
    representation faithfully reflects the original SMPL-X motion.

    NOTE: The skeleton offsets (bone lengths) are derived from the SMPL-X
    rest pose. If downstream models were trained on SMPL-based 263D data,
    the values will be slightly out-of-distribution in the IK-rotation
    channels (126 of 263 dims). For pred-vs-GT metrics where both sides
    use this pipeline the effect cancels out.
    """

    def __init__(self, smplx_model, rest_pelvis_smplx, skeleton_path, device='cuda'):
        self.device = device
        self.smplx_model = smplx_model
        self.rest_pelvis_smplx = rest_pelvis_smplx
        self.motion_converter = MotionConverter(skeleton_path, joints_num=22)

    # ------------------------------------------------------------------ #
    #  SMPL-X params → 22 body joints (no SMPL involved)                  #
    # ------------------------------------------------------------------ #
    def smplx_to_joints(self, smplx_params):
        """Extract 22 body joints directly from SMPL-X.

        Args:
            smplx_params: dict with numpy arrays
                - 'global_orient': (nf, 3)
                - 'body_pose':     (nf, 63)  — 21 joints × 3
                - 'transl':        (nf, 3)
                - 'betas':         (nf, 10)

        Returns:
            joints: numpy array (nf, 22, 3)
        """
        nf = smplx_params['global_orient'].shape[0]

        # Convert to tensors
        global_orient = torch.from_numpy(smplx_params['global_orient']).float().to(self.device)
        body_pose = torch.from_numpy(smplx_params['body_pose']).float().to(self.device)
        transl = torch.from_numpy(smplx_params['transl']).float().to(self.device)
        betas = torch.from_numpy(smplx_params['betas']).float().to(self.device)

        # BodyModel expects pose_body of shape (nf, n_joints*3)
        if body_pose.shape[-1] == 63:
            pose_body = body_pose  # already (nf, 63)
        else:
            pose_body = body_pose.reshape(nf, -1)

        with torch.no_grad():
            output = self.smplx_model(
                pose_body=pose_body,
                root_orient=global_orient,
                trans=transl - self.rest_pelvis_smplx,
                betas=betas,
            )

        # Jtr[:, :22] are the same 22 body joints as SMPL
        joints = output.Jtr[:, :22].detach().cpu().numpy()
        return joints

    # ------------------------------------------------------------------ #
    #  22 joints → 263D motion vector                                     #
    # ------------------------------------------------------------------ #
    def joints_to_263(self, joints):
        data, _, _, _ = self.motion_converter.convert_to_motion_vector(
            joints, feet_threshold=0.002
        )
        return data

    # ------------------------------------------------------------------ #
    #  Full pipeline                                                       #
    # ------------------------------------------------------------------ #
    def convert(self, smplx_params):
        """SMPL-X params → 263D (direct, no SMPL retargeting)."""
        joints = self.smplx_to_joints(smplx_params)
        motion_263 = self.joints_to_263(joints)
        return motion_263, None


def main():
    parser = argparse.ArgumentParser(
        description="Convert IMUPoser SMPL-X predictions to 263D motion representation."
    )
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing .pkl prediction files')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for 263D .npy files')
    parser.add_argument('--convert_gt', action='store_true', default=False,
                        help='Also convert ground truth to 263D')
    parser.add_argument('--direct', action='store_true', default=False,
                        help='Skip SMPL retargeting; extract joints directly from SMPL-X '
                             '(preserves motion fidelity, see class docstring for caveats)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load SMPL-X model
    smplx_model_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz'
    smplx_model = BodyModel(bm_fname=smplx_model_path, num_betas=10, model_type='smplx').to(device)
    smplx_model.eval()
    for p in smplx_model.parameters():
        p.requires_grad = False
    rest_pelvis_smplx = smplx_model().Jtr[0, 0]

    if args.direct:
        # ---------- Direct pipeline: SMPL-X → joints → 263D ---------- #
        # Skeleton offsets derived from SMPL-X rest pose
        skeleton_path = './smplx_offsets_temp.npy'
        if not os.path.exists(skeleton_path):
            default_smplx_output = smplx_model()
            np.save(skeleton_path,
                    default_smplx_output.Jtr[:, :22].detach().cpu().numpy())

        pipeline = SMPLXTo263DirectPipeline(
            smplx_model=smplx_model,
            rest_pelvis_smplx=rest_pelvis_smplx,
            skeleton_path=skeleton_path,
            device=device,
        )
        print("Using DIRECT pipeline (SMPL-X → joints → 263D, no SMPL retargeting)")
    else:
        # ---------- Retarget pipeline: SMPL-X → SMPL → joints → 263D -- #
        smpl_model_path = '/home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files/smpl/SMPL_NEUTRAL.pkl'
        smpl_model = SMPL(model_path=smpl_model_path).to(device)
        smpl_model.eval()
        for p in smpl_model.parameters():
            p.requires_grad = False
        default_smpl_output = smpl_model()
        rest_pelvis_smpl = default_smpl_output.joints[0, 0].detach().cpu().numpy()

        # Skeleton reference for 263D conversion
        skeleton_path = './smpl_offsets_temp.npy'
        if not os.path.exists(skeleton_path):
            np.save(skeleton_path, default_smpl_output.joints[:, :22].detach().cpu().numpy())

        pipeline = SMPLXTo263Pipeline(
            smplx_model=smplx_model, smpl_model=smpl_model,
            rest_pelvis_smplx=rest_pelvis_smplx, rest_pelvis_smpl=rest_pelvis_smpl,
            skeleton_path=skeleton_path, device=device,
        )
        print("Using RETARGET pipeline (SMPL-X → SMPL → joints → 263D)")

    # Gather .pkl files
    pkl_files = sorted(Path(args.pred_dir).glob('*.pkl'))
    print(f"Found {len(pkl_files)} .pkl files in {args.pred_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    for pkl_file in tqdm(pkl_files, desc="Converting"):
        output_file = Path(args.out_dir) / f"{pkl_file.stem}.npy"
        # if output_file.exists():
        #     continue

        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            pred = data['pred'] if 'pred' in data else data['recon']
            nf = pred['orient'].shape[0]

            smplx_params = {
                'global_orient': pred['orient'],
                'body_pose': pred['pose'][:, :21].reshape(nf, 63),
                'transl': pred['transl'],
                'betas': pred.get('betas', np.zeros((nf, 10), dtype=np.float32)),
            }
            pred_263, _ = pipeline.convert(smplx_params)
            np.save(output_file, pred_263.astype(np.float32))

            if args.convert_gt and 'gt' in data:
                gt_file = Path(args.out_dir) / f"{pkl_file.stem}_gt.npy"
                if not gt_file.exists():
                    gt = data['gt']
                    gt_params = {
                        'global_orient': gt['orient'],
                        'body_pose': gt['pose'][:, :21].reshape(nf, 63),
                        'transl': gt['transl'],
                        'betas': gt.get('betas', np.zeros((nf, 10), dtype=np.float32)),
                    }
                    gt_263, _ = pipeline.convert(gt_params)
                    np.save(gt_file, gt_263.astype(np.float32))

        except Exception as e:
            print(f"Error processing {pkl_file.stem}: {e}")

    print(f"\nDone. {len(pkl_files)} sequences → {args.out_dir}")


if __name__ == "__main__":
    main()

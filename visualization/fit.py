#!/usr/bin/env python3
"""
生成SMPL mesh用于可视化
参考MotionGPT的fit.py
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm
import smplx
import trimesh


class SMPLMeshGenerator:
    def __init__(self, 
                 smpl_model_path='./data/smplx_lockedhead_20230207/models_lockedhead',
                 device='cuda'):
        self.device = device
        
        # 加载SMPL-X模型（关闭PCA以简化）
        self.smpl_male = smplx.create(
            smpl_model_path,
            model_type='smplx',
            gender='male',
            use_face_contour=False,
            use_pca=False,  # 不使用PCA，使用完整的45维hand pose
            flat_hand_mean=True,  # 使用平直的手势
            ext='npz',
            num_betas=10,
        ).to(device)
        
        self.smpl_female = smplx.create(
            smpl_model_path,
            model_type='smplx',
            gender='female',
            use_face_contour=False,
            use_pca=False,
            flat_hand_mean=True,
            ext='npz',
            num_betas=10,
        ).to(device)
        
        self.faces = self.smpl_male.faces
    
    def generate_mesh(self, motion_data):
        """
        从motion数据生成SMPL mesh
        
        Args:
            motion_data: dict with keys:
                - transl: [T, 3]
                - global_orient: [T, 3, 3] 
                - body_pose: [T, 21, 3, 3]
                - betas: [10]
                - gender: 'male' or 'female'
        
        Returns:
            vertices: [T, 10475, 3] numpy array for SMPL-X
        """
        gender = motion_data.get('gender', 'male')
        smpl_model = self.smpl_male if gender == 'male' else self.smpl_female
        
        transl = torch.from_numpy(motion_data['transl']).float().to(self.device)
        global_orient = torch.from_numpy(motion_data['global_orient']).float().to(self.device)
        body_pose = torch.from_numpy(motion_data['body_pose']).float().to(self.device)
        betas = torch.from_numpy(motion_data['betas']).float().to(self.device)
        
        T = transl.shape[0]
        
        # 转换旋转矩阵为axis-angle
        from pytorch3d import transforms
        
        global_orient_aa = transforms.matrix_to_axis_angle(global_orient)  # [T, 3]
        body_pose_aa = transforms.matrix_to_axis_angle(body_pose.reshape(T*21, 3, 3))  # [T*21, 3]
        body_pose_aa = body_pose_aa.reshape(T, 21, 3)  # [T, 21, 3]
        
        # 扩展betas到每一帧
        betas = betas.unsqueeze(0).expand(T, -1)  # [T, 10]
        
        # SMPL-X body_pose需要 [T, 63] (21 joints * 3)
        body_pose_flat = body_pose_aa.reshape(T, 63)
        
        # 手部和脸部pose设为0 (自然姿势)
        # 因为use_pca=False，所以hand pose是45维 (15 joints * 3)
        left_hand_pose = torch.zeros(T, 45, device=self.device)
        right_hand_pose = torch.zeros(T, 45, device=self.device)
        jaw_pose = torch.zeros(T, 3, device=self.device)
        leye_pose = torch.zeros(T, 3, device=self.device)
        reye_pose = torch.zeros(T, 3, device=self.device)
        
        # SMPL-X还需要expression参数（面部表情）
        # 默认是10维
        expression = torch.zeros(T, 10, device=self.device)
        
        all_vertices = []
        
        # 分批处理以避免OOM
        batch_size = 64
        for i in tqdm(range(0, T, batch_size), desc="Generating SMPL meshes"):
            end_idx = min(i + batch_size, T)
            
            with torch.no_grad():
                output = smpl_model(
                    global_orient=global_orient_aa[i:end_idx],
                    body_pose=body_pose_flat[i:end_idx],
                    left_hand_pose=left_hand_pose[i:end_idx],
                    right_hand_pose=right_hand_pose[i:end_idx],
                    jaw_pose=jaw_pose[i:end_idx],
                    leye_pose=leye_pose[i:end_idx],
                    reye_pose=reye_pose[i:end_idx],
                    expression=expression[i:end_idx],
                    betas=betas[i:end_idx],
                    transl=transl[i:end_idx],
                )
            
            vertices = output.vertices.cpu().numpy()  # [B, 10475, 3] for SMPL-X
            all_vertices.append(vertices)
        
        all_vertices = np.concatenate(all_vertices, axis=0)  # [T, 10475, 3]
        
        return all_vertices
    
    def save_mesh_sequence(self, vertices, output_dir, save_npy=True, save_ply=True):
        """
        保存mesh序列
        
        Args:
            vertices: [T, 6890, 3]
            output_dir: 输出目录
            save_npy: 是否保存npy
            save_ply: 是否保存ply文件
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整的npy
        if save_npy:
            npy_path = output_dir / 'mesh_vertices.npy'
            np.save(npy_path, vertices)
            print(f"Saved mesh vertices to {npy_path}")
        
        # 保存每一帧的ply
        if save_ply:
            ply_dir = output_dir / 'ply'
            ply_dir.mkdir(exist_ok=True)
            
            for frame_idx in tqdm(range(len(vertices)), desc="Saving PLY files"):
                mesh = trimesh.Trimesh(
                    vertices=vertices[frame_idx],
                    faces=self.faces,
                    process=False
                )
                
                ply_path = ply_dir / f'frame_{frame_idx:04d}.ply'
                mesh.export(ply_path)
            
            print(f"Saved {len(vertices)} PLY files to {ply_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate SMPL meshes from motion')
    parser.add_argument('--dir', type=str, required=True,
                      help='Directory containing motion pkl files')
    parser.add_argument('--save_folder', type=str, required=True,
                      help='Output folder for meshes')
    parser.add_argument('--smpl_model_path', type=str,
                      default='./data/smplx_lockedhead_20230207/models_lockedhead',
                      help='Path to SMPL models')
    parser.add_argument('--cuda', action='store_true',
                      help='Use CUDA')
    parser.add_argument('--save_npy', type=int, default=1,
                      help='Save npy file')
    parser.add_argument('--save_ply', type=int, default=1,
                      help='Save ply files')
    
    args = parser.parse_args()
    
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    
    # 初始化mesh生成器
    print("Initializing SMPL mesh generator...")
    generator = SMPLMeshGenerator(
        smpl_model_path=args.smpl_model_path,
        device=device
    )
    
    # 查找所有pkl文件
    input_dir = Path(args.dir)
    pkl_files = list(input_dir.glob('*.pkl'))
    
    if len(pkl_files) == 0:
        print(f"No pkl files found in {input_dir}")
        return
    
    print(f"Found {len(pkl_files)} pkl files")
    
    # 处理每个文件
    for pkl_file in pkl_files:
        print(f"\nProcessing {pkl_file.name}...")
        
        # 加载motion数据
        with open(pkl_file, 'rb') as f:
            motion_data = pickle.load(f)
        
        print(f"Motion frames: {motion_data['transl'].shape[0]}")
        print(f"Gender: {motion_data.get('gender', 'male')}")
        
        # 生成mesh
        vertices = generator.generate_mesh(motion_data)
        print(f"Generated mesh vertices shape: {vertices.shape}")
        
        # 保存
        output_name = pkl_file.stem
        output_dir = Path(args.save_folder) / output_name
        
        generator.save_mesh_sequence(
            vertices,
            output_dir,
            save_npy=bool(args.save_npy),
            save_ply=bool(args.save_ply)
        )
        
        print(f"✅ Completed {pkl_file.name}")
    
    print(f"\n✅ All done! Meshes saved to {args.save_folder}")


if __name__ == '__main__':
    main()
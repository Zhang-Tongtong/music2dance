#!/usr/bin/env python3
"""
Music-to-Dance 推理和可视化脚本
从音乐生成舞蹈动作并渲染为视频
"""

import torch
import numpy as np
from pathlib import Path
import pickle
import sys
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from model.mld_vae import AutoMldVae
from model.mld_denoiser import DenoiserTransformer, DenoiserMLP
from data_loaders.humanml.data.dataset_finedance import FineDanceDataset
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion import gaussian_diffusion as gd
from utils.smpl_utils import PrimitiveUtility


class MusicToDanceGenerator:
    def __init__(self, 
                 vae_path,
                 denoiser_path,
                 device='cuda'):
        self.device = device
        
        # 加载VAE
        print("Loading VAE...")
        vae_checkpoint = torch.load(vae_path, map_location=device)
        vae_args = vae_checkpoint.get('args', None)
        
        self.vae = AutoMldVae(
            nfeats=276,
            latent_dim=[1, 256],
            h_dim=256,
            num_layers=7,
            ff_size=1024,
            num_heads=4,
            arch='all_encoder',
        ).to(device)
        
        self.vae.load_state_dict(vae_checkpoint['model_state_dict'])
        self.vae.eval()
        
        print(f"VAE loaded. Latent mean: {self.vae.latent_mean}, std: {self.vae.latent_std}")
        
        # 加载Denoiser
        print("Loading Denoiser...")
        denoiser_checkpoint = torch.load(denoiser_path, map_location=device)
        
        # 从checkpoint自动检测模型类型
        state_dict = denoiser_checkpoint['model_state_dict']
        
        # 检查是MLP还是Transformer
        if 'mlp.layers.0.layers.0.weight' in state_dict:
            model_type = 'mlp'
            print("Detected model type: MLP")
        elif 'seqTransEncoder.layers.0.self_attn.in_proj_weight' in state_dict:
            model_type = 'transformer'
            print("Detected model type: Transformer")
        else:
            # 默认尝试MLP
            model_type = 'mlp'
            print("Cannot detect model type, defaulting to MLP")
        
        if model_type == 'mlp':
            self.denoiser = DenoiserMLP(
                h_dim=512,
                n_blocks=2,
                dropout=0.1,
                activation='gelu',
                music_dim=35,
                history_shape=(2, 276),
                noise_shape=(1, 256),
                cond_mask_prob=0.0,  # 推理时不mask
            ).to(device)
        else:
            self.denoiser = DenoiserTransformer(
                h_dim=512,
                ff_size=1024,
                num_layers=8,
                num_heads=4,
                dropout=0.1,
                activation='gelu',
                music_dim=35,
                history_shape=(2, 276),
                noise_shape=(1, 256),
                cond_mask_prob=0.0,
            ).to(device)
        
        self.denoiser.load_state_dict(denoiser_checkpoint['model_state_dict'])
        self.denoiser.eval()
        
        print("Denoiser loaded successfully")
        
        # 创建Diffusion
        self.diffusion = self._create_diffusion()
        
        # 初始化工具
        self.primitive_utility = PrimitiveUtility(device=device, body_type='smplx')
        
        # 数据归一化参数（需要从训练数据集加载）
        self.motion_mean = None
        self.motion_std = None
    
    def _create_diffusion(self):
        """创建diffusion模型"""
        steps = 1000
        noise_schedule = 'cosine'
        betas = gd.get_named_beta_schedule(noise_schedule, steps)
        
        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, [steps]),
            betas=betas,
            model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.FIXED_SMALL,
            loss_type=gd.LossType.MSE,
        )
    
    def set_normalization(self, motion_mean, motion_std):
        """设置归一化参数"""
        self.motion_mean = torch.from_numpy(motion_mean).to(self.device).float()
        self.motion_std = torch.from_numpy(motion_std).to(self.device).float()
    
    def normalize(self, motion):
        """归一化motion"""
        return (motion - self.motion_mean) / self.motion_std
    
    def denormalize(self, motion):
        """反归一化motion"""
        return motion * self.motion_std + self.motion_mean
    
    @torch.no_grad()
    def generate(self, music_features, num_primitives=None, history_motion=None):
        """
        从音乐特征生成舞蹈动作
        
        Args:
            music_features: [T, 35] 音乐特征
            num_primitives: 生成的primitive数量
            history_motion: 初始历史动作 [H, 276]
        
        Returns:
            motion: [T, 276] 生成的动作序列
        """
        history_length = 2
        future_length = 8
        
        if num_primitives is None:
            # 根据音乐长度计算primitive数量
            num_primitives = len(music_features) // future_length
        
        # 初始化历史
        if history_motion is None:
            # 使用站立姿势作为初始历史
            history_motion = torch.zeros(history_length, 276).to(self.device)
        else:
            history_motion = torch.from_numpy(history_motion).float().to(self.device)
        
        # 归一化历史
        history_motion = self.normalize(history_motion)
        
        all_motion = []
        
        for i in range(num_primitives):
            print(f"Generating primitive {i+1}/{num_primitives}...")
            
            # 提取当前primitive的音乐特征
            start_idx = i * future_length
            end_idx = min((i + 1) * future_length, len(music_features))
            music_segment = music_features[start_idx:end_idx]
            
            # 平均池化得到音乐embedding
            music_emb = torch.from_numpy(music_segment.mean(0)).float().to(self.device).unsqueeze(0)  # [1, 35]
            
            # 准备condition
            y = {
                'music_embedding': music_emb,
                'history_motion_normalized': history_motion.unsqueeze(0),  # [1, H, 276]
            }
            
            # 包装成model_kwargs格式
            model_kwargs = {'y': y}
            
            # Diffusion采样生成latent
            shape = (1, 1, 256)  # [B, T=1, latent_dim]
            
            latent = self.diffusion.p_sample_loop(
                self.denoiser,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
            )  # [1, 1, 256]
            
            latent = latent.permute(1, 0, 2)  # [1, 1, 256] -> [T=1, B=1, 256]
            
            # VAE解码
            future_motion = self.vae.decode(
                latent, 
                history_motion.unsqueeze(0),  # [1, H, 276]
                nfuture=future_length,
                scale_latent=1
            )  # [1, F, 276]
            
            future_motion = future_motion.squeeze(0)  # [F, 276]
            
            # 添加到结果
            all_motion.append(future_motion)
            
            # 更新历史（使用最后的history_length帧）
            history_motion = future_motion[-history_length:]
        
        # 拼接所有motion
        motion = torch.cat(all_motion, dim=0)  # [Total_T, 276]
        
        # 反归一化
        motion = self.denormalize(motion)
        # 添加Z坐标修正
        min_z = motion[:, 2].min().item()
        if min_z < 0:
            motion[:, 2] -= min_z - 0.5  # 抬升到地面以上
        
        return motion.cpu().numpy()


def save_motion_for_visualization(motion, output_path, gender='male'):
    """
    保存motion为可视化格式
    
    Args:
        motion: [T, 276] numpy array
        output_path: 输出路径
        gender: 'male' or 'female'
    """
    # 解析motion
    T = motion.shape[0]
    
    transl = motion[:, 0:3]
    poses_6d = motion[:, 3:135]  # [T, 132]
    
    # 转换6D到旋转矩阵
    from pytorch3d import transforms
    
    poses_6d_t = torch.from_numpy(poses_6d).float()
    
    # 【关键修复】对6D表示进行Gram-Schmidt正交化
    # 确保每个6D向量代表有效的旋转
    def gram_schmidt_6d(x):
        """Gram-Schmidt正交化6D表示"""
        a1 = x[..., :3]
        a2 = x[..., 3:6]
        
        # 归一化第一个向量
        b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
        
        # 正交化第二个向量
        dot = torch.sum(a2 * b1, dim=-1, keepdim=True)
        b2 = a2 - dot * b1
        b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
        
        return torch.cat([b1, b2], dim=-1)
    
    # Reshape, 正交化, 再reshape回来
    poses_6d_reshaped = poses_6d_t.reshape(T, 22, 6)  # 1 global + 21 body
    poses_6d_normalized = gram_schmidt_6d(poses_6d_reshaped)
    poses_6d_t = poses_6d_normalized.reshape(T, 132)
    
    # 分离global_orient和body_pose
    global_orient_6d = poses_6d_t[:, :6]  # [T, 6]
    body_pose_6d = poses_6d_t[:, 6:].reshape(T, 21, 6)  # [T, 21, 6]
    
    # 转换为旋转矩阵
    global_orient = transforms.rotation_6d_to_matrix(global_orient_6d)  # [T, 3, 3]
    body_pose = transforms.rotation_6d_to_matrix(body_pose_6d)  # [T, 21, 3, 3]
    
    # 保存为pickle
    result = {
        'gender': gender,
        'transl': transl,
        'global_orient': global_orient.numpy(),
        'body_pose': body_pose.numpy(),
        'betas': np.zeros((10,)),  # 使用零beta
        'fps': 30,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Motion saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate dance from music')
    parser.add_argument('--vae_path', type=str, required=True,
                      help='Path to VAE checkpoint')
    parser.add_argument('--denoiser_path', type=str, required=True,
                      help='Path to Denoiser checkpoint')
    parser.add_argument('--music_path', type=str, required=True,
                      help='Path to music features (.npy)')
    parser.add_argument('--output_dir', type=str, default='./results/music_dance',
                      help='Output directory')
    parser.add_argument('--num_primitives', type=int, default=None,
                      help='Number of primitives to generate')
    parser.add_argument('--gender', type=str, default='male',
                      choices=['male', 'female'])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化生成器
    print("Initializing generator...")
    generator = MusicToDanceGenerator(
        vae_path=args.vae_path,
        denoiser_path=args.denoiser_path,
        device=args.device
    )
    
    # 加载归一化参数（从训练数据集）
    print("Loading normalization parameters...")
    dataset = FineDanceDataset(
        dataset_path='./data/finedance',
        cfg_path='./config_files/config_hydra/motion_primitive/finedance_h2_f8_r8.yaml',
        split='train',
        device='cpu'
    )
    generator.set_normalization(dataset.motion_mean, dataset.motion_std)
    
    # 加载音乐特征
    print(f"Loading music features from {args.music_path}...")
    music_features = np.load(args.music_path)  # [T, 35]
    print(f"Music features shape: {music_features.shape}")
    
    # 生成舞蹈
    print("Generating dance...")
    motion = generator.generate(
        music_features,
        num_primitives=args.num_primitives
    )
    print(f"Generated motion shape: {motion.shape}")
    
    # 保存结果
    output_path = output_dir / 'generated_dance.pkl'
    save_motion_for_visualization(motion, output_path, gender=args.gender)
    
    print(f"\n✅ Generation completed!")
    print(f"Motion saved to: {output_path}")
    print(f"\nTo visualize, run:")
    print(f"python visualize/vis_seq.py --motion_path {output_path} --output_path {output_dir}/dance.mp4")


if __name__ == '__main__':
    main()
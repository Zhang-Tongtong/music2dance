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
# from smplx import SMPLX

# def rebuild_276_consistent(motion_276: torch.Tensor, smpl_model) -> torch.Tensor:
#     """
#     motion_276: [T,276]  (原始空间 or 归一化空间都行，但你必须传入“原始空间”的才有物理意义)
#     return: [T,276]  保证 joints 和 delta 与 transl+poses 一致
#     """
#     import torch
#     from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

#     T = motion_276.shape[0]
#     out = motion_276.clone()

#     # 1) 取 transl + poses_6d
#     transl = out[:, 0:3]
#     poses6d = out[:, 3:135].reshape(T, 22, 6)  # global(1)+body(21)

#     # 2) 6D 正交化（一定要做，避免数值漂）
#     def gram_schmidt_6d(x):
#         a1 = x[..., 0:3]
#         a2 = x[..., 3:6]
#         b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
#         dot = torch.sum(a2 * b1, dim=-1, keepdim=True)
#         b2 = a2 - dot * b1
#         b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
#         return torch.cat([b1, b2], dim=-1)

#     poses6d = gram_schmidt_6d(poses6d)

#     # 3) 转 rotation matrix，喂 SMPL-X 重新算 joints
#     R = rotation_6d_to_matrix(poses6d.reshape(-1, 6)).reshape(T, 22, 3, 3)
#     global_orient = R[:, 0]          # [T,3,3]
#     body_pose = R[:, 1:]             # [T,21,3,3]

#     # smpl_model forward：取 joints (22,3)
#     # 这里假设你 smpl_model 的 forward 支持 global_orient/body_pose/transl
#     smpl_out = smpl_model(global_orient=global_orient, body_pose=body_pose, transl=transl)
#     joints = smpl_out.joints[:, :22, :]  # [T,22,3]

#     # 4) 写回 joints (66)
#     out[:, 135:201] = joints.reshape(T, 66)

#     # 5) 重新计算 deltas（用相邻帧差分，第一帧 delta=0）
#     transl_delta = torch.zeros_like(transl)
#     transl_delta[1:] = transl[1:] - transl[:-1]

#     rot6d = matrix_to_rotation_6d(R.reshape(-1, 3, 3)).reshape(T, 22, 6)
#     rot6d_delta = torch.zeros_like(rot6d)
#     rot6d_delta[1:] = rot6d[1:] - rot6d[:-1]

#     joints_delta = torch.zeros_like(joints)
#     joints_delta[1:] = joints[1:] - joints[:-1]

#     out[:, 201:204] = transl_delta
#     out[:, 204:210] = rot6d_delta[:, 0].reshape(T, 6)      # global delta
#     out[:, 210:276] = joints_delta.reshape(T, 66)

#     # 6) 把（可能被正交化后的）poses6d 写回原向量，保持一致
#     out[:, 3:135] = rot6d.reshape(T, 132)

#     return out

class MusicToDanceGenerator:
    def __init__(self, 
                 vae_path,
                 denoiser_path,
                 device='cuda'):
        self.device = device
        
        # 加载VAE
        print("Loading VAE...")
        # vae_checkpoint = torch.load(vae_path, map_location=device)
        # vae_args = vae_checkpoint.get('args', None)
        
        # self.vae = AutoMldVae(
        #     nfeats=276,
        #     latent_dim=[1, 256],
        #     h_dim=256,
        #     num_layers=7,
        #     ff_size=1024,
        #     num_heads=4,
        #     arch='all_encoder',
        # ).to(device)
        
        # self.vae.load_state_dict(vae_checkpoint['model_state_dict'])
        # self.vae.eval()
        
        # print(f"VAE loaded. Latent mean: {self.vae.latent_mean}, std: {self.vae.latent_std}")
        vae_checkpoint = torch.load(vae_path, map_location=device)

        self.vae = AutoMldVae(
            nfeats=276,
            latent_dim=[1, 256],
            h_dim=256,
            num_layers=7,
            ff_size=1024,
            num_heads=4,
            arch='all_encoder',
        ).to(device)

        model_state_dict = vae_checkpoint['model_state_dict']
        # 和训练脚本一致：如果checkpoint缺键，给默认值
        if 'latent_mean' not in model_state_dict:
            model_state_dict['latent_mean'] = torch.tensor(0)
        if 'latent_std' not in model_state_dict:
            model_state_dict['latent_std'] = torch.tensor(1)

        self.vae.load_state_dict(model_state_dict)
        # 和训练脚本一致：强制写回（训练脚本明确说 load_state_dict 可能不生效）
        self.vae.latent_mean = model_state_dict['latent_mean']
        self.vae.latent_std = model_state_dict['latent_std']

        self.vae.eval()
        print(f"VAE loaded. Latent mean: {self.vae.latent_mean}, std: {self.vae.latent_std}")
        # ===== HARD GUARD: latent_std must be > 0 =====
        try:
            _ls = float(self.vae.latent_std.item())
        except Exception:
            _ls = float(self.vae.latent_std)  # fallback
        if _ls == 0.0:
            print("[WARN] vae.latent_std is 0. Force disabling latent scaling in inference.")
        # =============================================
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
        # self.smpl_model = SMPLX(
        #     model_path="./data/smplx_lockedhead_20230207/models_lockedhead/smplx",
        #     gender="neutral",
        #     use_pca=False,
        #     num_pca_comps=12,
        #     batch_size=1
        # ).to(self.device)
        # self.smpl_model.eval()
        
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
        
        # 初始化历史（必须与训练分布对齐：归一化后应接近0）
        if self.motion_mean is None or self.motion_std is None:
            raise RuntimeError("Normalization params not set. Call set_normalization() before generate().")

        if history_motion is None:
            # 用训练集均值作为“中性姿态”历史：normalize后严格为0
            mean_276 = self.motion_mean.reshape(-1)  # 兼容 [276] 或 [1,276]
            assert mean_276.numel() == 276, f"motion_mean should have 276 elements, got {mean_276.numel()}"
            history_motion = mean_276.unsqueeze(0).repeat(history_length, 1)  # [H,276]
        else:
            history_motion = torch.from_numpy(history_motion).float().to(self.device)
            # 如果用户传入的是未归一化的原始276维，这里统一按原始处理并归一化
            # （如果你确认传入已经是归一化的，请不要走这个分支）
        # 归一化历史（训练时history就是归一化后的）
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
                scale_latent=0
            )  # [1, F, 276]
            # # === 关键：把未来片段先从“归一化空间”还原到原始空间，再做自洽重建，再归一化回去 ===
            # fm = future_motion[0]  # [T,276] 仍是归一化空间
            # fm_denorm = self.denormalize(fm)  # -> 原始空间
            # fm_denorm = rebuild_276_consistent(fm_denorm, self.smpl_model)  # 重新算 joints/deltas，保证自洽
            # fm = self.normalize(fm_denorm)  # 再归一化回去
            # future_motion = fm.unsqueeze(0)
            # # ==========================================================================================
            # # ========= 推理阶段根治：对VAE输出的6D做正交化 + 幅度限制 =========
            # # future_motion: [1, T, 276]，仍在“归一化空间”
            # fm = future_motion[0]  # [T,276]

            # # 取出6D：global(6) + body(126) 共132维
            # poses6d = fm[:, 3:135].reshape(-1, 22, 6)  # [T,22,6]

            # def gram_schmidt_6d(x):
            #     a1 = x[..., 0:3]
            #     a2 = x[..., 3:6]
            #     b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
            #     dot = torch.sum(a2 * b1, dim=-1, keepdim=True)
            #     b2 = a2 - dot * b1
            #     b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
            #     return torch.cat([b1, b2], dim=-1)

            # poses6d = gram_schmidt_6d(poses6d)

            # # 可选：角度幅度限制（建议保留，避免爆角再次出现）
            # from pytorch3d import transforms
            # R = transforms.rotation_6d_to_matrix(poses6d.reshape(-1,6)).reshape(-1,22,3,3)
            # aa = transforms.matrix_to_axis_angle(R.reshape(-1,3,3)).reshape(-1,22,3)

            # MAX_BODY = 0.6
            # MAX_GLOBAL = 2.0
            # angle = torch.norm(aa, dim=-1, keepdim=True) + 1e-8
            # max_angle = torch.ones_like(angle) * MAX_BODY
            # max_angle[:, 0:1, :] = MAX_GLOBAL  # joint0(global)稍大
            # scale = torch.clamp(max_angle / angle, max=1.0)
            # aa = aa * scale
            # R = transforms.axis_angle_to_matrix(aa.reshape(-1,3)).reshape(-1,22,3,3)

            # # 转回6D写回motion
            # poses6d = transforms.matrix_to_rotation_6d(R.reshape(-1,3,3)).reshape(-1,22,6)
            # fm[:, 3:135] = poses6d.reshape(-1,132)

            # future_motion = fm.unsqueeze(0)
            # # ============================================================
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
    
    # # 【关键修复】对6D表示进行Gram-Schmidt正交化
    # # 确保每个6D向量代表有效的旋转
    # def gram_schmidt_6d(x):
    #     """Gram-Schmidt正交化6D表示"""
    #     a1 = x[..., :3]
    #     a2 = x[..., 3:6]
        
    #     # 归一化第一个向量
    #     b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
        
    #     # 正交化第二个向量
    #     dot = torch.sum(a2 * b1, dim=-1, keepdim=True)
    #     b2 = a2 - dot * b1
    #     b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
        
    #     return torch.cat([b1, b2], dim=-1)
    
    # # Reshape, 正交化, 再reshape回来
    # poses_6d_reshaped = poses_6d_t.reshape(T, 22, 6)  # 1 global + 21 body
    # poses_6d_normalized = gram_schmidt_6d(poses_6d_reshaped)
    # poses_6d_t = poses_6d_normalized.reshape(T, 132)
    
    # 分离global_orient和body_pose
    global_orient_6d = poses_6d_t[:, :6]  # [T, 6]
    body_pose_6d = poses_6d_t[:, 6:].reshape(T, 21, 6)  # [T, 21, 6]
    
    # 转换为旋转矩阵
    global_orient = transforms.rotation_6d_to_matrix(global_orient_6d)  # [T, 3, 3]
    body_pose = transforms.rotation_6d_to_matrix(body_pose_6d)  # [T, 21, 3, 3]
    # # ========= 修复：限制axis-angle幅度，防止关节爆角导致SMPL扭曲 =========
    # # 把旋转矩阵转成axis-angle
    # global_aa = transforms.matrix_to_axis_angle(global_orient)                 # [T,3]
    # body_aa = transforms.matrix_to_axis_angle(body_pose.reshape(T*21, 3, 3))   # [T*21,3]
    # body_aa = body_aa.reshape(T, 21, 3)                                        # [T,21,3]

    # def clamp_axis_angle(aa, max_angle):
    #     # aa: [..., 3]
    #     angle = torch.norm(aa, dim=-1, keepdim=True) + 1e-8
    #     scale = torch.clamp(max_angle / angle, max=1.0)
    #     return aa * scale

    # # 经验上：global可以稍大，body关节严格一些（你也可以后面调）
    # MAX_GLOBAL = 1.5   # 弧度，约114°
    # MAX_BODY   = 0.6   

    # global_aa = clamp_axis_angle(global_aa, MAX_GLOBAL)
    # body_aa   = clamp_axis_angle(body_aa,   MAX_BODY)

    # # 转回旋转矩阵（写进pkl的就是被限制后的姿态）
    # global_orient = transforms.axis_angle_to_matrix(global_aa)                 # [T,3,3]
    # body_pose = transforms.axis_angle_to_matrix(body_aa.reshape(T*21, 3)).reshape(T, 21, 3, 3)
    # # ============================================================================
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
    # from utils.rot6d import enforce_pose6d_132

    # # motion: [T,276] torch
    # pose6d = motion[:, 3:135].reshape(-1,22,6)
    # n1 = pose6d[...,:3].norm(dim=-1).mean().item()
    # n2 = pose6d[...,3:].norm(dim=-1).mean().item()
    # dot = (pose6d[...,:3]*pose6d[...,3:]).sum(dim=-1).abs().mean().item()
    # print("[DEBUG] gen pose6d stats:", n1, n2, dot)
    # 保存结果
    output_path = output_dir / 'generated_dance.pkl'
    save_motion_for_visualization(motion, output_path, gender=args.gender)
    
    print(f"\n✅ Generation completed!")
    print(f"Motion saved to: {output_path}")
    print(f"\nTo visualize, run:")
    print(f"python visualize/vis_seq.py --motion_path {output_path} --output_path {output_dir}/dance.mp4")


if __name__ == '__main__':
    main()
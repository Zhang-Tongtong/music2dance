#!/usr/bin/env python3
"""
Music2Dance 测试和调试脚本
在远程服务器上运行此脚本来验证pipeline并诊断问题
"""

import torch
import numpy as np
import sys
from pathlib import Path
import argparse
import traceback

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_checkpoint_existence():
    """检查checkpoint文件是否存在"""
    print("=" * 70)
    print("1. Testing Checkpoint Existence")
    print("=" * 70)

    checkpoints = {
        'VAE': './mvae/mvae_finedance_h2_f8_r8/checkpoint_20000.pt',
        'VAE_200k': './mvae/mvae_finedance_h2_f8_r8/checkpoint_200000.pt',
        'Denoiser': './mld_denoiser/mld_finedance_h2_f8_r8/checkpoint_300000.pt',
    }

    results = {}
    for name, path in checkpoints.items():
        exists = Path(path).exists()
        results[name] = exists
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"  {name:20s} {status}")
        print(f"    Path: {path}")

    # 检查是否有任何checkpoint存在
    any_exists = any(results.values())
    print(f"\nSummary: {'✓ At least one checkpoint found' if any_exists else '✗ No checkpoints found'}")

    return results, any_exists


def test_data_directory():
    """检查数据目录是否存在"""
    print("\n" + "=" * 70)
    print("2. Testing Data Directory")
    print("=" * 70)

    data_paths = [
        './data/finedance',
        './data/finedance/motion',
        './data/finedance/music_npynew',
        './data/finedance/label_json',
    ]

    results = {}
    for path in data_paths:
        exists = Path(path).exists()
        results[path] = exists
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"  {path:50s} {status}")

    any_exists = any(results.values())
    print(f"\nSummary: {'✓ At least one data directory found' if any_exists else '✗ No data directories found'}")

    return results, any_exists


def test_vae_loading():
    """测试VAE模型加载"""
    print("\n" + "=" * 70)
    print("3. Testing VAE Model Loading")
    print("=" * 70)

    from model.mld_vae import AutoMldVae

    checkpoint_path = './mvae/mvae_finedance_h2_f8_r8/checkpoint_20000.pt'
    if not Path(checkpoint_path).exists():
        print("  ✗ VAE checkpoint not found, skipping test")
        return False, None

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        print(f"  ✓ Checkpoint loaded")

        # 检测latent_dim
        latent_dim = None
        if 'encoder_latent_proj.weight' in state_dict:
            latent_dim = state_dict['encoder_latent_proj.weight'].shape[0]
            print(f"  ✓ Detected latent_dim from encoder: {latent_dim}")
        elif 'decoder_latent_proj.weight' in state_dict:
            latent_dim = state_dict['decoder_latent_proj.weight'].shape[1]
            print(f"  ✓ Detected latent_dim from decoder: {latent_dim}")
        else:
            print("  ✗ Could not detect latent_dim from state_dict")
            return False, None

        # 创建模型
        vae = AutoMldVae(
            nfeats=276,
            latent_dim=[1, latent_dim],
            h_dim=256,
            num_layers=7,
            ff_size=1024,
            num_heads=4,
            arch='all_encoder',
        ).to(device)

        vae.load_state_dict(state_dict)
        vae.eval()

        print(f"  ✓ VAE model created and loaded")
        print(f"    latent_mean: {vae.latent_mean.item():.6f}")
        print(f"    latent_std: {vae.latent_std.item():.6f}")

        # 测试前向传播
        batch_size = 2
        history_length = 2
        future_length = 8
        nfeats = 276

        history_motion = torch.randn(batch_size, history_length, nfeats).to(device)
        future_motion = torch.randn(batch_size, future_length, nfeats).to(device)

        with torch.no_grad():
            latent, dist = vae.encode(future_motion=future_motion, history_motion=history_motion)
            print(f"\n  ✓ Encode test passed")
            print(f"    Input: future_motion {future_motion.shape}, history_motion {history_motion.shape}")
            print(f"    Output: latent {latent.shape}")

            reconstructed = vae.decode(latent, history_motion, nfuture=future_length, scale_latent=1)
            print(f"  ✓ Decode test passed")
            print(f"    Output: reconstructed {reconstructed.shape}")

            # 检查是否有NaN/Inf
            has_nan = torch.isnan(reconstructed).any()
            has_inf = torch.isinf(reconstructed).any()
            if has_nan:
                print("    ⚠ WARNING: NaN detected in reconstruction!")
            if has_inf:
                print("    ⚠ WARNING: Inf detected in reconstruction!")

        return True, latent_dim

    except Exception as e:
        print(f"  ✗ Error loading VAE: {e}")
        traceback.print_exc()
        return False, None


def test_denoiser_loading():
    """测试Denoiser模型加载"""
    print("\n" + "=" * 70)
    print("4. Testing Denoiser Model Loading")
    print("=" * 70)

    from model.mld_denoiser import DenoiserMLP, DenoiserTransformer

    checkpoint_path = './mld_denoiser/mld_finedance_h2_f8_r8/checkpoint_300000.pt'
    if not Path(checkpoint_path).exists():
        print("  ✗ Denoiser checkpoint not found, skipping test")
        return False, None, None

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        print(f"  ✓ Checkpoint loaded")

        # 检测模型类型和latent_dim
        model_type = None
        noise_dim = None

        if 'seqTransEncoder.layers.0.self_attn.in_proj_weight' in state_dict:
            model_type = 'Transformer'
            print("  ✓ Detected model type: Transformer")
            if 'output_process.weight' in state_dict:
                noise_dim = state_dict['output_process.weight'].shape[1]
                print(f"  ✓ Detected noise_dim from output_process: {noise_dim}")
        elif 'mlp.layers.0.layers.0.weight' in state_dict:
            model_type = 'MLP'
            print("  ✓ Detected model type: MLP")
            if 'mlp.out_fc.weight' in state_dict:
                noise_dim = state_dict['mlp.out_fc.weight'].shape[1]
                print(f"  ✓ Detected noise_dim from out_fc: {noise_dim}")

        if noise_dim is None:
            print("  ✗ Could not detect noise_dim from state_dict")
            return False, None, None

        # 创建模型
        latent_dim = noise_dim  # noise_dim对应latent_dim

        if model_type == 'MLP':
            denoiser = DenoiserMLP(
                h_dim=512,
                n_blocks=2,
                dropout=0.1,
                activation='gelu',
                music_dim=35,
                history_shape=(2, 276),
                noise_shape=(1, latent_dim),
                cond_mask_prob=0.0,
            ).to(device)
        else:  # Transformer
            denoiser = DenoiserTransformer(
                h_dim=512,
                ff_size=1024,
                num_layers=8,
                num_heads=4,
                dropout=0.1,
                activation='gelu',
                music_dim=35,
                history_shape=(2, 276),
                noise_shape=(1, latent_dim),
                cond_mask_prob=0.0,
            ).to(device)

        denoiser.load_state_dict(state_dict)
        denoiser.eval()

        print(f"  ✓ Denoiser model created and loaded")

        # 测试前向传播
        batch_size = 2

        x_t = torch.randn(batch_size, 1, latent_dim).to(device)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

        y = {
            'music_embedding': torch.randn(batch_size, 35).to(device),
            'history_motion_normalized': torch.randn(batch_size, 2, 276).to(device),
        }

        with torch.no_grad():
            output = denoiser(x_t, timesteps, y)
            print(f"\n  ✓ Forward pass test passed")
            print(f"    Input: x_t {x_t.shape}, timesteps {timesteps.shape}")
            print(f"    Output: {output.shape}")

            # 检查是否有NaN/Inf
            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()
            if has_nan:
                print("    ⚠ WARNING: NaN detected in denoiser output!")
            if has_inf:
                print("    ⚠ WARNING: Inf detected in denoiser output!")

        return True, model_type, latent_dim

    except Exception as e:
        print(f"  ✗ Error loading Denoiser: {e}")
        traceback.print_exc()
        return False, None, None


def test_6d_rotation():
    """测试6D旋转表示的归一化"""
    print("\n" + "=" * 70)
    print("5. Testing 6D Rotation Normalization")
    print("=" * 70)

    from pytorch3d import transforms

    # 创建测试用的6D旋转（模拟可能的问题）
    # 问题：随机6D值可能不构成有效的旋转
    print("  Testing with potentially invalid 6D rotations:")

    # Case 1: 随机值（可能无效）
    invalid_6d = torch.randn(10, 6)
    print(f"  Random 6D values: {invalid_6d[0]}")

    # 尝试转换为旋转矩阵
    try:
        mat_invalid = transforms.rotation_6d_to_matrix(invalid_6d)
        print(f"  ⚠ Random 6D converted to matrix (may be invalid): determinant range {torch.linalg.det(mat_invalid).min():.4f}, {torch.linalg.det(mat_invalid).max():.4f}")
    except Exception as e:
        print(f"  ✗ Error converting random 6D to matrix: {e}")

    # Case 2: 归一化后的值（更可靠）
    print("\n  Testing with normalized 6D rotations:")

    # 先对随机6D进行归一化
    a1 = invalid_6d[:, :3]
    a2 = invalid_6d[:, 3:6]

    norm_a1 = torch.norm(a1, dim=-1, keepdim=True)
    b1 = a1 / (norm_a1 + 1e-8)

    dot = torch.sum(a2 * b1, dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    norm_b2 = torch.norm(b2, dim=-1, keepdim=True)
    b2 = b2 / (norm_b2 + 1e-8)

    normalized_6d = torch.cat([b1, b2], dim=-1)
    print(f"  Normalized 6D values: {normalized_6d[0]}")

    mat_normalized = transforms.rotation_6d_to_matrix(normalized_6d)
    det_normalized = torch.linalg.det(mat_normalized)
    print(f"  ✓ Normalized 6D converted to valid matrix")
    print(f"    Determinant range: {det_normalized.min():.4f}, {det_normalized.max():.4f}")

    # 行列式接近1表示是有效的旋转矩阵
    if (det_normalized > 0.9).all():
        print("  ✓ All determinants are valid (close to 1)")
    else:
        print("  ⚠ Some determinants are not close to 1")

    return True


def test_smpl_parameters():
    """测试SMPL参数的有效性"""
    print("\n" + "=" * 70)
    print("6. Testing SMPL Parameter Validity")
    print("=" * 70)

    from pytorch3d import transforms

    # 创建有效的6D旋转
    valid_6d = torch.tensor([
        [1, 0, 0, 0, 0, 0],  # Identity rotation
    [0.707, 0.707, 0, 0],  # 45 degree around X
    [0.5, 0.5, 0.707, 0, 0],  # 45 degree around Z
    [0, 0, 1, 0, 0, 0],  # 90 degree around Z
    [0.5, 0.866, 0, 0, 0],  # 60 degree around X
    [0, 0.940, -0.342, 0, 0],  # 20 degree around Y
    [0, 0.866, -0.5, 0, 0],  # 30 degree around Y
        ], dtype=torch.float32)

    print(f"  Testing with valid 6D rotations: {valid_6d.shape}")

    # 转换为旋转矩阵
    rotation_mats = transforms.rotation_6d_to_matrix(valid_6d)
    print(f"  ✓ Converted to rotation matrices: {rotation_mats.shape}")

    # 检查行列式
    dets = torch.linalg.det(rotation_mats)
    print(f"  Determinants: {dets}")

    # 检查正交性（R^T = I）
    transposed = rotation_mats.transpose(-1, -2)
    ortho_check = torch.bmm(rotation_mats, transposed)
    diag = torch.diagonal(ortho_check, dim1=-2, dim2=-1)
    print(f"  Diagonal of R @ R^T (should be ~1): {diag}")

    # 检查每个关节
    for i in range(len(valid_6d)):
        det = dets[i].item()
        diag_i = diag[i]
        is_valid = abs(det - 1.0) < 0.1 and abs(diag_i - 1.0) < 0.1
        status = "✓" if is_valid else "✗"
        print(f"  Joint {i}: determinant={det:.4f}, diag={diag_i:.4f} {status}")

    return True


def test_complete_pipeline():
    """测试完整的生成流程（简化版，使用模拟数据）"""
    print("\n" + "=" * 70)
    print("7. Testing Complete Pipeline (with mock data)")
    print("=" * 70)

    from model.mld_vae import AutoMldVae
    from model.mld_denoiser import DenoiserMLP
    from diffusion.respace import SpacedDiffusion, space_timesteps
    from diffusion import gaussian_diffusion as gd
    from pytorch3d import transforms

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # 创建模拟的VAE和Denoiser（使用小维度快速测试）
    print("\n  Creating mock models for testing...")

    vae = AutoMldVae(
        nfeats=276,
        latent_dim=[1, 128],  # 使用较小维度
        h_dim=256,
        num_layers=3,  # 使用较少层
        ff_size=512,
        num_heads=2,
        arch='all_encoder',
    ).to(device)

    denoiser = DenoiserMLP(
        h_dim=256,
        n_blocks=2,
        dropout=0.0,
        activation='gelu',
        music_dim=35,
        history_shape=(2, 276),
        noise_shape=(1, 128),
        cond_mask_prob=0.0,
    ).to(device)

    vae.eval()
    denoiser.eval()

    print("  ✓ Mock models created")

    # 创建diffusion
    steps = 10  # 使用较少步数
    noise_schedule = 'cosine'
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, [steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
    )

    print("  ✓ Diffusion created")

    # 生成模拟音乐特征
    num_primitives = 4
    future_length = 8
    history_length = 2
    music_features = torch.randn(num_primitives * future_length, 35).to(device)

    print(f"\n  Mock music features: {music_features.shape}")

    # 生成motion
    print("\n  Starting generation test...")
    history_motion = torch.zeros(history_length, 276).to(device)

    all_motion = []

    for i in range(num_primitives):
        print(f"  Generating primitive {i+1}/{num_primitives}...")

        music_emb = music_features[i*future_length:(i+1)*future_length].mean(dim=0, keepdim=True)

        y = {
            'music_embedding': music_emb.unsqueeze(0),
            'history_motion_normalized': history_motion.unsqueeze(0),
        }

        model_kwargs = {'y': y}

        # Diffusion采样
        shape = (1, 1, 128)

        try:
            latent = diffusion.p_sample_loop(
                denoiser,
                shape,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                progress=False,
            )

            latent = latent.permute(1, 0, 2)

            # VAE解码
            future_motion = vae.decode(
                latent,
                history_motion.unsqueeze(0),
                nfuture=future_length,
                scale_latent=1
            )

            future_motion = future_motion.squeeze(0)

            # 检查NaN/Inf
            has_nan = torch.isnan(future_motion).any()
            has_inf = torch.isinf(future_motion).any()

            if has_nan or has_inf:
                print(f"    ⚠ WARNING: NaN/Inf detected in primitive {i+1}")
                future_motion = torch.zeros_like(future_motion)

            all_motion.append(future_motion)

            # 更新历史
            history_motion = future_motion[-history_length:]

            print(f"    ✓ Primitive {i+1} completed, output shape: {future_motion.shape}")

        except Exception as e:
            print(f"    ✗ Error in primitive {i+1}: {e}")
            traceback.print_exc()
            break

    if len(all_motion) == num_primitives:
        motion = torch.cat(all_motion, dim=0)
        print(f"\n  ✓ All primitives generated, final motion shape: {motion.shape}")

        # 测试6D旋转
        T = motion.shape[0]
        poses_6d = motion[:, 3:135]  # [T, 132]

        # 重塑为6D形式
        poses_6d_reshaped = poses_6d.reshape(T, 22, 6)

        # 检查旋转的有效性
        global_orient_6d = poses_6d_reshaped[:, 0, :]
        body_pose_6d = poses_6d_reshaped[:, 1:, :]

        global_orient_mat = transforms.rotation_6d_to_matrix(global_orient_6d.reshape(-1, 6))
        body_pose_mat = transforms.rotation_6d_to_matrix(body_pose_6d.reshape(-1, 6))

        dets_global = torch.linalg.det(global_orient_mat)
        dets_body = torch.linalg.det(body_pose_mat.reshape(-1, 3, 3))

        print(f"\n  Rotation matrix statistics:")
        print(f"    Global orient determinants: min={dets_global.min():.4f}, max={dets_global.max():.4f}, mean={dets_global.mean():.4f}")
        print(f"    Body pose determinants: min={dets_body.min():.4f}, max={dets_body.max():.4f}, mean={dets_body.mean():.4f}")

        # 检查有多少无效的旋转
        invalid_global = (torch.abs(dets_global - 1.0) > 0.5).sum().item()
        invalid_body = (torch.abs(dets_body - 1.0) > 0.5).sum().item()

        print(f"\n  Invalid rotations:")
        print(f"    Global orient: {invalid_global}/{len(dets_global)} ({invalid_global/len(dets_global)*100:.1f}%)")
        print(f"    Body pose: {invalid_body}/{len(dets_body)} ({invalid_body/len(dets_body)*100:.1f}%)")

        if invalid_global == 0 and invalid_body == 0:
            print("  ✓ All rotations are valid!")
        else:
            print("  ✗ Some rotations are INVALID - this causes human distortion!")

        return True
    else:
        print("  ✗ Generation failed")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test and debug Music2Dance pipeline')
    parser.add_argument('--full_test', action='store_true',
                      help='Run all tests including full pipeline')
    parser.add_argument('--test_6d', action='store_true',
                      help='Test 6D rotation normalization only')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Music2Dance - Test and Debug Script")
    print("=" * 70)
    print()

    # 检查checkpoint
    ckpt_results, any_ckpt_exists = test_checkpoint_existence()

    # 检查数据目录
    data_results, any_data_exists = test_data_directory()

    # 只有在有checkpoint和数据时才继续
    if not any_ckpt_exists or not any_data_exists:
        print("\n" + "=" * 70)
        print("CRITICAL: Cannot proceed with tests - missing checkpoints or data")
        print("=" * 70)
        print("\nPlease ensure:")
        print("1. Checkpoint files exist in:")
        print("   - ./mvae/mvae_finedance_h2_f8_r8/checkpoint_20000.pt")
        print("   - ./mld_denoiser/mld_finedance_h2_f8_r8/checkpoint_300000.pt")
        print("2. Data directory exists at:")
        print("   - ./data/finedance/")
        print("\nIf working on remote server s2.njucite.cn:")
        print("- Ensure conda environment 'DART' is activated")
        print("- Check that DART_clean folder has the correct structure")
        print("- Copy/sync data and checkpoints from remote to local for development")
        return

    # 测试VAE加载
    vae_ok, vae_latent_dim = test_vae_loading()

    # 测试Denoiser加载
    denoiser_ok, denoiser_type, denoiser_latent_dim = test_denoiser_loading()

    # 检查latent_dim匹配
    if vae_ok and denoiser_ok:
        if vae_latent_dim == denoiser_latent_dim:
            print(f"\n  ✓ Latent dimensions match: {vae_latent_dim}")
        else:
            print(f"\n  ✗ Latent dimensions DO NOT match!")
            print(f"    VAE latent_dim: {vae_latent_dim}")
            print(f"    Denoiser noise_dim: {denoiser_latent_dim}")
            print("    This WILL cause issues!")

    # 测试6D旋转
    test_6d_rotation()

    # 测试SMPL参数
    test_smpl_parameters()

    # 完整流程测试（如果请求）
    if args.full_test:
        test_complete_pipeline()

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    summary = []
    if any_ckpt_exists:
        summary.append("✓ Checkpoints available (at least some)")
    else:
        summary.append("✗ No checkpoints available")

    if any_data_exists:
        summary.append("✓ Data directories exist (at least some)")
    else:
        summary.append("✗ No data directories exist")

    if vae_ok:
        summary.append(f"✓ VAE can be loaded (latent_dim={vae_latent_dim})")
    else:
        summary.append("✗ VAE cannot be loaded")

    if denoiser_ok:
        summary.append(f"✓ Denoiser can be loaded (type={denoiser_type}, latent_dim={denoiser_latent_dim})")
    else:
        summary.append("✗ Denoiser cannot be loaded")

    for item in summary:
        print(f"  {item}")

    print("\n" + "=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print("""
1. If no checkpoints exist locally:
   - Copy checkpoints from remote server s2.njucite.cn:/root/DART_clean/
   - Or train models using the provided scripts

2. If no data exists locally:
   - Copy/sync data from remote server
   - Or use remote server directly for training/inference

3. For testing on remote server:
   - SSH to s2.njucite.cn
   - Navigate to DART_clean folder
   - Activate conda: conda activate DART
   - Run this script: python test_and_debug.py

4. After fixing issues:
   - Run: python generate_music_dance_fixed.py --vae_path <VAE> --denoiser_path <DENOISER> --music_path <MUSIC>
   - Visualize: python visualize/vis_seq.py --seq_path <OUTPUT_PKL>
   - Check for distortion in the visualization
    """)
    print("=" * 70)


if __name__ == '__main__':
    main()

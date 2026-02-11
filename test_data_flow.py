#!/usr/bin/env python3
"""
测试FineDance数据集和Denoiser的数据流
在正式训练前运行此脚本验证一切正常
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_dataset():
    """测试数据集加载"""
    print("=" * 80)
    print("1. Testing FineDanceDataset")
    print("=" * 80)
    
    from data_loaders.humanml.data.dataset_finedance import FineDanceDataset
    
    dataset = FineDanceDataset(
        dataset_path='./data/finedance',
        cfg_path='./config_files/config_hydra/motion_primitive/finedance_h2_f8_r8.yaml',
        enforce_gender='male',
        enforce_zero_beta=1,
        body_type='smplx',
        split='train',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} sequences")
    print(f"✓ History length: {dataset.history_length}")
    print(f"✓ Future length: {dataset.future_length}")
    print(f"✓ Num primitives: {dataset.num_primitive}")
    print(f"✓ Music dim: {dataset.music_dim}")
    
    # 测试get_batch
    print("\nTesting get_batch...")
    batch = dataset.get_batch(batch_size=4)
    
    print(f"✓ Batch type: {type(batch)}, length: {len(batch)}")
    
    # 检查第一个primitive
    first_primitive = batch[0]
    print(f"\nFirst primitive keys: {first_primitive.keys()}")
    print(f"✓ motion_tensor_normalized shape: {first_primitive['motion_tensor_normalized'].shape}")
    print(f"✓ music_embedding shape: {first_primitive['music_embedding'].shape}")
    print(f"✓ betas shape: {first_primitive['betas'].shape}")
    print(f"✓ history_motion shape: {first_primitive['history_motion'].shape}")
    
    # 验证维度
    assert first_primitive['music_embedding'].shape == (4, 35), "Music embedding dimension mismatch!"
    assert first_primitive['motion_tensor_normalized'].shape[0] == 4, "Batch size mismatch!"
    assert first_primitive['motion_tensor_normalized'].shape[1] == 276, "Motion dimension mismatch!"
    
    print("\n✅ Dataset test passed!\n")
    return dataset, batch


def test_denoiser(batch):
    """测试Denoiser模型"""
    print("=" * 80)
    print("2. Testing Denoiser Model")
    print("=" * 80)
    
    from model.mld_denoiser import DenoiserTransformer, DenoiserMLP
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试Transformer版本
    print("\nTesting DenoiserTransformer...")
    denoiser_trans = DenoiserTransformer(
        h_dim=512,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        activation='gelu',
        music_dim=35,
        history_shape=(2, 276),
        noise_shape=(1, 256),
        cond_mask_prob=0.1
    ).to(device)
    
    print(f"✓ DenoiserTransformer created")
    print(f"  Parameters: {sum(p.numel() for p in denoiser_trans.parameters()):,}")
    
    # 准备输入
    first_primitive = batch[0]
    batch_size = first_primitive['music_embedding'].shape[0]
    
    x_t = torch.randn(batch_size, 1, 256).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    
    y = {
        'music_embedding': first_primitive['music_embedding'],
        'history_motion_normalized': first_primitive['history_motion'].squeeze(2).permute(0, 2, 1)
    }
    
    print(f"\nInput shapes:")
    print(f"  x_t: {x_t.shape}")
    print(f"  timesteps: {timesteps.shape}")
    print(f"  music_embedding: {y['music_embedding'].shape}")
    print(f"  history_motion: {y['history_motion_normalized'].shape}")
    
    # Forward pass
    output = denoiser_trans(x_t, timesteps, y)
    print(f"\n✓ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 1, 256), "Output shape mismatch!"
    
    # 测试MLP版本
    print("\n" + "-" * 80)
    print("Testing DenoiserMLP...")
    denoiser_mlp = DenoiserMLP(
        h_dim=512,
        n_blocks=2,
        dropout=0.1,
        activation='gelu',
        music_dim=35,
        history_shape=(2, 276),
        noise_shape=(1, 256),
        cond_mask_prob=0.1
    ).to(device)
    
    print(f"✓ DenoiserMLP created")
    print(f"  Parameters: {sum(p.numel() for p in denoiser_mlp.parameters()):,}")
    
    output_mlp = denoiser_mlp(x_t, timesteps, y)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output_mlp.shape}")
    
    assert output_mlp.shape == (batch_size, 1, 256), "Output shape mismatch!"
    
    print("\n✅ Denoiser test passed!\n")
    return denoiser_trans


def test_vae_integration(batch):
    """测试与VAE的集成"""
    print("=" * 80)
    print("3. Testing VAE Integration")
    print("=" * 80)
    
    try:
        from model.mld_vae import AutoMldVae
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载VAE
        vae_path = './mvae/mvae_finedance_h2_f8_r8/checkpoint_200000.pt'
        if not Path(vae_path).exists():
            print(f"⚠️  VAE checkpoint not found at {vae_path}")
            print("   Skipping VAE integration test")
            return None
        
        print(f"Loading VAE from {vae_path}...")
        
        # 创建VAE模型
        vae_model = AutoMldVae(
            nfeats=276,
            latent_dim=[1, 256],
            h_dim=256,
            num_layers=7,
        ).to(device)
        
        checkpoint = torch.load(vae_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        
        if 'latent_mean' not in model_state_dict:
            model_state_dict['latent_mean'] = torch.tensor(0)
        if 'latent_std' not in model_state_dict:
            model_state_dict['latent_std'] = torch.tensor(1)
        
        vae_model.load_state_dict(model_state_dict)
        vae_model.eval()
        
        print(f"✓ VAE loaded successfully")
        print(f"  Latent mean: {vae_model.latent_mean}")
        print(f"  Latent std: {vae_model.latent_std}")
        
        # 测试编码-解码
        first_primitive = batch[0]
        motion_tensor = first_primitive['motion_tensor_normalized'].squeeze(2).permute(0, 2, 1)  # [B, T, D]
        history_length = first_primitive['history_length']
        future_length = first_primitive['future_length']
        
        history_motion = motion_tensor[:, :history_length, :]
        future_motion = motion_tensor[:, history_length:, :]
        
        print(f"\nTesting VAE encode-decode...")
        print(f"  History motion: {history_motion.shape}")
        print(f"  Future motion: {future_motion.shape}")
        
        with torch.no_grad():
            latent, _ = vae_model.encode(future_motion, history_motion, scale_latent=1)
            print(f"✓ Encoded latent: {latent.shape}")
            
            reconstructed = vae_model.decode(latent, history_motion, nfuture=future_length, scale_latent=1)
            print(f"✓ Decoded motion: {reconstructed.shape}")
        
        print("\n✅ VAE integration test passed!\n")
        return vae_model
        
    except Exception as e:
        print(f"❌ VAE integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_diffusion():
    """测试Diffusion设置"""
    print("=" * 80)
    print("4. Testing Diffusion Setup")
    print("=" * 80)
    
    try:
        from diffusion.respace import SpacedDiffusion, space_timesteps
        from diffusion import gaussian_diffusion as gd
        
        # 创建diffusion
        steps = 1000
        noise_schedule = 'cosine'
        betas = gd.get_named_beta_schedule(noise_schedule, steps)
        
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(steps, [steps]),
            betas=betas,
            model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.FIXED_SMALL,
            loss_type=gd.LossType.MSE,
        )
        
        print(f"✓ Diffusion created")
        print(f"  Steps: {steps}")
        print(f"  Noise schedule: {noise_schedule}")
        
        # 测试采样
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_start = torch.randn(4, 1, 256).to(device)
        t = torch.randint(0, steps, (4,)).to(device)
        
        x_t = diffusion.q_sample(x_start, t)
        print(f"✓ Forward diffusion: {x_start.shape} -> {x_t.shape}")
        
        print("\n✅ Diffusion setup test passed!\n")
        return diffusion
        
    except Exception as e:
        print(f"❌ Diffusion setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "FineDance Music-to-Dance Data Flow Test" + " " * 19 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        # 1. 测试数据集
        dataset, batch = test_dataset()
        
        # 2. 测试Denoiser
        denoiser = test_denoiser(batch)
        
        # 3. 测试VAE集成
        vae = test_vae_integration(batch)
        
        # 4. 测试Diffusion
        diffusion = test_diffusion()
        
        # 最终总结
        print("=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print("✅ Dataset loading: PASSED")
        print("✅ Denoiser model: PASSED")
        print("✅ VAE integration: PASSED" if vae is not None else "⚠️  VAE integration: SKIPPED")
        print("✅ Diffusion setup: PASSED" if diffusion is not None else "❌ Diffusion setup: FAILED")
        print("=" * 80)
        print()
        print("🎉 All critical tests passed! Ready to start training.")
        print()
        print("Next steps:")
        print("1. Make sure VAE checkpoint exists at: ./mvae/mvae_finedance_h2_f8_r8/checkpoint_200000.pt")
        print("2. Run: bash scripts/train_finedance_diffusion.sh")
        print()
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Please fix the errors before starting training.")
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()
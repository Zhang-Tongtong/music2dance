import os
import argparse
import numpy as np
import torch

from data_loaders.humanml.data.dataset_finedance import FineDanceDataset
from model.mld_vae import AutoMldVae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_ckpt", type=str, required=True)
    ap.add_argument("--motion_npy", type=str, required=True)
    ap.add_argument("--save_npy", type=str, default="results/vae_recon/recon.npy")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--history_length", type=int, default=2)
    ap.add_argument("--future_length", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_npy), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) load dataset for mean/std (与你训练保持一致)
    ds = FineDanceDataset(split="train",dataset_path='./data/finedance')
    mean = torch.from_numpy(ds.motion_mean).float().to(device)  # (1,276)
    std = torch.from_numpy(ds.motion_std).float().to(device)    # (1,276)

    # 2) load GT motion (T,276)
    motion = np.load(args.motion_npy).astype(np.float32)
    x = torch.from_numpy(motion).to(device)  # (T,276)

    # 3) normalize
    x_norm = (x - mean) / (std + 1e-8)  # (T,276)

    # 4) prepare one primitive window: (H+F) frames
    H = args.history_length
    F = args.future_length
    if x_norm.shape[0] < (H + F):
        raise RuntimeError(f"Motion too short: T={x_norm.shape[0]}, need >= {H+F}")

    # use first H+F frames for reconstruction test
    window = x_norm[: H + F, :]                 # (H+F,276)
    window = window.unsqueeze(0)                # (1,H+F,276)  => [B,T,D]

    history_motion = window[:, :H, :]           # (1,H,276)
    future_motion = window[:, H:H+F, :]         # (1,F,276)

    print("\n[INFO] VAE recon test")
    print("  history_motion:", history_motion.shape)
    print("  future_motion :", future_motion.shape)

    # 5) load VAE ckpt
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  # fallback

    # 你的训练日志显示 h_dim=256, latent_dim=(1,256), num_layers=7, num_heads=4, ff_size=1024...
    # 这里必须和训练一致，否则会 size mismatch
    vae = AutoMldVae(
        nfeats=276,
        latent_dim=(1, 256),
        h_dim=256,
        ff_size=1024,
        num_layers=7,
        num_heads=4,
        dropout=0.1,
        arch="all_encoder",
        normalize_before=False,
        activation="gelu",
        position_embedding="learned",
    )
    vae.load_state_dict(state, strict=True)
    vae.to(device)
    vae.eval()

    # 6) encode -> decode
    with torch.no_grad():
        latent, dist = vae.encode(future_motion, history_motion, scale_latent=False)
        recon_future = vae.decode(latent, history_motion, nfuture=F, scale_latent=False)  # (1,F,276)

    # 7) 拼回完整片段：history + recon_future
    recon_full = torch.cat([history_motion, recon_future], dim=1)  # (1,H+F,276)

    # 8) denormalize 回 276
    recon_denorm = recon_full.squeeze(0) * std + mean  # (H+F,276)

    recon_np = recon_denorm.detach().cpu().numpy().astype(np.float32)
    np.save(args.save_npy, recon_np)
    print(f"[OK] saved recon motion to {args.save_npy}, shape={recon_np.shape}")

    # 9) quick check: 6D basic stats
    pose6d = recon_denorm[:, 3:135].reshape(-1, 22, 6)
    a1 = pose6d[..., :3]
    a2 = pose6d[..., 3:]
    n1 = a1.norm(dim=-1).mean().item()
    n2 = a2.norm(dim=-1).mean().item()
    dot = (a1 * a2).sum(dim=-1).abs().mean().item()
    print(f"[CHECK] recon 6D: mean(norm1)={n1:.6f}, mean(norm2)={n2:.6f}, mean(|dot|)={dot:.6e}")

    print("\n[NEXT] Render results/vae_recon/recon.npy first frame and compare with GT.\n")


if __name__ == "__main__":
    main()

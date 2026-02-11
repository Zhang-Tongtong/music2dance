import os
import sys
import numpy as np
import torch
import smplx
from tqdm import tqdm
import pickle

# 将项目根目录添加到 sys.path，确保能 import mld
sys.path.append(os.getcwd())

# 引入 DART/MLD 的特征提取函数
from mld.data.humanml.scripts.motion_process import extract_features
# 引入 骨架定义
from mld.data.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

# ================= 配置区域 ================= #
# 1. 原始 FineDance 动作数据路径 (包含 .npy 文件)
FINEDANCE_MOTION_DIR = "./data/finedance/motion" 

# 2. 处理结果保存路径
SAVE_DIR = "./data/finedance/processed_finedance"

# 3. 目标帧率
TARGET_FPS = 30 

# 4. SMPL-X 模型路径
SMPLX_MODEL_DIR = "./data/smplx_lockedhead_20230207/models_lockedhead/smplx"

# 5. HumanML3D 标准参数 (paramUtil.py 中缺失的部分在此补全)
# 用于计算身体朝向: Right Hip, Left Hip, Right Shoulder, Left Shoulder
FACE_JOINT_INDX = [2, 1, 17, 16] 
# 脚部接触检测索引: Right Foot, Left Foot
FID_R = [8, 11]
FID_L = [7, 10]
# =========================================== #

def get_body_model(model_path, gender='neutral', device='cuda'):
    """加载 SMPL-X 模型"""
    model_fn = os.path.join(model_path, f"SMPLX_{gender.upper()}.npz")
    return smplx.create(
        model_path=model_fn, 
        model_type='smplx',
        gender=gender, 
        use_pca=False,
        batch_size=1
    ).to(device)

def resample_motion(motion, src_fps, tgt_fps):
    """
    线性插值重采样
    motion: (T, D)
    """
    T, D = motion.shape
    if abs(src_fps - tgt_fps) < 0.1:
        return motion
    
    duration = T / src_fps
    new_T = int(duration * tgt_fps)
    
    times_src = np.linspace(0, duration, T)
    times_tgt = np.linspace(0, duration, new_T)
    
    new_motion = np.zeros((new_T, D))
    for i in range(D):
        new_motion[:, i] = np.interp(times_tgt, times_src, motion[:, i])
    
    return new_motion

def process_single_file(file_path, body_model, device):
    """针对 (T, 315) 数据的严格对齐版本"""
    try:
        # 1. 加载数据 (T, 315)
        data = np.load(file_path, allow_pickle=True)
        
        # 处理可能的封装情况
        if data.shape == () or data.dtype == 'O':
            data = data.item()
            if isinstance(data, dict):
                # 如果是字典格式
                poses_raw = np.array(data['poses'])
                T = poses_raw.shape[0]
                trans_raw = np.array(data.get('trans', data.get('transl', np.zeros((T, 3)))))
            else:
                # 解包后是数组
                poses_raw = data
        else:
            poses_raw = data

        T = poses_raw.shape[0]

        # 2. 核心切片逻辑 (确保所有 Tensor 的第一维都是 T)
        # global_orient: [T, 3], body_pose: [T, 63]
        global_orient = torch.from_numpy(poses_raw[:, :3]).float().to(device)
        body_pose = torch.from_numpy(poses_raw[:, 3:66]).float().to(device)
        
        # transl: 位移，通常在 309:312 维
        if poses_raw.shape[1] >= 312:
            trans = torch.from_numpy(poses_raw[:, 309:312]).float().to(device)
        else:
            trans = torch.zeros(T, 3).float().to(device)

        # betas: SMPL-X 需要 10 维，确保它是 (T, 10) 而不是 (1, 10)
        # 报错的原因通常是这里只给了 1 帧，需要 repeat 或直接初始化为 T 帧
        betas = torch.zeros(T, 10).float().to(device) 
        
        src_fps = 30 # FineDance 默认 FPS
        
    except Exception as e:
        raise RuntimeError(f"切片对齐失败: {e}")

    # --- SMPL-X Forward ---
    with torch.no_grad():
        # 此时所有输入的第一维均为 T，满足 batch 要求
        output = body_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=trans
        )
        # 提取 HumanML3D 需要的 22 个主要关节
        # output.joints 维度通常是 [T, 118, 3]
        joints = output.joints[:, :22, :].cpu().numpy()

    # --- 后续特征提取 (保持不变) ---
    joints_flat = joints.reshape(T, -1)
    joints_resampled = resample_motion(joints_flat, src_fps, TARGET_FPS)
    joints_resampled = joints_resampled.reshape(-1, 22, 3)
    
    feature_data = extract_features(
        joints_resampled, 0.002, t2m_raw_offsets, t2m_kinematic_chain, 
        FACE_JOINT_INDX, FID_R, FID_L
    )
    
    return feature_data

def main():
    # 检查路径
    if not os.path.exists(FINEDANCE_MOTION_DIR):
        print(f"Error: FineDance motion dir not found at {FINEDANCE_MOTION_DIR}")
        return
    
    output_vec_dir = os.path.join(SAVE_DIR, "new_joint_vecs")
    os.makedirs(output_vec_dir, exist_ok=True)
    
    test_data = np.load('./data/finedance/motion/138.npy', allow_pickle=True)
    print(type(test_data))
    print(test_data.shape)
    print(test_data.dtype)
    
    # 初始化设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading SMPL-X model from {SMPLX_MODEL_DIR}...")
    # 默认加载 neutral 模型，如果 FineDance 有 gender 信息，可以在循环中动态加载，
    # 但通常 neutral 对特征提取影响不大（主要影响 Mesh 形状，不太影响骨骼相对运动）
    try:
        body_model = get_body_model(SMPLX_MODEL_DIR, gender='neutral', device=device)
    except Exception as e:
        print(f"Failed to load SMPL-X model: {e}")
        print(f"Please check if {SMPLX_MODEL_DIR} contains SMPLX_NEUTRAL.npz")
        return

    # 获取文件列表
    file_list = [f for f in os.listdir(FINEDANCE_MOTION_DIR) if f.endswith('.npy')]
    valid_files = []
    
    print(f"Found {len(file_list)} motion files. Start processing...")
    
    for filename in tqdm(file_list):
        file_path = os.path.join(FINEDANCE_MOTION_DIR, filename)
        save_name = filename.replace('.npy', '')
        save_path = os.path.join(output_vec_dir, save_name + '.npy')
        

        # if os.path.exists(save_path):
        #     valid_files.append(save_name)
        #     continue

        try:
            features = process_single_file(file_path, body_model, device)
            
            # 保存特征 (T, 263)
            np.save(save_path, features)
            valid_files.append(save_name)
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue

    # ================= 生成数据集列表和统计数据 ================= #
    print(f"\nSuccessfully processed {len(valid_files)} files.")
    print("Generating train/val split and calculating Mean/Std...")
    
    # 1. 划分数据集 (90% Train, 10% Val)
    # 也可以读取 FineDance 官方的 split.txt 如果有的话
    num_train = int(len(valid_files) * 0.9)
    train_files = valid_files[:num_train]
    val_files = valid_files[num_train:]
    
    with open(os.path.join(SAVE_DIR, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_files))
    with open(os.path.join(SAVE_DIR, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_files))
        
    # 2. 计算 Mean 和 Std
    all_features = []
    # 随机采样最多 2000 个样本计算均值，避免内存爆炸
    sample_size = min(len(train_files), 2000)
    sample_indices = np.random.choice(len(train_files), sample_size, replace=False)
    
    for idx in tqdm(sample_indices, desc="Computing Mean/Std"):
        name = train_files[idx]
        feat = np.load(os.path.join(output_vec_dir, name + '.npy'))
        all_features.append(feat)
        
    all_features = np.concatenate(all_features, axis=0)
    
    # 计算
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    
    # 极小值保护，防止除以 0
    std[std < 1e-4] = 1e-4
    
    # 保存
    np.save(os.path.join(SAVE_DIR, "Mean.npy"), mean)
    np.save(os.path.join(SAVE_DIR, "Std.npy"), std)
    
    print(f"Preprocessing Complete! Data saved to {SAVE_DIR}")
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")

if __name__ == "__main__":
    main()
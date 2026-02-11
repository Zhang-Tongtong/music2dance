"""
将FineDance的315维数据转换为DART的276维

FineDance 315维:
- [0:3]: 根节点位置 (x, y, z)
- [3:315]: 52个关节 × 6维旋转 = 312维

DART 276维:
- [0:3]: 平移 t (3维)
- [3:9]: 根旋转 R (6D, 6维)
- [9:135]: 21个身体关节旋转 θ (21×6=126维)
- [135:201]: 22个关节位置 J (22×3=66维)
- [201:204]: 平移差分 dt (3维)
- [204:210]: 旋转差分 dR (6D, 6维)
- [210:276]: 关节位置差分 dJ (22×3=66维)

关键映射:
- FineDance的52个关节需要映射到SMPLX的22个身体关节
- 需要使用SMPLX模型从旋转参数计算关节位置
"""
#新
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import sys
sys.path.append('.')
from utils.smpl_utils import PrimitiveUtility
from pytorch3d import transforms

# FineDance到SMPLX的关节映射
# FineDance使用52个关节，SMPLX使用22个主要关节
# 这个映射需要根据FineDance的关节定义来确定
# 以下是一个推测的映射，可能需要调整

FINEDANCE_TO_SMPLX_JOINT_MAPPING = {
    # SMPLX关节索引: FineDance关节索引
    # 假设FineDance前22个关节对应SMPLX的主要骨骼
    0: 0,   # pelvis
    1: 1,   # left_hip
    2: 2,   # right_hip
    3: 3,   # spine1
    4: 4,   # left_knee
    5: 5,   # right_knee
    6: 6,   # spine2
    7: 7,   # left_ankle
    8: 8,   # right_ankle
    9: 9,   # spine3
    10: 10, # left_foot
    11: 11, # right_foot
    12: 12, # neck
    13: 13, # left_collar
    14: 14, # right_collar
    15: 15, # head
    16: 16, # left_shoulder
    17: 17, # right_shoulder
    18: 18, # left_elbow
    19: 19, # right_elbow
    20: 20, # left_wrist
    21: 21, # right_wrist
}

def convert_finedance_to_dart(input_dir, output_dir, use_smplx_joints=True):
    """
    转换FineDance数据到DART格式
    
    Args:
        input_dir: FineDance motion目录
        output_dir: 输出目录
        use_smplx_joints: 是否使用SMPLX模型计算关节位置(推荐True)
    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if use_smplx_joints:
        primitive_utility = PrimitiveUtility(device=device)
        print("Will use SMPLX model to compute joint positions")
    
    motion_files = sorted(input_dir.glob('*.npy'))
    print(f"Found {len(motion_files)} motion files")
    
    if len(motion_files) == 0:
        print("No .npy files found!")
        return
    
    # 分析第一个文件
    print("\n分析第一个文件...")
    sample_data = np.load(motion_files[0])
    print(f"Sample shape: {sample_data.shape}")
    print(f"Expected: (T, 315)")
    assert sample_data.shape[1] == 315, f"Expected 315 dims, got {sample_data.shape[1]}"
    
    print(f"\n根节点位置 [0:3]:")
    print(f"  Mean: {sample_data[:, 0:3].mean(axis=0)}")
    print(f"  Std:  {sample_data[:, 0:3].std(axis=0)}")
    
    print(f"\n关节旋转 [3:315]:")
    print(f"  Mean: {sample_data[:, 3:315].mean():.6f}")
    print(f"  Std:  {sample_data[:, 3:315].std():.6f}")
    print(f"  Min:  {sample_data[:, 3:315].min():.6f}")
    print(f"  Max:  {sample_data[:, 3:315].max():.6f}")
    
    # 转换所有文件
    successful = 0
    failed = 0
    
    for motion_file in tqdm(motion_files, desc="Converting"):
        try:
            data = np.load(motion_file)  # [T, 315]
            T = data.shape[0]
            
            if data.shape[1] != 315:
                print(f"\nSkipping {motion_file.name}: wrong dimension {data.shape[1]}")
                failed += 1
                continue
            
            # === 解析FineDance数据 ===
            root_trans = data[:, 0:3]  # [T, 3] 根节点位置
            all_joints_rot6d = data[:, 3:315].reshape(T, 52, 6)  # [T, 52, 6]
            
            # === 提取SMPLX需要的22个关节 ===
            # 关节0是root (global_orient)
            # 关节1-21是body_pose
            
            global_orient_6d = all_joints_rot6d[:, 0, :]  # [T, 6] - 第0个关节作为root
            
            # 提取21个body关节
            body_joints_6d = np.zeros((T, 21, 6), dtype=np.float32)
            for smplx_idx in range(1, 22):  # SMPLX的1-21号关节
                finedance_idx = FINEDANCE_TO_SMPLX_JOINT_MAPPING.get(smplx_idx, smplx_idx)
                if finedance_idx < 52:
                    body_joints_6d[:, smplx_idx-1, :] = all_joints_rot6d[:, finedance_idx, :]
            
            body_pose_6d = body_joints_6d.reshape(T, 126)  # [T, 126]
            
            # === 计算关节位置 ===
            if use_smplx_joints:
                # 转换6D到旋转矩阵
                global_orient_6d_t = torch.from_numpy(global_orient_6d).float().to(device)
                body_pose_6d_t = torch.from_numpy(body_pose_6d).float().to(device).reshape(T, 21, 6)
                root_trans_t = torch.from_numpy(root_trans).float().to(device)
                
                global_orient_matrix = transforms.rotation_6d_to_matrix(global_orient_6d_t)  # [T, 3, 3]
                body_pose_matrix = transforms.rotation_6d_to_matrix(body_pose_6d_t)  # [T, 21, 3, 3]
                
                # 使用零shape参数
                betas_t = torch.zeros(T, 10, device=device)
                
                # 调用SMPLX模型
                joints = primitive_utility.smpl_dict_inference({
                    'gender': 'male',
                    'betas': betas_t,
                    'transl': root_trans_t,
                    'global_orient': global_orient_matrix,
                    'body_pose': body_pose_matrix,
                }, return_vertices=False)  # [T, 22, 3]
                
                joints_pos = joints.cpu().numpy().reshape(T, 66)  # [T, 66]
            else:
                # 简单方法：使用根节点位置作为第一个关节，其他设为0（不推荐）
                joints_pos = np.zeros((T, 66), dtype=np.float32)
                joints_pos[:, 0:3] = root_trans  # pelvis位置
                print("Warning: Not using SMPLX model, joint positions will be inaccurate!")
            
            # === 计算差分特征 ===
            # 平移差分
            trans_delta = np.zeros((T, 3), dtype=np.float32)
            trans_delta[1:] = root_trans[1:] - root_trans[:-1]
            
            # 旋转差分 (6D)
            global_orient_6d_t = torch.from_numpy(global_orient_6d).float().to(device)
            global_orient_matrix = transforms.rotation_6d_to_matrix(global_orient_6d_t)
            
            orient_delta_6d = np.zeros((T, 6), dtype=np.float32)
            global_orient_np = global_orient_matrix.cpu().numpy()
            for t in range(1, T):
                # R_delta = R_t * R_{t-1}^T
                delta_matrix = global_orient_np[t] @ global_orient_np[t-1].T
                delta_matrix_t = torch.from_numpy(delta_matrix).float().unsqueeze(0).to(device)
                orient_delta_6d[t] = transforms.matrix_to_rotation_6d(delta_matrix_t).cpu().numpy()[0]
            
            # 关节位置差分
            joints_delta = np.zeros((T, 66), dtype=np.float32)
            joints_delta[1:] = joints_pos[1:] - joints_pos[:-1]
            
            # === 组装DART 276维特征 ===
            dart_motion = np.concatenate([
                root_trans,         # [T, 3]   - 平移 t
                global_orient_6d,   # [T, 6]   - 旋转 R (6D)
                body_pose_6d,       # [T, 126] - 身体姿态 θ (21×6)
                joints_pos,         # [T, 66]  - 关节位置 J (22×3)
                trans_delta,        # [T, 3]   - 平移差分 dt
                orient_delta_6d,    # [T, 6]   - 旋转差分 dR (6D)
                joints_delta,       # [T, 66]  - 关节位置差分 dJ (22×3)
            ], axis=1)  # [T, 276]
            
            assert dart_motion.shape == (T, 276), f"Expected (T, 276), got {dart_motion.shape}"
            
            # 保存
            output_file = output_dir / motion_file.name
            np.save(output_file, dart_motion.astype(np.float32))
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {motion_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"转换完成!")
    print(f"成功: {successful} 个文件")
    print(f"失败: {failed} 个文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}")
    
    # 验证一个输出文件
    if successful > 0:
        output_files = sorted(output_dir.glob('*.npy'))
        sample_output = np.load(output_files[0])
        print(f"\n验证输出文件: {output_files[0].name}")
        print(f"Shape: {sample_output.shape}")
        print(f"Dtype: {sample_output.dtype}")
        print(f"\n各部分统计:")
        parts = [
            ("平移 t", 0, 3),
            ("旋转 R", 3, 9),
            ("身体姿态 θ", 9, 135),
            ("关节位置 J", 135, 201),
            ("平移差分 dt", 201, 204),
            ("旋转差分 dR", 204, 210),
            ("关节位置差分 dJ", 210, 276),
        ]
        for name, start, end in parts:
            seg = sample_output[:, start:end]
            print(f"{name:20s} [{start:3d}:{end:3d}] ({end-start:3d}维): "
                  f"mean={seg.mean():8.4f}, std={seg.std():8.4f}, "
                  f"min={seg.min():8.4f}, max={seg.max():8.4f}")

if __name__ == '__main__':
    convert_finedance_to_dart(
        input_dir='./data/finedance/motion',
        output_dir='./data/finedance/motion_276d',
        use_smplx_joints=True  # 推荐使用SMPLX模型计算关节位置
    )
"""
将FineDance数据转换为训练所需格式
"""
import numpy as np
from pathlib import Path
import shutil

def prepare_finedance_data(
    finedance_root='./data/finedance',
    output_root='./data/finedance_processed'
):
    """
    准备FineDance数据
    
    假设finedance_root结构:
    finedance/
        ├── motion/        # .npy文件, shape=[T, 315]
        ├── music_npynew/  # LODGE特征, shape=[T, 35]
        └── label_json/
    """
    finedance_root = Path(finedance_root)
    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True, parents=True)
    
    # 直接创建软链接或复制
    for subdir in ['motion_276d', 'music_npynew', 'label_json']:
        src = finedance_root / subdir
        dst = output_root / subdir
        
        if src.exists():
            if dst.exists():
                print(f"{dst} already exists, skipping...")
            else:
                # 创建软链接
                dst.symlink_to(src.absolute())
                print(f"Created symlink: {dst} -> {src}")
        else:
            print(f"Warning: {src} does not exist!")
    
    print("FineDance data preparation completed!")
    print(f"Data location: {output_root}")

if __name__ == '__main__':
    prepare_finedance_data()
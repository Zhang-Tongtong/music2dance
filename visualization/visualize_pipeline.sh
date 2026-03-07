#!/bin/bash

# =====================================================
# Music-to-Dance 完整可视化流程
# 参考MotionGPT的可视化pipeline
# =====================================================

set -e  # 遇到错误立即退出

echo "============================================"
echo "Music-to-Dance Visualization Pipeline"
echo "============================================"

# =====================================================
# 配置参数
# =====================================================

# Blender路径 (需要根据你的系统调整)
BLENDER_PATH="/snap/bin/blender"  # Ubuntu默认路径
# BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"  # macOS路径
# BLENDER_PATH="C:/Program Files/Blender Foundation/Blender/blender.exe"  # Windows路径

# Python路径 (Blender内置Python)
BLENDER_PYTHON="$BLENDER_PATH/3.6/python/bin/python3.10"  # 根据Blender版本调整

# 输入输出路径
GENERATED_MOTION="./results/demo/generated_dance.pkl"
OUTPUT_DIR="./results/visualization"
TEMP_DIR="./results/temp_visualization"

# SMPL模型路径
SMPL_MODEL_PATH="./data/smplx_lockedhead_20230207/models_lockedhead"

# =====================================================
# 步骤0: 检查依赖
# =====================================================

echo ""
echo "Step 0: Checking dependencies..."

# 检查Blender
if [ ! -f "$BLENDER_PATH" ]; then
    echo "❌ Blender not found at $BLENDER_PATH"
    echo "Please install Blender from https://www.blender.org/"
    echo "Then update BLENDER_PATH in this script"
    exit 1
fi
echo "✅ Blender found at $BLENDER_PATH"

# 检查生成的motion文件
if [ ! -f "$GENERATED_MOTION" ]; then
    echo "❌ Generated motion not found at $GENERATED_MOTION"
    echo "Please generate motion first using generate_music_dance.py"
    exit 1
fi
echo "✅ Generated motion found"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# =====================================================
# 步骤1: 安装Blender Python依赖
# =====================================================

echo ""
echo "Step 1: Installing Blender Python dependencies..."

# 只安装系统Python需要的依赖
echo "Installing trimesh for mesh processing..."
pip install trimesh pytorch3d -q

echo "✅ Dependencies installed"

# =====================================================
# 步骤2: 转换motion为SMPL mesh
# =====================================================

echo ""
echo "Step 2: Converting motion to SMPL meshes..."

python visualization/fit.py \
    --dir $(dirname "$GENERATED_MOTION") \
    --save_folder "$TEMP_DIR" \
    --smpl_model_path "$SMPL_MODEL_PATH" \
    --cuda \
    --save_npy 1 \
    --save_ply 1

echo "✅ SMPL meshes generated"

# =====================================================
# 步骤3: 使用Blender渲染视频
# =====================================================

echo ""
echo "Step 3: Rendering video with Blender..."

# 提取motion名称
MOTION_NAME=$(basename "$GENERATED_MOTION" .pkl)
MESH_DIR="$TEMP_DIR/$MOTION_NAME"

# 渲染视频（输出路径在render.py内部生成）
$BLENDER_PATH --background --python visualization/render.py -- \
    --dir "$MESH_DIR" \
    --mode video

# 合成视频
ffmpeg -y -r 30 -i results/temp_visualization/generated_dance/render/frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p \
  results/visualization/generated_dance.mp4

echo "✅ Video rendered"

# =====================================================
# 步骤4: (可选) 渲染序列图
# =====================================================

echo ""
echo "Step 4: Rendering sequence image..."

$BLENDER_PATH --background --python visualization/render.py -- \
    --dir "$MESH_DIR" \
    --mode sequence

echo "✅ Sequence image rendered"

# =====================================================
# 步骤5: 清理临时文件 (可选)
# =====================================================

echo ""
read -p "Do you want to clean up temporary files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    echo "✅ Temporary files cleaned"
fi

# =====================================================
# 完成
# =====================================================

echo ""
echo "============================================"
echo "✅ Visualization completed!"
echo "============================================"
echo "Output files:"
echo "  Video: $OUTPUT_DIR/${MOTION_NAME}.mp4"
echo "  Sequence: $OUTPUT_DIR/${MOTION_NAME}_sequence.png"
echo "============================================"
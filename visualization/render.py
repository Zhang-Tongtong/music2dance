"""
Blender渲染脚本 - 简化版
直接为每一帧导入mesh，不使用shape keys
"""

import bpy
import sys
import os
from pathlib import Path
import argparse
from math import radians


def setup_scene():
    """设置Blender场景"""
    # 清除现有对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # 设置渲染引擎
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 64  # 降低采样以加快速度
    
    # 设置分辨率
    bpy.context.scene.render.resolution_x = 512  # 降低分辨率加快渲染
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.resolution_percentage = 100
    
    # 添加相机
    bpy.ops.object.camera_add(location=(0, -4, 1.5))
    camera = bpy.context.object
    camera.rotation_euler = (radians(90), 0, 0)
    bpy.context.scene.camera = camera
    
    # 添加光源
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.object
    sun.data.energy = 2.0
    
    # 补光
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
    area = bpy.context.object
    area.data.energy = 300
    
    # 添加地板
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    floor = bpy.context.object
    
    # 地板材质
    mat = bpy.data.materials.new(name="FloorMaterial")
    if mat.use_nodes:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.8
    floor.data.materials.append(mat)
    
    # 设置背景
    world = bpy.data.worlds['World']
    if world.use_nodes:
        bg = world.node_tree.nodes.get('Background')
        if bg:
            bg.inputs['Color'].default_value = (1, 1, 1, 1)
            bg.inputs['Strength'].default_value = 0.5


def create_material():
    """创建人体材质"""
    mat = bpy.data.materials.new(name="BodyMaterial")
    if mat.use_nodes:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # 肤色
            bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1.0)
            # 粗糙度
            if 'Roughness' in bsdf.inputs:
                bsdf.inputs['Roughness'].default_value = 0.4
            # Subsurface（新版本可能叫Subsurface Weight）
            if 'Subsurface' in bsdf.inputs:
                bsdf.inputs['Subsurface'].default_value = 0.1
            elif 'Subsurface Weight' in bsdf.inputs:
                bsdf.inputs['Subsurface Weight'].default_value = 0.1
    return mat


def render_video_simple(args):
    """
    简化的视频渲染：
    为每一帧单独导入mesh，渲染后删除，然后用ffmpeg合成
    """
    print("Setting up scene...")
    setup_scene()
    
    ply_dir = Path(args.dir) / 'ply'
    if not ply_dir.exists():
        print(f"PLY directory not found: {ply_dir}")
        return
    
    ply_files = sorted(ply_dir.glob('frame_*.ply'))
    if len(ply_files) == 0:
        print(f"No PLY files found in {ply_dir}")
        return
    
    print(f"Found {len(ply_files)} PLY files")
    
    # 创建输出目录
    output_path = Path(args.dir) / 'render'
    output_path.mkdir(exist_ok=True)
    frames_dir = output_path / 'frames'
    frames_dir.mkdir(exist_ok=True)
    
    # 创建材质（复用）
    body_mat = create_material()
    
    # 设置帧率
    bpy.context.scene.render.fps = 30
    
    # 为每一帧渲染
    print("Rendering frames...")
    for i, ply_file in enumerate(ply_files):
        # 进度
        if (i + 1) % 10 == 0:
            print(f"  Rendering frame {i+1}/{len(ply_files)}")
        
        # 导入PLY
        bpy.ops.wm.ply_import(filepath=str(ply_file))
        mesh_obj = bpy.context.selected_objects[0]
        
        # 添加材质
        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(body_mat)
        
        # 平滑着色
        for poly in mesh_obj.data.polygons:
            poly.use_smooth = True
        
        # 设置输出文件名
        frame_path = frames_dir / f"frame_{i:04d}.png"
        bpy.context.scene.render.filepath = str(frame_path)
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        
        # 渲染当前帧
        bpy.ops.render.render(write_still=True)
        
        # 删除mesh为下一帧做准备
        bpy.data.objects.remove(mesh_obj, do_unlink=True)
    
    print(f"✅ Rendered {len(ply_files)} frames to {frames_dir}")
    
    # 使用ffmpeg合成视频（不依赖libx264）
    print("\nConverting frames to video...")
    video_path = output_path / 'dance_video.mp4'
    
    # 使用系统ffmpeg
    import subprocess
    
    # 方法1: 尝试使用libx264
    cmd1 = [
        'ffmpeg', '-y',
        '-framerate', '30',
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        str(video_path)
    ]
    
    # 方法2: 如果libx264不可用，使用mpeg4
    cmd2 = [
        'ffmpeg', '-y',
        '-framerate', '30',
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'mpeg4',
        '-q:v', '5',
        str(video_path)
    ]
    
    # 方法3: 使用libvpx (webm格式)
    video_path_webm = output_path / 'dance_video.webm'
    cmd3 = [
        'ffmpeg', '-y',
        '-framerate', '30',
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libvpx',
        '-b:v', '1M',
        str(video_path_webm)
    ]
    
    # 尝试不同的编码器
    success = False
    for cmd, output_file in [(cmd1, video_path), (cmd2, video_path), (cmd3, video_path_webm)]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Video saved to {output_file}")
                success = True
                break
        except:
            continue
    
    if not success:
        print("⚠️  Could not encode video with ffmpeg")
        print(f"   Frames are available at: {frames_dir}")
        print(f"   You can manually convert them with:")
        print(f"   ffmpeg -framerate 30 -i {frames_dir}/frame_%04d.png -c:v libx264 output.mp4")


def render_sequence(args):
    """渲染序列图"""
    print("Setting up scene...")
    setup_scene()
    
    ply_dir = Path(args.dir) / 'ply'
    if not ply_dir.exists():
        print(f"PLY directory not found: {ply_dir}")
        return
    
    ply_files = sorted(ply_dir.glob('frame_*.ply'))
    if len(ply_files) == 0:
        print(f"No PLY files found in {ply_dir}")
        return
    
    print(f"Found {len(ply_files)} PLY files")
    
    # 创建材质
    body_mat = create_material()
    
    # 计算布局
    num_frames = min(len(ply_files), 50)  # 最多显示50帧
    frames_per_row = 10
    num_rows = (num_frames + frames_per_row - 1) // frames_per_row
    
    # 调整相机
    bpy.context.scene.camera.location = (0, -frames_per_row * 1.2, num_rows * 1.2)
    
    # 导入并排列所有帧
    print(f"Creating sequence visualization (showing {num_frames} frames)...")
    for i in range(num_frames):
        ply_file = ply_files[i]
        
        # 导入
        bpy.ops.wm.ply_import(filepath=str(ply_file))
        frame_mesh = bpy.context.selected_objects[0]
        
        # 计算位置
        row = i // frames_per_row
        col = i % frames_per_row
        frame_mesh.location = (col * 2.0, 0, -row * 2.0)
        
        # 添加材质
        frame_mesh.data.materials.clear()
        frame_mesh.data.materials.append(body_mat)
        
        # 平滑着色
        for poly in frame_mesh.data.polygons:
            poly.use_smooth = True
    
    # 渲染
    output_path = Path(args.dir) / 'render'
    output_path.mkdir(exist_ok=True)
    
    sequence_path = output_path / 'sequence.png'
    bpy.context.scene.render.filepath = str(sequence_path)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    print("Rendering sequence image...")
    bpy.ops.render.render(write_still=True)
    
    print(f"✅ Sequence image saved to {sequence_path}")


def main():
    # 解析参数
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='video', 
                       choices=['video', 'sequence'])
    
    args = parser.parse_args(argv)
    
    print(f"Rendering mode: {args.mode}")
    print(f"Input directory: {args.dir}")
    
    if args.mode == 'video':
        render_video_simple(args)
    elif args.mode == 'sequence':
        render_sequence(args)


if __name__ == '__main__':
    main()
import os
import sys
import time
import json
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import torch
import numpy as np
import cv2
from PIL import Image

# Add CausVid to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf


def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    将文本转换为安全的文件名
    
    Args:
        text: 原始文本
        max_length: 最大长度
        
    Returns:
        安全的文件名
    """
    # 移除或替换不安全的字符
    safe_text = re.sub(r'[<>:"/\\|?*]', '_', text)
    safe_text = re.sub(r'\s+', '_', safe_text)  # 替换空格为下划线
    safe_text = re.sub(r'_+', '_', safe_text)   # 合并多个下划线
    safe_text = safe_text.strip('_')            # 移除首尾下划线
    
    # 截断长度
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length].rstrip('_')
    
    # 确保不为空
    if not safe_text:
        safe_text = "untitled"
    
    return safe_text


def generate_trajectory_visualization(video_path: str, output_path: str = None, 
                                    color_scheme: str = 'heat', 
                                    erase_opacity: float = 0.3,
                                    add_opacity: float = 1.0,
                                      threshold=0.1) -> str:
    """
    从视频中生成绘画轨迹可视化图片，区分新增和擦除操作
    
    Args:
        video_path: 视频文件路径
        output_path: 输出图片路径（可选）
        color_scheme: 颜色方案 ('rainbow', 'heat', 'cool')
        erase_opacity: 擦除操作的不透明度 (0-1)
        add_opacity: 新增操作的不透明度 (0-1)
        
    Returns:
        生成的轨迹图片路径
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return None
    
    # 生成输出路径
    if output_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_path = f"{base_name}_trajectory.png"
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return None
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为灰度图像并归一化到0-1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(gray_frame)
    
    cap.release()
    
    if len(frames) < 2:
        logging.error("Need at least 2 frames for trajectory analysis")
        return None
    
    logging.info(f"Analyzing trajectory from {len(frames)} frames")
    
    # 初始化轨迹图像
    h, w = frames[0].shape
    trajectory_img = np.ones((h, w, 3), dtype=np.float32) * 255  # 白色背景
    
    # 分别统计新增和擦除的运动量来计算进度
    add_amounts = []
    erase_amounts = []
    
    for i in range(1, len(frames)):
        prev_frame = frames[i-1]
        curr_frame = frames[i]
        
        # 计算有向差分：正值表示新增（变暗），负值表示擦除（变亮）
        directed_diff = prev_frame - curr_frame
        
        # 分离新增和擦除
        add_mask = directed_diff > threshold  # 像素变暗（新增内容）
        erase_mask = directed_diff < -threshold  # 像素变亮（擦除内容）
        
        add_amount = np.sum(directed_diff[add_mask]) if np.any(add_mask) else 0
        erase_amount = np.sum(-directed_diff[erase_mask]) if np.any(erase_mask) else 0
        
        add_amounts.append(add_amount)
        erase_amounts.append(erase_amount)
    
    # 计算累计运动量和进度
    add_amounts = np.array(add_amounts, dtype=np.float32)
    erase_amounts = np.array(erase_amounts, dtype=np.float32)
    
    cumulative_add = np.cumsum(add_amounts)
    cumulative_erase = np.cumsum(erase_amounts)
    
    # 避免除零错误
    total_add = cumulative_add[-1] if cumulative_add[-1] > 0 else 1
    total_erase = cumulative_erase[-1] if cumulative_erase[-1] > 0 else 1
    
    progress_by_add = cumulative_add / total_add
    # progress_by_erase = cumulative_erase / total_erase
    progress_by_erase = progress_by_add
    
    # 重新绘制轨迹，区分新增和擦除
    for i in range(1, len(frames)):
        prev_frame = frames[i-1]
        curr_frame = frames[i]
        
        # 计算有向差分
        directed_diff = prev_frame - curr_frame
        
        # 分离新增和擦除区域
        add_mask = directed_diff > threshold
        erase_mask = directed_diff < -threshold
        
        # 获取当前帧的进度
        add_progress = progress_by_add[i-1]
        erase_progress = progress_by_erase[i-1]
        
        # 为新增和擦除分别计算颜色
        if color_scheme == 'rainbow':
            # 新增：完整彩虹色
            add_hue = int((1 - add_progress) * 240)
            add_color_hsv = np.array([[[add_hue, 255, 255]]], dtype=np.uint8)
            add_color_bgr = cv2.cvtColor(add_color_hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
            # 擦除：绿色渐变
            erase_color_bgr = np.array([
                0,  # Blue
                255 * (1 - 0.7 * erase_progress),  # Green from 76 to 255
                0   # Red
            ], dtype=np.float32)
        elif color_scheme == 'heat':
            # 新增：蓝到红的热力图
            add_color_bgr = np.array([
                (1 - add_progress) * 255,  # Blue
                0,                         # Green
                add_progress * 255,        # Red
            ], dtype=np.float32)
            # 擦除：绿色渐变
            erase_color_bgr = np.array([
                0,  # Blue
                255 * (1 - 0.7 * erase_progress),  # Green from 76 to 255
                0   # Red
            ], dtype=np.float32)
            # erase_color_bgr = add_color_bgr
        else:  # cool
            # 新增：冷色调
            add_color_bgr = np.array([
                255 * (0.5 + 0.5 * add_progress),
                255 * (1 - add_progress * 0.5),
                255 * (0.3 + 0.7 * add_progress)
            ], dtype=np.float32)
            # 擦除：绿色渐变
            erase_color_bgr = np.array([
                0,  # Blue
                255 * (1 - 0.7 * erase_progress),  # Green from 76 to 255
                0   # Red
            ], dtype=np.float32)
        
        # 应用新增内容（完整不透明度）
        if np.any(add_mask):
            # add_intensity = directed_diff[add_mask]
            add_intensity = 1
            for c in range(3):
                # 使用混合模式叠加新增内容
                current_values = trajectory_img[:, :, c][add_mask]
                new_values = current_values * (1 - add_opacity * add_intensity ) + add_color_bgr[c] * add_intensity * add_opacity
                trajectory_img[:, :, c][add_mask] = np.clip(new_values, 0, 255)
        
        # 应用擦除内容（降低不透明度）
        if np.any(erase_mask):
            # erase_intensity = -directed_diff[erase_mask]
            erase_intensity = 1
            for c in range(3):
                # 使用混合模式叠加擦除内容，但透明度更低
                current_values = trajectory_img[:, :, c][erase_mask]
                new_values = 255 * (1 - erase_opacity * erase_intensity) + erase_color_bgr[c] * erase_opacity * erase_intensity
                trajectory_img[:, :, c][erase_mask] = np.clip(new_values, 0, 255)
    
    # 转换为uint8
    trajectory_img = np.clip(trajectory_img, 0, 255).astype(np.uint8)
    
    # 添加增强的时间刻度条，包含新增/擦除说明
    trajectory_img = _add_enhanced_time_legend(trajectory_img, color_scheme, erase_opacity, add_opacity)
    
    # 保存图像
    cv2.imwrite(output_path, trajectory_img)
    logging.info(f"Enhanced trajectory visualization saved to: {output_path}")
    logging.info(f"Add operations shown with {add_opacity*100:.0f}% opacity, erase operations with {erase_opacity*100:.0f}% opacity")
    
    return output_path


def _add_enhanced_time_legend(img: np.ndarray, color_scheme: str = 'rainbow', 
                             erase_opacity: float = 0.3, add_opacity: float = 1.0) -> np.ndarray:
    """
    为轨迹图像添加增强的时间图例，显示新增和擦除操作的区别
    
    Args:
        img: 输入图像
        color_scheme: 颜色方案
        erase_opacity: 擦除操作的不透明度
        add_opacity: 新增操作的不透明度
        
    Returns:
        添加图例后的图像
    """
    h, w = img.shape[:2]
    legend_height = 60  # 增加高度以容纳两行图例
    legend_width = min(250, w // 2)
    
    # 创建图例背景
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255  # 白色背景
    # 绘制边框
    cv2.rectangle(legend, (1, 1), (legend_width-2, legend_height-2), (100, 100, 100), 1)
    
    # 上半部分：新增操作图例
    add_legend_y = 8
    add_legend_height = 15
    for i in range(legend_width - 20):
        progress = i / (legend_width - 20)
        
        if color_scheme == 'rainbow':
            hue = int((1 - progress) * 240)
            color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        elif color_scheme == 'heat':
            color_bgr = [
                int((1 - progress) * 255),  # Blue
                0,                          # Green
                int(progress * 255),        # Red
            ]
        else:  # cool
            color_bgr = [
                int(255 * (0.5 + 0.5 * progress)),
                int(255 * (1 - progress * 0.5)),
                int(255 * (0.3 + 0.7 * progress))
            ]
        
        # 应用新增操作的不透明度
        final_color = [int(c * add_opacity + (1-add_opacity) * 255) for c in color_bgr]
        legend[add_legend_y:add_legend_y + add_legend_height, 10 + i] = final_color
    
    # 下半部分：擦除操作图例（绿色渐变）
    erase_legend_y = 28
    erase_legend_height = 15
    for i in range(legend_width - 20):
        progress = i / (legend_width - 20)
        color_bgr = [
            0,
            int(255 * (1 - 0.7 * progress)),
            0
        ]

        # color_bgr = [
        #     int((1 - progress) * 255),  # Blue
        #     0,                          # Green
        #     int(progress * 255),        # Red
        # ]

        final_color = [int(c * erase_opacity + (1-erase_opacity) * 255) for c in color_bgr]
        legend[erase_legend_y:erase_legend_y + erase_legend_height, 10 + i] = final_color
    
    # 添加文字标签
    cv2.putText(legend, f"Add ({add_opacity*100:.0f}%)", (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    # cv2.putText(legend, "Start", (12, 33),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.25, (180, 180, 180), 1)
    # cv2.putText(legend, "End", (legend_width - 35, 33),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.25, (180, 180, 180), 1)
    
    cv2.putText(legend, f"Erase ({erase_opacity*100:.0f}%)", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    # 将图例添加到主图像右上角
    legend_x = w - legend_width - 10
    legend_y = 10
    
    # 确保不超出边界
    if legend_x + legend_width <= w and legend_y + legend_height <= h:
        # 使用alpha混合添加图例
        img_region = img[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width]
        blended = cv2.addWeighted(img_region, 0.3, legend, 0.7, 0)
        img[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width] = blended
    
    return img


def extract_caption_keywords(prompt: str, max_words: int = 3) -> str:
    """
    从prompt中提取关键词作为caption缩略形式
    
    Args:
        prompt: 原始提示词
        max_words: 最大单词数
        
    Returns:
        caption缩略形式
    """
    # 移除常见的无意义词汇
    stop_words = {'a', 'an', 'the', 'of', 'step', 'by', 'draw', 'sketch', 'process', 
                  'first', 'then', 'next', 'finally', 'and', 'or', 'for', 'with', 'in', 'on'}
    
    # 分词并过滤
    words = re.findall(r'\b\w+\b', prompt.lower())
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # 取前几个关键词
    keywords = filtered_words[:max_words]
    
    if not keywords:
        # 如果没有关键词，使用原始方法
        return sanitize_filename(prompt, max_length=20)
    
    # 组合关键词
    caption = '_'.join(keywords)
    return sanitize_filename(caption, max_length=30)


def generate_output_paths(output_dir: str, prompt: str, checkpoint_name: str = None, 
                         timestamp: str = None, seed: int = None) -> Dict[str, str]:
    """
    生成输出文件路径，使用caption缩略形式命名文件夹和文件
    
    Args:
        output_dir: 基础输出目录
        prompt: 提示词
        checkpoint_name: checkpoint名称
        timestamp: 时间戳（可选）
        seed: 随机种子（可选）
        
    Returns:
        包含各种文件路径的字典
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成安全的文件夹名（caption缩略形式）
    safe_folder_name = extract_caption_keywords(prompt)
    
    # # 如果有种子，添加到文件夹名
    # if seed is not None:
    #     safe_folder_name += f"_s{seed}"
    
    caption_dir = os.path.join(output_dir, safe_folder_name)
    os.makedirs(caption_dir, exist_ok=True)
    
    # 文件名使用caption缩略形式
    base_filename = safe_folder_name
    
    return {
        "video_path": os.path.join(caption_dir, f"{base_filename}.mp4"),
        "config_path": os.path.join(caption_dir, f"{base_filename}_config.json"),
        "log_path": os.path.join(caption_dir, f"{base_filename}_log.txt"),
        "trajectory_path": os.path.join(caption_dir, f"{base_filename}_trajectory.png"),
        "folder_name": safe_folder_name,
        "caption_dir": caption_dir,
        "prefix": base_filename,
        "timestamp": timestamp
    }


def save_experiment_config(
    config_path: str,
    args: argparse.Namespace,
    config: Dict[str, Any],
    start_time: float,
    end_time: float,
    video_path: str,
    success: bool = True,
    error_message: str = None,
    prompt: Optional[str] = None
) -> None:
    """
    保存实验配置信息
    
    Args:
        config_path: 配置文件保存路径
        args: 命令行参数
        config: 模型配置
        start_time: 开始时间戳
        end_time: 结束时间戳
        video_path: 视频文件路径
        success: 是否成功
        error_message: 错误信息（如果有）
    """
    if prompt is None:
        prompt = getattr(args, 'prompt', None)
    config_data = {
        "experiment_info": {
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "duration_seconds": round(end_time - start_time, 2),
            "success": success,
            "error_message": error_message,
            "output_video": os.path.basename(video_path)
        },
        "generation_parameters": {
            "prompt": prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "seed": args.seed,
            "fps": args.fps,
            "num_inference_steps": len(config.get('denoising_step_list', [])),
            "denoising_step_list": list(config.get('denoising_step_list', [])),
            # 增长相关参数
            "num_rollout": getattr(args, 'num_rollout', None),
            "num_overlap_frames": getattr(args, 'num_overlap_frames', None),
            "enable_long_video": getattr(args, 'num_rollout', None) is not None,
            # 背景图像参数
            "background_image": getattr(args, 'background_image', None),
            "use_background_image": getattr(args, 'background_image', None) is not None
        },
        "model_configuration": {
            "config_path": args.config_path,
            "checkpoint_folder": args.checkpoint_folder,
            "model_name": config.get('model_name', 'unknown'),
            "generator_name": config.get('generator_name', config.get('model_name', 'unknown')),
            "num_frame_per_block": config.get('num_frame_per_block', 1),
            "warp_denoising_step": config.get('warp_denoising_step', False)
        },
        "system_parameters": {
            "device": args.device,
            "torch_dtype": str(args.torch_dtype),
            "log_level": args.log_level
        },
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }
    

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Experiment config saved to: {config_path}")



def load_prompts_from_file(prompts_file: str) -> List[str]:
    """
    从文件中加载prompts列表
    
    Args:
        prompts_file: prompts文件路径
        
    Returns:
        prompts列表
    """
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        if not prompts:
            raise ValueError(f"No valid prompts found in file: {prompts_file}")
        
        logging.info(f"Loaded {len(prompts)} prompts from: {prompts_file}")
        return prompts
    
    except Exception as e:
        logging.error(f"Failed to load prompts from {prompts_file}: {e}")
        raise


def generate_batch_output_path(base_output_dir: str, checkpoint_folder: str) -> str:
    """
    为批量推理生成输出目录
    
    Args:
        base_output_dir: 基础输出目录
        checkpoint_folder: checkpoint文件夹路径
        
    Returns:
        批量推理的输出目录路径
    """
    # 从checkpoint路径提取模型名称
    checkpoint_name = os.path.basename(checkpoint_folder.rstrip('/'))
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成批量输出目录名
    batch_dir_name = f"{checkpoint_name}_batch_{timestamp}"
    batch_output_dir = os.path.join(base_output_dir, batch_dir_name)
    
    os.makedirs(batch_output_dir, exist_ok=True)
    return batch_output_dir


def validate_inputs(args) -> None:
    """验证输入参数"""
    # 检查checkpoint文件夹是否存在
    if not os.path.exists(args.checkpoint_folder):
        raise FileNotFoundError(f"Checkpoint folder not found: {args.checkpoint_folder}")
    
    # 检查config文件是否存在
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    
    # 检查prompts文件（如果使用）
    if args.prompts_file and not os.path.exists(args.prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
    
    # 检查背景图像（如果使用）
    if hasattr(args, 'background_image') and args.background_image and not os.path.exists(args.background_image):
        raise FileNotFoundError(f"Background image not found: {args.background_image}")
    
    # 验证分辨率和帧数
    if args.height <= 0 or args.width <= 0:
        raise ValueError("Height and width must be positive integers")
    
    if args.num_frames <= 0:
        raise ValueError("Number of frames must be positive")
    
    # 验证长视频相关参数
    if hasattr(args, 'num_rollout') and args.num_rollout is not None:
        if args.num_rollout <= 0:
            raise ValueError("num_rollout must be positive")
        if not hasattr(args, 'num_overlap_frames') or args.num_overlap_frames is None:
            raise ValueError("num_overlap_frames is required when num_rollout is specified")
        if args.num_overlap_frames <= 0:
            raise ValueError("num_overlap_frames must be positive")


def load_checkpoint_and_initialize_pipeline(
    config_path: str,
    checkpoint_folder: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16
) -> tuple:
    """
    加载checkpoint并初始化pipeline（只执行一次）
    
    Args:
        config_path: 配置文件路径
        checkpoint_folder: checkpoint文件夹路径
        device: 计算设备
        torch_dtype: 数据类型
        
    Returns:
        (pipeline, config) 元组
    """
    # 禁用梯度计算
    torch.set_grad_enabled(False)

    # 1. 加载配置
    logging.info(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    # 2. 初始化pipeline
    logging.info("Initializing inference pipeline...")
    pipeline = InferencePipeline(config, device=device)
    pipeline.to(device=device, dtype=torch_dtype)

    # 3. 加载checkpoint
    checkpoint_folder = os.path.abspath(checkpoint_folder)
    checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.pt')]
    assert len(checkpoint_files) == 1, f"Expected exactly one checkpoint file in {checkpoint_folder}, found: {checkpoint_files}"
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_files[0])

    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)['generator']
    pipeline.generator.load_state_dict(state_dict, strict=True)
    logging.info("Checkpoint loaded successfully")
    
    return pipeline, config


def encode_vae(vae, videos: torch.Tensor) -> torch.Tensor:
    """VAE编码函数，用于长视频生成"""
    device, dtype = videos[0].device, videos[0].dtype
    scale = [vae.mean.to(device=device, dtype=dtype),
             1.0 / vae.std.to(device=device, dtype=dtype)]
    output = [
        vae.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]
    output = torch.stack(output, dim=0)
    return output


def load_and_process_background_image(image_path: str, height: int = 480, width: int = 832, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    加载并处理背景图像，转换为适合作为start_latents的格式
    
    Args:
        image_path: 背景图像路径
        height: 目标高度
        width: 目标宽度
        device: 计算设备
        dtype: 数据类型
        
    Returns:
        处理后的图像tensor，格式为[1, 3, H, W]，值范围[0, 1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Background image not found: {image_path}")
    
    # 使用PIL加载图像
    image = Image.open(image_path).convert('RGB')
    logging.info(f"Loaded background image: {image_path}, original size: {image.size}")
    
    # 调整图像尺寸
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    logging.info(f"Resized background image to: {width}x{height}")
    
    # 转换为numpy数组并归一化到[0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # 转换为tensor: [H, W, 3] -> [1, 3, H, W]
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=dtype)
    
    logging.info(f"Background image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
    return image_tensor


def encode_background_to_latents(pipeline, background_image: torch.Tensor) -> torch.Tensor:
    """
    将背景图像编码为VAE latents格式
    
    Args:
        pipeline: 推理pipeline
        background_image: 背景图像tensor [1, 3, H, W]，值范围[0, 1]
        
    Returns:
        编码后的latents tensor
    """
    # 将图像从[0, 1]范围转换为[-1, 1]范围（VAE输入要求）
    background_image_scaled = background_image * 2.0 - 1.0
    # convert to bz, 3, T,  H, W
    background_image_scaled = background_image_scaled.unsqueeze(2)
    background_image_scaled = background_image_scaled.repeat(1, 1, 9, 1, 1)  # [1, 3, T, H, W], T=3 for start frames

    # 使用VAE编码
    device, dtype = background_image.device, background_image.dtype
    scale = [pipeline.vae.mean.to(device=device, dtype=dtype),
             1.0 / pipeline.vae.std.to(device=device, dtype=dtype)]
    
    # VAE编码
    latents = pipeline.vae.model.encode(background_image_scaled, scale).permute(0, 2, 1, 3, 4)  # [1, T, C, H', W']
    
    logging.info(f"Encoded background to latents shape: {latents.shape}, dtype: {latents.dtype}")
    return latents


def run_long_video_inference_with_pipeline(
    pipeline,
    prompt: str,
    num_rollout: int = 3,
    num_overlap_frames: int = 3,
    output_paths: Dict[str, str] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    fps: int = 15,
    trajectory_color_scheme: str = 'rainbow',
    erase_opacity: float = 0.3,
    add_opacity: float = 1.0,
    background_image_path: Optional[str] = None,
    height: int = 480,
    width: int = 832
) -> bool:
    """
    使用已加载的pipeline进行长视频自回归推理
    
    Args:
        pipeline: 已初始化的InferencePipeline
        prompt: 文本提示
        num_rollout: 自回归轮数
        num_overlap_frames: 重叠帧数
        output_paths: 输出路径字典
        seed: 随机种子
        device: 计算设备
        torch_dtype: 数据类型
        fps: 帧率
        trajectory_color_scheme: 轨迹图颜色方案
        erase_opacity: 擦除操作的不透明度 (0-1)
        add_opacity: 新增操作的不透明度 (0-1)
        background_image_path: 背景图像路径（可选）
        height: 视频高度
        width: 视频宽度
        
    Returns:
        是否成功
    """
    output_path = output_paths["video_path"] if output_paths else "output_video.mp4"
    
    # 检查文件是否已存在
    status = True
    if os.path.exists(output_path):
        logging.info(f"Video already exists: {output_path}")
        status =  "exist"
    else:
        # 验证num_overlap_frames与pipeline的兼容性
        if hasattr(pipeline, 'num_frame_per_block'):
            assert num_overlap_frames % pipeline.num_frame_per_block == 0, \
                "num_overlap_frames must be divisible by num_frame_per_block"

        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            logging.info(f"Set random seed to: {seed}")

        logging.info("Starting long video autoregressive inference...")
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Rollout count: {num_rollout}")
        logging.info(f"Overlap frames: {num_overlap_frames}")

        # 处理背景图像（如果提供）
        start_latents = None
        if background_image_path:
            logging.info(f"Using background image for long video: {background_image_path}")
            background_image = load_and_process_background_image(
                background_image_path, height, width, device, torch_dtype
            )
            start_latents = encode_background_to_latents(pipeline, background_image)
            logging.info("Background image encoded to start_latents for long video generation")

        all_video = []

        for rollout_index in range(num_rollout):
            logging.info(f"Processing rollout {rollout_index + 1}/{num_rollout}")

            # 生成随机噪声
            sampled_noise = torch.randn(
                [1, 21, 16, 60, 104], device=device, dtype=torch_dtype
            )

            # 进行推理，获取视频和latents
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=[prompt],
                return_latents=True,
                start_latents=start_latents
            )

            # 转换视频格式
            current_video = video[0].permute(0, 2, 3, 1).cpu().numpy()
            logging.info(f"Generated video segment shape: {current_video.shape}")

            # 为下一轮准备start_latents（除了最后一轮）
            if rollout_index < num_rollout - 1:
                # 编码最后几帧作为下一轮的起始帧
                start_frame = encode_vae(pipeline.vae, (
                    video[:, -4 * (num_overlap_frames - 1) - 1:-4 * (num_overlap_frames - 1), :] * 2.0 - 1.0
                ).transpose(2, 1).to(torch_dtype)).transpose(2, 1).to(torch_dtype)

                # 拼接start_frame和latents的最后几帧
                start_latents = torch.cat(
                    [start_frame, latents[:, -(num_overlap_frames - 1):]], dim=1
                )

                # 添加当前视频片段（移除重叠部分）
                all_video.append(current_video[:-(4 * (num_overlap_frames - 1) + 1)])
            else:
                # 最后一轮，添加完整视频
                all_video.append(current_video)

        # 拼接所有视频片段
        final_video = np.concatenate(all_video, axis=0)
        logging.info(f"Final concatenated video shape: {final_video.shape}")

        # 保存视频
        logging.info(f"Saving long video to: {output_path}")
        export_to_video(final_video, output_path, fps=fps)
        logging.info("Long video saved successfully")

    # 生成轨迹图
    if output_paths and "trajectory_path" in output_paths:
        logging.info("Generating trajectory visualization for long video...")
        trajectory_path = generate_trajectory_visualization(
            video_path=output_paths["video_path"],
            output_path=output_paths["trajectory_path"],
            color_scheme=trajectory_color_scheme,
            erase_opacity=erase_opacity,
            add_opacity=add_opacity
        )
        if trajectory_path:
            logging.info(f"Trajectory visualization generated: {os.path.basename(trajectory_path)}")
        else:
            logging.warning("Failed to generate trajectory visualization")

    return status


def set_random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        logging.info(f"Set random seed to: {seed}")

def generate_noise(shape, device, dtype):
    return torch.randn(shape, device=device, dtype=dtype)

def run_inference_with_pipeline(
    pipeline,
    prompt: str,
    output_paths: Dict[str, str] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    fps: int = 15,
    num_rollout: Optional[int] = None,
    num_overlap_frames: Optional[int] = None,
    generate_trajectory: bool = True,
    trajectory_color_scheme: str = 'rainbow',
    erase_opacity: float = 0.3,
    add_opacity: float = 1.0,
    background_image_path: Optional[str] = None,
    height: int = 480,
    width: int = 832
) -> tuple:
    """
    使用已加载的pipeline进行推理（支持长视频和普通推理）
    
    Args:
        pipeline: 已初始化的InferencePipeline
        prompt: 文本提示
        output_paths: 输出路径字典
        seed: 随机种子
        device: 计算设备
        torch_dtype: 数据类型
        fps: 帧率
        num_rollout: 自回归轮数（用于长视频）
        num_overlap_frames: 重叠帧数（用于长视频）
        generate_trajectory: 是否生成轨迹图
        trajectory_color_scheme: 轨迹图颜色方案
        erase_opacity: 擦除操作的不透明度 (0-1)
        add_opacity: 新增操作的不透明度 (0-1)
        background_image_path: 背景图像路径（可选）
        height: 视频高度
        width: 视频宽度
        
    Returns:
        (是否成功, 推理时长) 元组
    """
    output_path = output_paths["video_path"] if output_paths else "output_video.mp4"
    # 已存在直接返回
    if os.path.exists(output_path):
        logging.info(f"Video already exists: {output_path}")
        result, duration = "exist", 0
    else:
        set_random_seed(seed)
        if num_rollout and num_rollout > 1:
            result = run_long_video_inference_with_pipeline(
                pipeline, prompt, num_rollout, num_overlap_frames,
                output_paths, seed, device, torch_dtype, fps, trajectory_color_scheme,
                erase_opacity, add_opacity, background_image_path, height, width
            )
            duration = 0  # 长视频内部未计时，如需可加
        else:
            noise_shape = [1, 21, 16, 60, 104]
            sampled_noise = generate_noise(noise_shape, device, torch_dtype)
            logging.info(f"Generated noise tensor with shape: {sampled_noise.shape}")
            
            # 处理背景图像（如果提供）
            start_latents = None
            if background_image_path:
                logging.info(f"Using background image: {background_image_path}")
                background_image = load_and_process_background_image(
                    background_image_path, height, width, device, torch_dtype
                )
                start_latents = encode_background_to_latents(pipeline, background_image)
                logging.info("Background image encoded to start_latents")
            
            logging.info("Starting inference...")
            logging.info(f"Prompt: {prompt}")
            if start_latents is not None:
                logging.info("Using background image as starting point")
            start_time = time.time()
            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts=[prompt],
                start_latents=start_latents
            )[0].permute(0, 2, 3, 1).cpu().numpy()
            duration = time.time() - start_time
            logging.info(f"Inference completed. Generated video shape: {video.shape}")
            export_to_video(video, output_path, fps=fps)
            logging.info("Video saved successfully")
            result = True

    # 轨迹图生成（仅在需要时）
    if result and generate_trajectory and output_paths and "trajectory_path" in output_paths:
        logging.info("Generating trajectory visualization...")
        trajectory_path = generate_trajectory_visualization(
            video_path=output_paths["video_path"],
            output_path=output_paths["trajectory_path"],
            color_scheme=trajectory_color_scheme,
            erase_opacity=erase_opacity,
            add_opacity=add_opacity
        )
        if trajectory_path:
            logging.info(f"Trajectory visualization generated: {os.path.basename(trajectory_path)}")
        else:
            logging.warning("Failed to generate trajectory visualization")

    return result, duration


def load_checkpoint_and_inference(
    config_path: str,
    checkpoint_folder: str,
    prompt: str,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    seed: Optional[int] = None,
    output_paths: Dict[str, str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    fps: int = 15,
    num_rollout: Optional[int] = None,
    num_overlap_frames: Optional[int] = None,
    generate_trajectory: bool = True,
    trajectory_color_scheme: str = 'rainbow',
    erase_opacity: float = 0.3,
    add_opacity: float = 1.0
) -> bool:
    """
    使用CausVid进行自回归推理（单次推理的完整流程）
    
    Args:
        config_path: 配置文件路径
        checkpoint_folder: checkpoint文件夹路径
        prompt: 文本提示
        height: 视频高度
        width: 视频宽度
        num_frames: 帧数
        seed: 随机种子
        output_paths: 输出路径字典
        device: 计算设备
        torch_dtype: 数据类型
        fps: 帧率
        num_rollout: 自回归轮数（可选，用于长视频）
        num_overlap_frames: 重叠帧数（可选，用于长视频）
        generate_trajectory: 是否生成轨迹图
        trajectory_color_scheme: 轨迹图颜色方案
        
    Returns:
        是否成功
    """
    # 加载checkpoint和初始化pipeline
    pipeline, config = load_checkpoint_and_initialize_pipeline(
        config_path, checkpoint_folder, device, torch_dtype
    )
    
    # 运行推理
    return run_inference_with_pipeline(
        pipeline, prompt, output_paths, seed, device, torch_dtype, fps,
        num_rollout, num_overlap_frames, generate_trajectory, trajectory_color_scheme,
        erase_opacity, add_opacity
    )


def run_batch_inference(
    prompts: List[str],
    args: argparse.Namespace,
    config: Dict[str, Any]
) -> None:
    """
    运行批量推理
    
    Args:
        prompts: prompts列表
        args: 命令行参数
        config: 模型配置
    """
    # 创建批量输出目录
    checkpoint_name = os.path.basename(args.checkpoint_folder.rstrip('/'))
    seed_suffix = f"seed{args.seed}" if args.seed is not None else ""
    long_suffix = f"_long{args.num_rollout}-{args.num_overlap_frames}" if args.num_rollout and args.num_rollout > 1 else ""
    background_suffix = f"_bg-{args.background_image.split('/')[-1].split('.')[0]}" if args.background_image else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.checkpoint_folder, f"{checkpoint_name}_inference", f"{args.prompts_file.split('/')[-1].split('.')[0]}",f"{seed_suffix}{long_suffix}{background_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Starting batch inference with {len(prompts)} prompts")
    logging.info(f"Batch output directory: {output_dir}")
    if args.num_rollout:
        logging.info(f"Long video mode: {args.num_rollout} rollouts with {args.num_overlap_frames} overlap frames")
    
    # 批量推理统计
    success_count = 0
    failed_count = 0
    failed_prompts = []
    all_results = []
    
    # 转换数据类型
    torch_dtype = getattr(torch, args.torch_dtype)
    
    # 先加载一次checkpoint和初始化pipeline
    logging.info("Loading checkpoint and initializing pipeline...")
    pipeline, loaded_config = load_checkpoint_and_initialize_pipeline(
        config_path=args.config_path,
        checkpoint_folder=args.checkpoint_folder,
        device=args.device,
        torch_dtype=torch_dtype
    )
    # pipeline, loaded_config = None, None
    logging.info("Checkpoint loaded successfully, starting batch inference...")
    
    for i, prompt in enumerate(prompts):
        logging.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt}")
        
        # 为每个prompt生成输出路径
        output_paths = generate_output_paths(
            output_dir,
            prompt,
            checkpoint_name=checkpoint_name,
            seed=args.seed
        )

        start_time = time.time()

        # 执行推理（使用已加载的pipeline）
        success, duration = run_inference_with_pipeline(
            pipeline=pipeline,
            prompt=prompt,
            output_paths=output_paths,
            seed=args.seed,
            device=args.device,
            torch_dtype=torch_dtype,
            fps=args.fps,
            num_rollout=args.num_rollout,
            num_overlap_frames=args.num_overlap_frames,
            generate_trajectory=args.generate_trajectory,
            trajectory_color_scheme=args.trajectory_color_scheme,
            erase_opacity=args.erase_opacity,
            add_opacity=args.add_opacity,
            background_image_path=args.background_image,
            height=args.height,
            width=args.width
        )

        end_time = time.time()

        if success:
            success_count += 1
            duration = round(duration, 2) if not success == "exist" else "skipped (existed)"

            all_results.append({
                "index": i + 1,
                "prompt": prompt,
                "success": True,
                "video_path": output_paths["video_path"],
                "trajectory_path": output_paths.get("trajectory_path", ""),
                "folder_name": output_paths["folder_name"],
                "duration": duration,
                "seed": args.seed,
                "num_rollout": args.num_rollout,
                "num_overlap_frames": args.num_overlap_frames,
                "trajectory_generated": args.generate_trajectory
            })

            # 保存单个推理的配置
            if duration != "skipped (existed)":
                save_experiment_config(
                    config_path=output_paths["config_path"],
                    args=args,
                    config=config,
                    start_time=start_time,
                    end_time=end_time,
                    video_path=output_paths["video_path"],
                    success=True,
                    error_message=None,
                    prompt=prompt
                )

            logging.info(f"✓ Video {i+1} generated successfully in {duration}s")
        else:
            failed_count += 1

        logging.info("---")
    
    # 生成批量推理总结
    summary_path = os.path.join(output_dir, "batch_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CausVid Autoregressive Inference - Batch Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Config: {args.config_path}\n")
        f.write(f"Checkpoint: {args.checkpoint_folder}\n")
        f.write(f"Prompts File: {args.prompts_file}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"- Resolution: {args.width}x{args.height}\n")
        f.write(f"- Frames: {args.num_frames}\n")
        f.write(f"- FPS: {args.fps}\n")
        f.write(f"- Seed: {args.seed}\n")
        f.write(f"- Data Type: {args.torch_dtype}\n")
        f.write(f"- Device: {args.device}\n")
        f.write(f"- Generate Trajectory: {args.generate_trajectory}\n")
        f.write(f"- Trajectory Color Scheme: {args.trajectory_color_scheme}\n")
        if args.num_rollout:
            f.write(f"- Long Video: {args.num_rollout} rollouts, {args.num_overlap_frames} overlap frames\n")
        if args.background_image:
            f.write(f"- Background Image: {args.background_image}\n")
        f.write("\n")
        
        f.write("Results:\n")
        f.write(f"- Total prompts: {len(prompts)}\n")
        f.write(f"- Successful: {success_count}\n")
        f.write(f"- Failed: {failed_count}\n")
        f.write(f"- Success rate: {success_count / len(prompts) * 100:.1f}%\n\n")
        
        f.write("Prompts and Results:\n")
        for result in all_results:
            status = "✓" if result["success"] else "✗"
            f.write(f"{result['index']}. {status} {result['prompt']}\n")
            if result["success"]:
                f.write(f"   → Video: {os.path.basename(result['video_path'])}\n")
                if result.get("trajectory_path") and result.get("trajectory_generated"):
                    f.write(f"   → Trajectory: {os.path.basename(result['trajectory_path'])}\n")
                f.write(f"   → Folder: {result['folder_name']}\n")
                f.write(f"   → Duration: {result['duration']}\n")
            else:
                f.write(f"   → Error: {result['error']}\n")
            if result.get("seed") is not None:
                f.write(f"   → Seed: {result['seed']}\n")
            if result.get("num_rollout"):
                f.write(f"   → Long Video: {result['num_rollout']} rollouts, {result['num_overlap_frames']} overlap\n")
            f.write("\n")
        
        if failed_prompts:
            f.write("Failed Prompts Details:\n")
            for idx, prompt, error in failed_prompts:
                f.write(f"{idx}. {prompt}\n")
                f.write(f"   Error: {error}\n\n")
    
    # 记录最终结果
    logging.info("Batch inference completed!")
    logging.info(f"Generated videos are in: {output_dir}")
    logging.info(f"Total: {len(prompts)}, Success: {success_count}, Failed: {failed_count}")
    logging.info(f"Success rate: {success_count / len(prompts) * 100:.1f}%")
    logging.info(f"Batch summary saved: {summary_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Comprehensive CausVid Autoregressive Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="/home/coder/code/video_sketch/libs/CausVid/configs/wan_causal_ode_finetune.yaml",
        help="Path to CausVid config file"
    )
    parser.add_argument(
        "--checkpoint_folder", 
        type=str, 
        default="/home/coder/code/video_sketch/libs/CausVid/pretrained/tianweiy/CausVid/autoregressive_checkpoint",
        help="Path to checkpoint folder containing .pt"
    )
    
    # Prompt参数 - 支持单个prompt或batch文件
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt", 
        type=str,
        help="Single text prompt for video generation"
    )
    prompt_group.add_argument(
        "--prompts_file", 
        type=str,
        help="Path to text file containing prompts (one per line) for batch inference"
    )
    
    # 生成参数
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--height", 
        type=int, 
        default=480,
        help="Video height (note: actual output resolution depends on model)"
    )
    gen_group.add_argument(
        "--width", 
        type=int, 
        default=832,
        help="Video width (note: actual output resolution depends on model)"
    )
    gen_group.add_argument(
        "--num_frames", 
        type=int, 
        default=81,
        help="Number of frames to generate (note: actual frames depend on model)"
    )
    gen_group.add_argument(
        "--seed", 
        type=int,
        help="Random seed for reproducible generation"
    )
    gen_group.add_argument(
        "--fps", 
        type=int, 
        default=15,
        help="Frames per second for output video"
    )
    
    # 长视频参数
    long_group = parser.add_argument_group("Long Video Parameters")
    long_group.add_argument(
        "--num_rollout",
        default=1,
        type=int,
        help="Number of autoregressive rollouts for long video generation (default: disabled)"
    )
    long_group.add_argument(
        "--num_overlap_frames",
        type=int,
        default=3,
        help="Number of overlap frames between rollouts (required when num_rollout is specified)"
    )
    
    # 输出参数
    output_group = parser.add_argument_group("Output Parameters")
    output_group.add_argument(
        "--output_folder", 
        type=str, 
        default="experiments/causvid_ar_inference",
        help="Output directory path"
    )
    output_group.add_argument(
        "--use_batch_output", 
        action="store_true",
        help="Use batch output directory structure (for script usage)"
    )
    
    # 系统参数
    sys_group = parser.add_argument_group("System Parameters")
    sys_group.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run inference on"
    )
    sys_group.add_argument(
        "--torch_dtype", 
        type=str, 
        choices=["float16", "bfloat16", "float32"], 
        default="bfloat16",
        help="PyTorch data type"
    )
    
    # 其他参数
    parser.add_argument(
        "--log_level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level"
    )
    
    # 轨迹可视化参数
    trajectory_group = parser.add_argument_group("Trajectory Visualization Parameters")
    trajectory_group.add_argument(
        "--generate_trajectory",
        action="store_true",
        default=True,
        help="Generate trajectory visualization image"
    )
    trajectory_group.add_argument(
        "--no_trajectory",
        action="store_true",
        help="Disable trajectory generation"
    )
    trajectory_group.add_argument(
        "--trajectory_color_scheme",
        type=str,
        choices=["rainbow", "heat", "cool"],
        default="heat",
        help="Color scheme for trajectory visualization"
    )
    trajectory_group.add_argument(
        "--erase_opacity",
        type=float,
        default=0.3,
        help="Opacity for erase operations in trajectory visualization (0.0-1.0)"
    )
    trajectory_group.add_argument(
        "--add_opacity",
        type=float,
        default=1.0,
        help="Opacity for add operations in trajectory visualization (0.0-1.0)"
    )
    
    # 背景图像参数
    background_group = parser.add_argument_group("Background Image Parameters")
    background_group.add_argument(
        "--background_image",
        type=str,
        help="Path to background image to start generation from (e.g., blank canvas)"
    )
    
    args = parser.parse_args()
    
    # 处理轨迹生成标志
    if args.no_trajectory:
        args.generate_trajectory = False
    
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 验证输入
    validate_inputs(args)
    
    # 加载配置以获取模型信息
    config = OmegaConf.load(args.config_path)
    config_dict = dict(config)
    
    # 检查是否为批量推理模式
    if args.prompts_file:
        # 批量推理模式
        prompts = load_prompts_from_file(args.prompts_file)
        
        logging.info("=" * 60)
        logging.info("CausVid Autoregressive Batch Inference")
        logging.info("=" * 60)
        logging.info(f"Config: {args.config_path}")
        logging.info(f"Checkpoint: {args.checkpoint_folder}")
        logging.info(f"Model: {config_dict.get('model_name', 'unknown')}")
        logging.info(f"Generator: {config_dict.get('generator_name', config_dict.get('model_name', 'unknown'))}")
        logging.info(f"Prompts: {len(prompts)} from {args.prompts_file}")
        logging.info(f"Seed: {args.seed}")
        logging.info(f"Device: {args.device}, dtype: {args.torch_dtype}")
        if args.num_rollout:
            logging.info(f"Long Video: {args.num_rollout} rollouts with {args.num_overlap_frames} overlap frames")
        if args.background_image:
            logging.info(f"Background Image: {args.background_image}")
        logging.info("=" * 60)
        
        # 运行批量推理
        run_batch_inference(prompts, args, config_dict)
        return
    raise ValueError("Invalid mode")
    # # 单个推理模式
    # logging.info("=" * 60)
    # if args.num_rollout:
    #     logging.info("CausVid Autoregressive Long Video Inference")
    # else:
    #     logging.info("CausVid Autoregressive Single Inference")
    # logging.info("=" * 60)
    
    # # 生成输出路径
    # if args.use_batch_output:
    #     # 批量推理模式，直接使用传入的output_folder
    #     base_output_dir = args.output_folder
    #     os.makedirs(base_output_dir, exist_ok=True)
    #     output_paths = generate_output_paths(
    #         base_output_dir, 
    #         args.prompt, 
    #         checkpoint_name=os.path.basename(args.checkpoint_folder.rstrip('/')),
    #         seed=args.seed
    #     )
    # else:
    #     # 单个推理模式，在output_folder下创建caption文件夹
    #     output_paths = generate_output_paths(
    #         args.output_folder, 
    #         args.prompt, 
    #         checkpoint_name=os.path.basename(args.checkpoint_folder.rstrip('/')),
    #         seed=args.seed
    #     )
    
    # # 设置文件日志处理器
    # file_handler = logging.FileHandler(output_paths["log_path"], encoding='utf-8')
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(logging.Formatter(
    #     '%(asctime)s - %(levelname)s - %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S'
    # ))
    # logging.getLogger().addHandler(file_handler)
    
    # logging.info(f"Starting inference experiment: {output_paths['prefix']}")
    # logging.info(f"Caption folder: {output_paths['folder_name']}")
    # logging.info(f"Output directory: {output_paths['caption_dir']}")
    # logging.info(f"Video will be saved as: {os.path.basename(output_paths['video_path'])}")
    # logging.info(f"Config will be saved as: {os.path.basename(output_paths['config_path'])}")
    # if args.generate_trajectory:
    #     logging.info(f"Trajectory visualization will be saved as: {os.path.basename(output_paths['trajectory_path'])}")
    #     logging.info(f"Trajectory color scheme: {args.trajectory_color_scheme}")
    # if args.num_rollout:
    #     logging.info(f"Long video mode: {args.num_rollout} rollouts with {args.num_overlap_frames} overlap frames")
    
    # start_time = time.time()
    # success = False
    # error_message = None
    

    # # 转换数据类型
    # torch_dtype = getattr(torch, args.torch_dtype)

    # # 执行推理
    # success = load_checkpoint_and_inference(
    #     config_path=args.config_path,
    #     checkpoint_folder=args.checkpoint_folder,
    #     prompt=args.prompt,
    #     height=args.height,
    #     width=args.width,
    #     num_frames=args.num_frames,
    #     seed=args.seed,
    #     output_paths=output_paths,
    #     device=args.device,
    #     torch_dtype=torch_dtype,
    #     fps=args.fps,
    #     num_rollout=args.num_rollout,
    #     num_overlap_frames=args.num_overlap_frames,
    #     generate_trajectory=args.generate_trajectory,
    #     trajectory_color_scheme=args.trajectory_color_scheme,
    #     erase_opacity=args.erase_opacity,
    #     add_opacity=args.add_opacity
    # )

    # end_time = time.time()

    # if success:
    #     logging.info("Inference completed successfully!")

    #     # 保存配置
    #     save_experiment_config(
    #         config_path=output_paths["config_path"],
    #         args=args,
    #         config=config_dict,
    #         start_time=start_time,
    #         end_time=end_time,
    #         video_path=output_paths["video_path"],
    #         success=True,
    #         error_message=None
    #     )

    # else:
    #     logging.error("Inference failed!")



if __name__ == "__main__":
    main()
    # for threshold in [0.1, 0.05]:
    #     generate_trajectory_visualization("/home/coder/code/video_sketch/data/custom_sketch0618/trunc_compress81_sample/bird.mp4", f"bird_trajectory_ref_thres{threshold}.png",threshold=threshold)
    #     generate_trajectory_visualization("/home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr1e-5_ep10_f81_480x832_custom_sketch0618_tr_metadata_detailed_slurm258/checkpoint_epoch_009/checkpoint_epoch_009_inference_seed0/bird_oval_shape/bird_oval_shape.mp4", f"bird_trajectory_gen_thres{threshold}.png",threshold=threshold)
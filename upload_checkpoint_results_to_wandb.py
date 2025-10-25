#!/usr/bin/env python3
"""
Upload checkpoint inference results to WandB (one video per run)
将 checkpoint 测试生成的视频结果上传到 wandb，每个视频作为独立的 run，同时上传相关的训练和推理配置
支持轨迹图像的自动检测和上传功能
支持完整推理配置的记录，包括实验信息、生成参数、模型配置、系统参数和环境信息
支持长视频inference结果的检测和处理（如seed0_long3-3目录结构）
支持负数epoch的检测和处理（如checkpoint_epoch_-001目录结构）
"""

import os
import json
import argparse
import logging
import re
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import wandb


def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_experiment_name(experiment_path: str) -> Dict[str, Any]:
    """
    从实验路径中解析训练参数信息
    
    Args:
        experiment_path: 实验路径
        
    Returns:
        解析出的参数字典
    """
    exp_name = os.path.basename(experiment_path)
    
    # 示例: train_dit_lr1e-5_ep10_f81_480x832_custom_sketch0618_tr_metadata_short_slurm48
    parsed_info = {
        "experiment_name": exp_name,
        "raw_path": experiment_path
    }
    
    # 解析学习率
    lr_match = re.search(r'lr([\d\.e\-]+)', exp_name)
    if lr_match:
        parsed_info["learning_rate"] = lr_match.group(1)
    
    # 解析 epochs
    ep_match = re.search(r'ep(\d+)', exp_name)
    if ep_match:
        parsed_info["num_epochs"] = int(ep_match.group(1))
    
    # 解析帧数
    f_match = re.search(r'f(\d+)', exp_name)
    if f_match:
        parsed_info["num_frames"] = int(f_match.group(1))
    
    # 解析分辨率
    res_match = re.search(r'(\d+)x(\d+)', exp_name)
    if res_match:
        parsed_info["height"] = int(res_match.group(2))
        parsed_info["width"] = int(res_match.group(1))
    
    # 解析数据集信息
    if 'custom_sketch0618' in exp_name:
        parsed_info["dataset"] = "custom_sketch0618"
    
    # 解析 metadata 类型
    # if 'metadata_detailed_noanimal' in exp_name:
    #     parsed_info["metadata_type"] = "detailed_noanimal"
    # elif 'metadata_detailed_nodogcat' in exp_name:
    #     parsed_info["metadata_type"] = "detailed_nodogcat"
    # elif 'metadata_short' in exp_name:
    #     parsed_info["metadata_type"] = "short"
    # elif 'metadata_detailed' in exp_name:
    #     parsed_info["metadata_type"] = "detailed"
    
    # 解析 slurm job id
    slurm_match = re.search(r'slurm(\d+)', exp_name)
    if slurm_match:
        parsed_info["slurm_job_id"] = int(slurm_match.group(1))
    
    return parsed_info


def load_training_config(experiment_path: str) -> Dict[str, Any]:
    """
    加载训练配置信息
    
    Args:
        experiment_path: 实验路径
        
    Returns:
        训练配置字典
    """
    # 加载 args.json
    args_path = os.path.join(experiment_path, "args.json")
    train_config = {}
    
    if os.path.exists(args_path):
        with open(args_path, 'r', encoding='utf-8') as f:
            train_config["training_config"] = json.load(f)
        logging.info(f"Loaded training config from: {args_path}")
    else:
        raise Exception(f"Training config not found: {args_path}")
    
    # 加载环境信息
    env_path = os.path.join(experiment_path, "env_info.json")
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            env_info = json.load(f)
            train_config["environment_info"] = env_info
        logging.info(f"Loaded environment info from: {env_path}")
    
    # 解析实验名称信息
    parsed_info = parse_experiment_name(experiment_path)
    train_config["parsed_experiment_info"] = parsed_info

    
    return train_config


def find_checkpoint_inference_dirs(experiment_path: str) -> List[str]:
    """
    查找所有 checkpoint 推理结果目录（支持新的 checkpoint_epoch_XXX 结构）
    
    Args:
        experiment_path: 实验路径
        
    Returns:
        推理结果目录列表
    """
    # 检查新结构：直接在实验目录下的 checkpoint_epoch_XXX 文件夹
    inference_dirs = []
    
    if not os.path.exists(experiment_path):
        return inference_dirs
    
    # 遍历实验目录中的所有项目
    for item in os.listdir(experiment_path):
        item_path = os.path.join(experiment_path, item)
        
        if not os.path.isdir(item_path):
            continue
            
        # 跳过包含"old"的路径
        if "old" in item.lower():
            logging.info(f"Skipping directory with 'old' in path: {item_path}")
            continue
            
        # 检查是否是 checkpoint_epoch_XXX 格式的目录
        if item.startswith("checkpoint_epoch_"):
            # 在 checkpoint_epoch_XXX 目录中查找推理结果
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                if os.path.isdir(sub_item_path) and "_inference" in sub_item:
                    # 跳过包含"old"的路径
                    if "old" in sub_item.lower():
                        logging.info(f"Skipping inference directory with 'old' in name: {sub_item_path}")
                        continue
                    inference_dirs.append(sub_item_path)
    
    # 按 epoch 排序
    def extract_epoch(path):
        # 从路径中提取 epoch 信息
        # 例如: .../checkpoint_epoch_003/checkpoint_epoch_003_inference_seed0
        # 支持负数 epoch: .../checkpoint_epoch_-001/checkpoint_epoch_-001_inference_seed0
        parent_dir = os.path.basename(os.path.dirname(path))
        match = re.search(r'checkpoint_epoch_(-?\d+)', parent_dir)
        return int(match.group(1)) if match else float('-inf')
    
    inference_dirs.sort(key=extract_epoch)
    return inference_dirs


def find_seed_dirs(inference_dir: str) -> List[str]:
    """
    查找推理目录中的所有子目录（支持多层目录结构）
    
    Args:
        inference_dir: 推理结果目录
        
    Returns:
        包含推理结果的目录列表
    """
    if not os.path.exists(inference_dir):
        return []
    
    def has_video_files(directory):
        """检查目录是否包含视频文件"""
        if not os.path.exists(directory):
            return False
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                for video_ext in ['*.mp4', '*.avi', '*.mov']:
                    if glob.glob(os.path.join(item_path, video_ext)):
                        return True
        return False
    
    def find_seed_dirs_recursive(directory, max_depth=3, current_depth=0):
        """递归查找seed目录，最多递归3层"""
        seed_dirs = []
        
        if current_depth > max_depth:
            return seed_dirs
            
        if not os.path.exists(directory):
            return seed_dirs
        
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if not os.path.isdir(item_path):
                continue
                
            # 跳过包含"old"的目录
            if "old" in item.lower():
                logging.info(f"Skipping directory with 'old' in name: {item_path}")
                continue
            
            # 检查是否是seed目录（包含seed字样或batch_summary.txt）
            is_seed_dir = (item.startswith("seed") or 
                          os.path.exists(os.path.join(item_path, "batch_summary.txt")))
            
            # 如果是seed目录且包含视频文件夹，则添加
            if is_seed_dir and has_video_files(item_path):
                logging.info(f"Found seed directory: {item_path}")
                seed_dirs.append(item_path)
            # 如果不是seed目录，继续递归查找
            elif not is_seed_dir and current_depth < max_depth:
                seed_dirs.extend(find_seed_dirs_recursive(item_path, max_depth, current_depth + 1))
        
        return seed_dirs
    
    # 首先检查是否直接包含视频文件夹（最新结构）
    if has_video_files(inference_dir):
        logging.info(f"Found direct video files in inference directory: {inference_dir}")
        return [inference_dir]
    
    # 否则递归查找seed目录
    seed_dirs = find_seed_dirs_recursive(inference_dir)
    
    # 按seed数字排序
    def extract_seed_num(path):
        # 尝试从路径中提取seed数字
        path_parts = path.split(os.sep)
        for part in reversed(path_parts):  # 从后往前找
            match = re.search(r'seed(\d+)', part)
            if match:
                return int(match.group(1))
        return -1
    
    seed_dirs.sort(key=extract_seed_num)
    return seed_dirs


def load_batch_summary(seed_dir: str) -> Optional[Dict[str, Any]]:
    """
    加载批量推理摘要信息（支持新的文件夹结构）
    
    Args:
        seed_dir: seed目录路径或inference目录路径
        
    Returns:
        摘要信息字典
    """
    # 尝试多个可能的batch_summary.txt位置
    possible_paths = [
        os.path.join(seed_dir, "batch_summary.txt"),
        os.path.join(os.path.dirname(seed_dir), "batch_summary.txt"),  # 上级目录
    ]
    
    summary_path = None
    for path in possible_paths:
        if os.path.exists(path):
            summary_path = path
            break
    
    if not summary_path:
        logging.warning(f"No batch_summary.txt found for {seed_dir}")
        return None
    
    summary_info = {}
    
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析基本信息
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("Checkpoint:"):
                summary_info["checkpoint_path"] = line.split(":", 1)[1].strip()
            elif line.startswith("Prompts File:"):
                summary_info["prompts_file"] = line.split(":", 1)[1].strip()
            elif line.startswith("Output Directory:"):
                summary_info["output_directory"] = line.split(":", 1)[1].strip()
            elif line.startswith("Timestamp:"):
                summary_info["timestamp"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Total prompts:"):
                summary_info["total_prompts"] = int(line.split(":")[1].strip())
            elif line.startswith("- Successful:"):
                summary_info["successful_count"] = int(line.split(":")[1].strip())
            elif line.startswith("- Failed:"):
                summary_info["failed_count"] = int(line.split(":")[1].strip())
            elif line.startswith("- Success rate:"):
                summary_info["success_rate"] = float(line.split(":")[1].strip().rstrip('%'))
            elif line.startswith("- Resolution:"):
                resolution = line.split(":")[1].strip()
                if 'x' in resolution:
                    w, h = resolution.split('x')
                    summary_info["resolution"] = {"width": int(w), "height": int(h)}
            elif line.startswith("- Frames:"):
                summary_info["frames"] = int(line.split(":")[1].strip())
            elif line.startswith("- CFG Scale:"):
                summary_info["cfg_scale"] = float(line.split(":")[1].strip())
            elif line.startswith("- Steps:"):
                summary_info["inference_steps"] = int(line.split(":")[1].strip())
            elif line.startswith("- Data Type:"):
                summary_info["data_type"] = line.split(":")[1].strip()
        
        # 解析每个 prompt 的结果
        prompt_results = []
        in_prompts_section = False
        
        for line in lines:
            line = line.strip()
            if line == "Prompts and Results:":
                in_prompts_section = True
                continue
            elif line == "Failed Prompts Details:" or line == "Generated Folders:":
                in_prompts_section = False
                continue
                
            if in_prompts_section and line and re.match(r'^\d+\.', line):
                # 解析 prompt 结果行
                parts = line.split(" ", 2)
                if len(parts) >= 3:
                    index = parts[0].rstrip('.')
                    status = parts[1]
                    prompt = parts[2]
                    
                    prompt_results.append({
                        "index": int(index),
                        "success": status == "✓",
                        "prompt": prompt
                    })
        
        summary_info["prompt_results"] = prompt_results
        
        return summary_info
        
    except Exception as e:
        logging.error(f"Failed to parse batch summary from {summary_path}: {e}")
        return None


def extract_prompt_index_from_batch_summary(batch_summary: Dict[str, Any], folder_name: str) -> Optional[int]:
    """
    从批量摘要信息中提取指定文件夹对应的prompt索引
    
    Args:
        batch_summary: 批量摘要信息
        folder_name: 文件夹名称
        
    Returns:
        对应的prompt索引，如果找不到返回None
    """
    if not batch_summary or "prompt_results" not in batch_summary:
        return None
    
    # 遍历所有prompt结果，查找匹配的文件夹名
    for prompt_result in batch_summary["prompt_results"]:
        if prompt_result.get("success", False):
            # 从prompt生成文件夹名进行比较
            # 这里使用与 extract_caption_keywords 相同的逻辑
            prompt = prompt_result.get("prompt", "")
            expected_folder = extract_caption_keywords(prompt)
            if expected_folder == folder_name:
                return prompt_result.get("index")
    
    return None


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    清理文本作为安全的文件名（与推理脚本保持一致）
    
    Args:
        text: 原始文本
        max_length: 最大长度
        
    Returns:
        安全的文件名
    """
    # 移除或替换不安全的字符
    safe_text = re.sub(r'[^\w\s\-_]', '', text)
    safe_text = re.sub(r'[\s]+', '_', safe_text)
    safe_text = re.sub(r'[_\-]+', '_', safe_text)
    safe_text = safe_text.strip('_-')
    
    # 限制长度
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length].rstrip('_-')
    
    # 确保不为空
    if not safe_text:
        safe_text = "video"
    
    return safe_text


def extract_caption_keywords(prompt: str, max_words: int = 3) -> str:
    """
    从prompt中提取关键词作为caption缩略形式（与推理脚本保持一致）
    
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


def find_video_files(seed_dir: str, batch_summary: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """
    查找目录中的所有视频文件，并尝试提取原始prompt索引（支持新的文件夹结构）
    
    Args:
        seed_dir: seed目录路径或inference目录路径
        batch_summary: 批量摘要信息（用于提取prompt索引）
        
    Returns:
        视频文件信息列表
    """
    videos = []
    
    # 从目录路径中提取seed编号和长视频参数（支持新结构）
    # 新结构例如: checkpoint_epoch_003_inference_seed0 或 checkpoint_epoch_003_inference_seed0_long3-3
    dir_name = os.path.basename(seed_dir)
    seed_match = re.search(r'seed(\d+)', dir_name)
    seed_number = int(seed_match.group(1)) if seed_match else None
    
    # 提取长视频参数（如果存在）
    long_video_params = None
    long_match = re.search(r'_long(\d+)-(\d+)', dir_name)
    if long_match:
        num_rollout = int(long_match.group(1))
        num_overlap_frames = int(long_match.group(2))
        long_video_params = {
            "num_rollout": num_rollout,
            "num_overlap_frames": num_overlap_frames,
            "is_long_video": True
        }
        logging.info(f"Detected long video parameters from directory {dir_name}: rollout={num_rollout}, overlap={num_overlap_frames}")
    else:
        long_video_params = {
            "num_rollout": 1,
            "num_overlap_frames": 0,
            "is_long_video": False
        }
    
    # 遍历目录下的所有子目录（prompt文件夹）
    if not os.path.exists(seed_dir):
        return videos
        
    items = sorted(os.listdir(seed_dir))
    for item in items:
        item_path = os.path.join(seed_dir, item)
        if os.path.isdir(item_path):
            # 跳过包含"old"的目录
            if "old" in item.lower():
                logging.info(f"Skipping video directory with 'old' in name: {item_path}")
                continue
                
            # 查找视频文件
            for video_ext in ['*.mp4', '*.avi', '*.mov']:
                video_files = sorted(glob.glob(os.path.join(item_path, video_ext)))
                for video_file in video_files:
                    # 查找对应的配置文件
                    video_name = os.path.splitext(os.path.basename(video_file))[0]
                    config_file = os.path.join(item_path, f"{video_name}_config.json")
                    
                    # 查找对应的轨迹图像文件
                    trajectory_file = os.path.join(item_path, f"{video_name}_trajectory.png")
                    
                    # 尝试从批量摘要中提取prompt索引
                    prompt_index = extract_prompt_index_from_batch_summary(batch_summary, item)
                    
                    video_info = {
                        "video_path": video_file,
                        "video_name": video_name,
                        "folder_name": item,
                        "seed_number": seed_number,  # 添加seed编号
                        "output_dir": seed_dir,      # 添加输出目录路径
                        "seed_dir_name": dir_name,   # 添加完整的seed目录名称（如seed0_long3-3_bg）
                        "config_path": config_file if os.path.exists(config_file) else None,
                        "trajectory_path": trajectory_file if os.path.exists(trajectory_file) else None,  # 添加轨迹图像路径
                        "prompt_index": prompt_index,  # 添加prompt索引
                        "long_video_params": long_video_params  # 添加长视频参数
                    }
                    
                    # 加载配置信息
                    if video_info["config_path"]:
                        try:
                            with open(video_info["config_path"], 'r', encoding='utf-8') as f:
                                video_info["config"] = json.load(f)
                        except Exception as e:
                            logging.warning(f"Failed to load config for {video_file}: {e}")
                    
                    videos.append(video_info)
    
    return videos


def extract_epoch_from_path(inference_dir: str) -> Optional[int]:
    """
    从推理目录路径中提取 epoch 信息（支持新的文件夹结构和负数 epoch）
    
    Args:
        inference_dir: 推理目录路径
        
    Returns:
        epoch 数字（可能为负数）
    """
    # 新结构：从 checkpoint_epoch_XXX_inference_... 中提取
    # 例如: .../checkpoint_epoch_003/checkpoint_epoch_003_inference_seed0
    # 支持负数: .../checkpoint_epoch_-001/checkpoint_epoch_-001_inference_seed0
    
    # 首先尝试从父目录提取（新结构）
    parent_dir = os.path.basename(os.path.dirname(inference_dir))
    match = re.search(r'checkpoint_epoch_(-?\d+)', parent_dir)
    if match:
        return int(match.group(1))
    
    # 然后尝试从目录名提取（新结构）
    dir_name = os.path.basename(inference_dir)
    match = re.search(r'checkpoint_epoch_(-?\d+)', dir_name)
    if match:
        return int(match.group(1))
    
    # 最后尝试旧结构（也支持负数）
    match = re.search(r'epoch-(-?\d+)_inference', dir_name)
    if match:
        return int(match.group(1))
    
    return None


def get_prompt_text(video_info, batch_summary):
    """
    获取视频的原始prompt，优先顺序：
    1. config中的prompt
    2. batch_summary中的prompt（通过folder_name匹配）
    3. 文件夹名兜底
    """
    prompt = video_info.get("config", {}).get("generation_parameters", {}).get("prompt")
    if prompt and prompt.strip():  # 确保prompt不是None或空字符串
        return prompt
    folder_name = video_info.get("folder_name", "unknown")
    if batch_summary and "prompt_results" in batch_summary:
        for pr in batch_summary["prompt_results"]:
            if pr.get("success") and extract_caption_keywords(pr.get("prompt", "")) == folder_name:
                return pr["prompt"]
    return folder_name


def get_existing_run_names(entity: str, project: str) -> set:
    """
    获取指定entity/project下所有run的名称集合
    """
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        return set(run.name for run in runs)
    except Exception as e:
        logging.warning(f"Failed to fetch existing runs for project {project} in entity {entity}: {e}")
        return set()


def upload_video_to_wandb(
    video_info: Dict[str, Any],
    epoch: int,
    train_config: Dict[str, Any],
    batch_summary: Optional[Dict[str, Any]],
    project_name: str,
    base_tags: List[str] = None,
    experiment_name: str = None,
    existing_run_names: set = None  # 新增参数
) -> None:
    """
    上传单个视频到 WandB 作为独立的 run
    
    Args:
        video_info: 视频信息字典
        epoch: 训练轮次
        train_config: 训练配置
        batch_summary: 批量摘要信息
        project_name: WandB 项目名称
        base_tags: 基础标签列表
        experiment_name: 实验名称
        existing_run_names: 已有run名称集合
    """
    # try:
    # 获取prompt索引、文本和seed信息
    prompt_index = video_info.get("prompt_index", 0)
    prompt_text = get_prompt_text(video_info, batch_summary)
    seed_number = video_info.get("seed_number", 0)

    # 调试日志
    logging.debug(f"Video info debug - prompt_index: {prompt_index}, seed_number: {seed_number}")
    logging.debug(f"Video info debug - prompt_text: {prompt_text}")
    logging.debug(f"Video info debug - experiment_name: {experiment_name}")

    # 生成运行名称（包含seed信息）
    config_seed = video_info.get('config', {}).get('generation_parameters', {}).get('seed', seed_number)
    safe_experiment_name = experiment_name or "unknown_experiment"

    # 确保所有变量都不是None
    safe_prompt_index = prompt_index if prompt_index is not None else 0
    safe_folder_name = video_info.get("folder_name", "unknown")
    safe_seed_number = seed_number if seed_number is not None else 0
    safe_config_seed = config_seed if config_seed is not None else safe_seed_number
    safe_inference_steps = video_info.get("config", {}).get("generation_parameters", {}).get("num_inference_steps", "unknown")
    
    # 提取长视频参数，优先使用目录名中的参数，其次使用配置文件中的参数
    long_video_params = video_info.get("long_video_params", {})
    num_roll_out = long_video_params.get("num_rollout") or video_info.get("config", {}).get("generation_parameters", {}).get("num_rollout", 1)
    num_overlap_frames = long_video_params.get("num_overlap_frames") or video_info.get("config", {}).get("generation_parameters", {}).get("num_overlap_frames", 0)
    is_long_video = long_video_params.get("is_long_video", False)

    logging.debug(f"Safe variables - prompt_index: {safe_prompt_index}, seed_number: {safe_seed_number}")
    logging.debug(f"Safe variables - folder_name: {safe_folder_name}, config_seed: {safe_config_seed}")
    logging.debug(f"Long video params - rollout: {num_roll_out}, overlap: {num_overlap_frames}, is_long: {is_long_video}")

    # 获取完整的seed目录名称，如果存在则使用，否则回退到原来的逻辑
    seed_dir_name = video_info.get("seed_dir_name")
    if seed_dir_name:
        # 使用完整的目录名称作为结尾，提供更多灵活性
        run_name_parts = [
            safe_experiment_name,
            f"epoch{epoch}",
            f"prompt{safe_prompt_index:02d}",
            safe_folder_name,
            f"steps{safe_inference_steps}",
            seed_dir_name  # 直接使用完整的目录名称
        ]
    else:
        raise ValueError(f"No seed directory name found for video {video_info['video_name']}")
        # 回退到原来的逻辑，用于向后兼容
        run_name_parts = [
            safe_experiment_name,
            f"epoch{epoch}",
            f"prompt{safe_prompt_index:02d}",
            safe_folder_name,
            f"seed{safe_config_seed}",
            f"steps{safe_inference_steps}"
        ]
        
        if is_long_video and num_roll_out > 1:
            run_name_parts.append(f"long{num_roll_out}-{num_overlap_frames}")
    
    run_name = "_".join(run_name_parts)

    logging.debug(f"Generated run_name: {run_name}")

    # 准备标签
    tags = base_tags.copy() if base_tags else []
    tags.extend([
        f"epoch:{epoch}",
        f"prompt_idx:{safe_prompt_index}",
        f"seed:{safe_seed_number}",
        "single_video"
    ])
    
    # 添加长视频相关标签
    if is_long_video and num_roll_out > 1:
        tags.extend([
            "long_video",
            f"rollout:{num_roll_out}",
            f"overlap:{num_overlap_frames}"
        ])
    else:
        tags.append("short_video")

    # 准备 WandB config
    config = {
        "experiment_name": safe_experiment_name,
        "epoch": epoch,
        "prompt_index": safe_prompt_index,
        "prompt": prompt_text,
        "folder_name": safe_folder_name,
        "video_name": video_info.get("video_name", "unknown"),
        "seed_number": safe_seed_number,  # 添加seed编号
        "output_dir": video_info.get("output_dir"),  # 添加输出目录路径
        "seed_dir_name": video_info.get("seed_dir_name"),  # 添加完整的seed目录名称
        "has_trajectory_image": video_info.get("trajectory_path") is not None and os.path.exists(video_info.get("trajectory_path", "")),  # 添加是否有轨迹图像的标志
        # 长视频参数
        "is_long_video": is_long_video,
        "num_rollout": num_roll_out,
        "num_overlap_frames": num_overlap_frames,
    }

    # 添加训练配置信息
    if train_config:
        parsed_info = train_config.get("parsed_experiment_info", {})
        config.update({
            "learning_rate": parsed_info.get("learning_rate"),
            "num_epochs": parsed_info.get("num_epochs"),
            "num_frames": parsed_info.get("num_frames"),
            "height": parsed_info.get("height"),
            "width": parsed_info.get("width"),
            "dataset": parsed_info.get("dataset"),
            "metadata_type": parsed_info.get("metadata_type"),
            "slurm_job_id": parsed_info.get("slurm_job_id"),
            "experiment_path": parsed_info.get("raw_path"),
        })

        # 添加完整的训练配置
        if "training_config" in train_config and train_config["training_config"]:
            config["training_args"] = train_config["training_config"]
            metadata_path = train_config["training_config"].get('metadata_path')
            if metadata_path:
                config["metadata_type"] = os.path.basename(metadata_path).split('.')[0]

    # 添加完整的推理配置
    if video_info.get("config"):
        config["inference_config"] = video_info["config"]
        logging.debug("Added complete inference config to WandB config")

    # 添加推理配置信息
    if video_info.get("config"):
        config_data = video_info["config"]
        if "generation_parameters" in config_data:
            gen_params = config_data["generation_parameters"]
            config.update({
                "inference_seed": gen_params.get("seed"),
                "inference_cfg_scale": gen_params.get("cfg_scale"),
                "inference_num_steps": gen_params.get("num_inference_steps"),
                "inference_height": gen_params.get("height"),
                "inference_width": gen_params.get("width"),
                "inference_num_frames": gen_params.get("num_frames"),
                "inference_data_type": gen_params.get("data_type"),
                "trajectory_color_scheme": gen_params.get("trajectory_color_scheme"),
                "trajectory_erase_opacity": gen_params.get("erase_opacity"),
                "trajectory_add_opacity": gen_params.get("add_opacity"),
            })

        if "experiment_info" in config_data:
            exp_info = config_data["experiment_info"]
            config.update({
                "generation_duration_seconds": exp_info.get("duration_seconds"),
                "generation_timestamp": exp_info.get("timestamp"),
            })

    # 添加批量摘要信息
    if batch_summary:
        config.update({
            "prompts_file": batch_summary.get("prompts_file"),
            "batch_checkpoint_path": batch_summary.get("checkpoint_path"),
            "batch_output_directory": batch_summary.get("output_directory"),
            "batch_timestamp": batch_summary.get("timestamp"),
            "batch_total_prompts": batch_summary.get("total_prompts"),
            "batch_successful_count": batch_summary.get("successful_count"),
            "batch_failed_count": batch_summary.get("failed_count"),
            "batch_success_rate": batch_summary.get("success_rate"),
        })

    # 初始化 WandB run
    if existing_run_names is not None and run_name in existing_run_names:
        logging.info(f"Run {run_name} already exists, skipping upload.")
        return

    run = wandb.init(
        entity="video-sketch",
        project=project_name,
        name=run_name,
        tags=tags,
        config=config,
        resume="allow",
        reinit=True  # 允许在同一个进程中初始化多个run
    )

    # try:
        # 创建视频caption
    caption = f"Epoch {epoch} - Prompt #{safe_prompt_index:02d} - Seed {safe_seed_number}: {prompt_text}"

    # 上传视频
    wandb_video = wandb.Video(
        video_info["video_path"],
        # caption=caption,
        format="mp4"
    )

    # 准备记录数据
    log_data = {
        "video": wandb_video,
        "prompt": prompt_text,
        "epoch": epoch,
        "prompt_index": safe_prompt_index,
    }
    
    # 添加轨迹图像（如果存在）
    if video_info.get("trajectory_path") and os.path.exists(video_info["trajectory_path"]):
        trajectory_image = wandb.Image(
            video_info["trajectory_path"],
            caption=f"Trajectory visualization for: {prompt_text}"
        )
        log_data["trajectory_image"] = trajectory_image
        logging.info(f"Added trajectory image: {os.path.basename(video_info['trajectory_path'])}")
    else:
        logging.debug("No trajectory image found for this video")
    
    # 记录视频、轨迹图像和元数据
    wandb.log(log_data)

    logging.info(f"Uploaded video to WandB: {run_name}")
    
    # 实时更新已有run名称集合
    if existing_run_names is not None:
        existing_run_names.add(run_name)
    
    # 完成当前run
    wandb.finish()


def upload_to_wandb(
    experiment_path: str,
    project_name: str,
    run_name: str = None,
    tags: List[str] = None,
    epochs_to_upload: List[int] = None,
    # max_videos_per_epoch: int = 10,
    run_id: str = None,
    resume: bool = True
) -> None:
    """
    上传实验结果到 WandB，每个视频作为独立的 run（支持新的seed子文件夹结构）
    
    Args:
        experiment_path: 实验路径
        project_name: WandB 项目名称
        run_name: 运行名称（用作前缀）
        tags: 标签列表
        epochs_to_upload: 要上传的 epoch 列表，None 表示上传所有
        max_videos_per_epoch: 每个 epoch 最多上传的视频数量
        run_id: WandB run id (在新模式下不使用)
        resume: 是否覆盖/续写同一个run (在新模式下不使用)
    """
    # 加载训练配置
    train_config = load_training_config(experiment_path)
    parsed_info = train_config.get("parsed_experiment_info", {})
    
    # 生成实验名称
    experiment_name = run_name or parsed_info.get('experiment_name', 'unknown')
    
    # 生成基础标签
    base_tags = tags.copy() if tags else []
    
    # 添加自动标签
    if parsed_info.get("dataset"):
        base_tags.append(f"dataset:{parsed_info['dataset']}")
    if parsed_info.get("metadata_type"):
        base_tags.append(f"metadata:{parsed_info['metadata_type']}")
    if parsed_info.get("learning_rate"):
        base_tags.append(f"lr:{parsed_info['learning_rate']}")
    
    base_tags.extend(["checkpoint_inference", "video_generation"])
    
    logging.info(f"Starting upload for experiment: {experiment_name}")
    
    # 一次性获取所有已有run名称
    logging.info(f"Fetching existing run names for project: {project_name}")
    existing_run_names = get_existing_run_names("video-sketch", project_name)
    logging.info(f"Fetched {len(existing_run_names)} existing runs.")

    # 查找所有推理结果目录
    inference_dirs = find_checkpoint_inference_dirs(experiment_path)
    logging.info(f"Found {len(inference_dirs)} inference directories")
    
    total_videos_uploaded = 0
    
    for inference_dir in inference_dirs:
        epoch = extract_epoch_from_path(inference_dir)
        
        # 检查是否需要上传这个 epoch
        if epochs_to_upload is not None and epoch not in epochs_to_upload:
            logging.info(f"Skipping epoch {epoch} (not in upload list)")
            continue
        
        logging.info(f"Processing epoch {epoch} inference results...")
        
        # 查找该epoch下的所有seed目录
        seed_dirs = find_seed_dirs(inference_dir)
        logging.info(f"Found {len(seed_dirs)} seed directories for epoch {epoch}")
        
        # 为每个seed目录处理视频
        for seed_dir in seed_dirs:
            seed_name = os.path.basename(seed_dir)
            logging.info(f"Processing {seed_name} for epoch {epoch}...")
            
            # 加载该seed的批量摘要
            batch_summary = load_batch_summary(seed_dir)
            
            # 查找视频文件
            videos = find_video_files(seed_dir, batch_summary)
            logging.info(f"Found {len(videos)} videos in {seed_name} for epoch {epoch}")
            
            # 按照prompt索引排序，确保视频按原始顺序处理
            videos.sort(key=lambda x: x.get("prompt_index") or float('inf'))
            
            # 限制上传数量（每个seed目录单独计算）
            # if len(videos) > max_videos_per_epoch:
            #     logging.info(f"Limiting to {max_videos_per_epoch} videos per seed directory")
            #     videos = videos[:max_videos_per_epoch]
            
            # 为每个视频创建独立的 WandB run
            for i, video_info in enumerate(videos):
                upload_video_to_wandb(
                    video_info=video_info,
                    epoch=epoch,
                    train_config=train_config,
                    batch_summary=batch_summary,
                    project_name=project_name,
                    base_tags=base_tags,
                    experiment_name=experiment_name,
                    existing_run_names=existing_run_names  # 传递已有run名称集合
                )
                total_videos_uploaded += 1

                prompt_index = video_info.get("prompt_index", i + 1)
                seed_number = video_info.get("seed_number", 0)
                logging.info(f"prompt_index: {prompt_index}, seed_number: {seed_number}")
                logging.info(f"Uploaded video {i+1}/{len(videos)} for epoch {epoch} seed {seed_number}: Prompt #{prompt_index:02d}")
                        

    
    # 统计轨迹图像数量
    total_trajectory_images = 0
    inference_dirs = find_checkpoint_inference_dirs(experiment_path)
    for inference_dir in inference_dirs:
        epoch = extract_epoch_from_path(inference_dir)
        if epochs_to_upload is not None and epoch not in epochs_to_upload:
            continue
        
        seed_dirs = find_seed_dirs(inference_dir)
        for seed_dir in seed_dirs:
            batch_summary = load_batch_summary(seed_dir)
            videos = find_video_files(seed_dir, batch_summary)
            videos.sort(key=lambda x: x.get("prompt_index") or float('inf'))
            
            # if len(videos) > max_videos_per_epoch:
            #     videos = videos[:max_videos_per_epoch]
            
            for video_info in videos:
                if video_info.get("trajectory_path") and os.path.exists(video_info["trajectory_path"]):
                    total_trajectory_images += 1
    
    logging.info(f"WandB upload completed! Total videos uploaded: {total_videos_uploaded}")
    logging.info(f"Total trajectory images uploaded: {total_trajectory_images}")
    if total_trajectory_images > 0:
        logging.info(f"Trajectory image upload rate: {total_trajectory_images}/{total_videos_uploaded} ({100*total_trajectory_images/total_videos_uploaded:.1f}%)")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Upload checkpoint inference results to WandB (one video per run)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to the experiment directory"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="Causvid_odefinetune_inference",
        help="WandB project name"
    )
    
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        help="WandB run name prefix (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Additional tags for all runs"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="*",
        help="Specific epochs to upload (upload all if not specified)"
    )
    
    # parser.add_argument(
    #     "--max_videos_per_epoch",
    #     type=int,
    #     default=15,
    #     help="Maximum number of videos to upload per epoch"
    # )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 验证实验路径
    if not os.path.exists(args.experiment_path):
        logging.error(f"Experiment path does not exist: {args.experiment_path}")
        return
    
    if not os.path.isdir(args.experiment_path):
        logging.error(f"Experiment path is not a directory: {args.experiment_path}")
        return
    
    # 上传到 WandB (每个视频作为独立的run)
    try:
        upload_to_wandb(
            experiment_path=args.experiment_path,
            project_name=args.project,
            run_name=args.run_name_prefix,
            tags=args.tags,
            epochs_to_upload=args.epochs,
            # max_videos_per_epoch=args.max_videos_per_epoch,
            run_id=None,  # 每个run都会自动生成新的id
            resume=False  # 每个run都是新的
        )

        logging.info("Upload completed successfully!")
            
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise


if __name__ == "__main__":
    main() 

"""
使用示例 (Usage Examples):

1. 上传所有epoch的视频 (支持新的checkpoint_epoch_XXX结构):
   python upload_checkpoint_results_to_wandb.py /path/to/experiment

2. 只上传特定epoch的视频:
   python upload_checkpoint_results_to_wandb.py /path/to/experiment --epochs 5 10 15

3. 限制每个epoch的每个inference目录上传的视频数量:
   python upload_checkpoint_results_to_wandb.py /path/to/experiment --max_videos_per_epoch 5

4. 指定项目名称和run名称前缀:
   python upload_checkpoint_results_to_wandb.py /path/to/experiment \
       --project "my_video_project" \
       --run_name_prefix "sketch_experiment_v1"

5. 添加自定义标签:
   python upload_checkpoint_results_to_wandb.py /path/to/experiment \
       --tags "test_run" "high_quality" "batch_1"

新的文件夹结构支持:
脚本现在支持以下文件夹结构:
```
experiment_path/
├── checkpoint_epoch_-001/    # 支持负数epoch
│   ├── checkpoint_epoch_-001_inference/
│   │   └── validation_prompts_detailed/
│   │       └── seed0_long3-3/
│   │           ├── batch_summary.txt
│   │           ├── prompt_folder1/
│   │           │   ├── video.mp4
│   │           │   └── video_config.json
│   │           └── ...
├── checkpoint_epoch_000/
│   ├── checkpoint_epoch_000_inference_seed0/
│   │   ├── prompt_folder1/
│   │   │   ├── video.mp4
│   │   │   └── video_config.json
│   │   ├── prompt_folder2/
│   │   └── ...
│   ├── checkpoint_epoch_000_inference_seed0_long3-3/
│   │   ├── prompt_folder1/
│   │   │   ├── video.mp4
│   │   │   └── video_config.json
│   │   └── ...
│   └── batch_summary.txt (可能在inference目录中)
├── checkpoint_epoch_001/
└── ...
```

旧的文件夹结构也继续支持:
```
experiment_path/ckpt/
├── epoch-1_inference/
│   ├── seed0/
│   │   ├── batch_summary.txt
│   │   ├── prompt_folder1/
│   │   │   ├── video.mp4
│   │   │   └── video_config.json
│   │   └── ...
│   ├── seed1/
│   ├── seed2/
│   └── old/  (会被自动跳过)
└── ...
```

注意事项:
- 每个视频会创建一个独立的WandB run
- run名称格式: 
  * 新版本: {prefix}_epoch{epoch}_prompt{index:02d}_{folder_name}_steps{inference_steps}_{seed_dir_name}
  * 向后兼容: {prefix}_epoch{epoch}_prompt{index:02d}_{folder_name}_seed{config_seed}_steps{inference_steps}[_long{rollout}-{overlap}]
- 新版本直接使用完整的seed目录名称（如seed0_long3-3_bg），提供更多灵活性
- 自动支持新的checkpoint_epoch_XXX文件夹结构和旧的ckpt/epoch-X_inference结构
- 支持负数epoch（如checkpoint_epoch_-001），按数值大小正确排序（-5, -3, -1, 1, 3, 5）
- 自动跳过任何包含"old"字段的文件夹路径
- 支持batch_summary.txt文件在多个位置（inference目录或其父目录）
- 所有相关的训练和推理配置都会作为WandB config上传
- 视频按照原始prompt索引排序上传
- max_videos_per_epoch 参数现在是针对每个inference目录的限制
- 自动提取seed信息从inference目录名称（如checkpoint_epoch_003_inference_seed0或checkpoint_epoch_003_inference_seed0_long3-3）
- 完整保留seed目录名称（如seed0_long3-3_bg）在run名称中，提供最大灵活性
- 自动检测长视频参数从目录名称（如_long3-3表示3次rollout和3帧重叠）
- 长视频会添加相应的标签（long_video, rollout:X, overlap:Y）和配置参数（仅向后兼容模式）
- 自动检测并上传轨迹图像（{video_name}_trajectory.png），如果存在的话
- 轨迹图像将作为WandB Image对象上传，包含对应的caption信息
- 配置中会记录是否包含轨迹图像以及轨迹生成的相关参数
- 完整的推理配置会作为inference_config字段上传，包含实验信息、生成参数、模型配置、系统参数和环境信息
- 同时保持向后兼容的单独inference_*参数，便于WandB过滤和分析
- 自动记录prompts file路径以及相关的批量推理信息（checkpoint路径、输出目录、时间戳、成功率等）
""" 
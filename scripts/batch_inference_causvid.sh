#!/bin/bash

# CausVid Autoregressive Batch Inference Script
# This script runs batch inference using the comprehensive autoregressive inference script

CHECKPOINT_FOLDER=$1
PROMPTS_FILE=$2
SEED=${3:-0}
NUM_ROLLOUT=${4:-0}
NUM_OVERLAP_FRAMES=${5:-3}
BACKGROUND_IMAGE=${6:-None}

# 配置参数
CONFIG_PATH="/home/coder/code/video_sketch/libs/CausVid/configs/wan_causal_ode_finetune.yaml"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=15
TORCH_DTYPE="bfloat16"
LOG_LEVEL="INFO"


# 切换到CausVid目录
cd /home/coder/code/video_sketch/libs/CausVid

if [ "$BACKGROUND_IMAGE" != "None" ]; then
    bg_args="--background_image $BACKGROUND_IMAGE"
else
    bg_args=""
fi

echo "Starting inference"

# 调用Python脚本进行批量推理
python minimal_inference/comprehensive_autoregressive_inference.py \
    --config_path "$CONFIG_PATH" \
    --checkpoint_folder "$CHECKPOINT_FOLDER" \
    --prompts_file "$PROMPTS_FILE" \
    --height $HEIGHT \
    --width $WIDTH \
    --num_frames $NUM_FRAMES \
    --seed $SEED \
    --fps $FPS \
    --torch_dtype "$TORCH_DTYPE" \
    --log_level "$LOG_LEVEL" \
    --num_rollout $NUM_ROLLOUT \
    --num_overlap_frames $NUM_OVERLAP_FRAMES \
    $bg_args

echo "Inference completed"
#!/bin/bash

# CausVid Autoregressive Batch Inference Script
# This script runs batch inference using the comprehensive autoregressive inference script

CHECKPOINT_FOLDER=$1
PROMPTS_FILE=$2
SEED=${3:-42}
NUM_ROLLOUT=${4:-0}
NUM_OVERLAP_FRAMES=${5:-3}


# 配置参数
CONFIG_PATH="/home/coder/code/video_sketch/libs/CausVid/configs/wan_causal_dmd.yaml"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
DEVICE="cuda"
TORCH_DTYPE="bfloat16"
LOG_LEVEL="INFO"

echo "Starting CausVid Autoregressive Batch Inference..."
echo "Checkpoint folder: $CHECKPOINT_FOLDER"
echo "Prompts file: $PROMPTS_FILE"
echo "Seed: $SEED"
echo "Output folder: $OUTPUT_FOLDER"
echo "Config: $CONFIG_PATH"
echo "Resolution: ${WIDTH}x${HEIGHT}, Frames: $NUM_FRAMES"
echo "FPS: $FPS, Device: $DEVICE, Data Type: $TORCH_DTYPE"

# 切换到CausVid目录
cd /home/coder/code/video_sketch/libs/CausVid

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
    --device "$DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    --log_level "$LOG_LEVEL"
    --num_rollout $NUM_ROLLOUT \
    --num_overlap_frames $NUM_OVERLAP_FRAMES

if [ $? -eq 0 ]; then
    echo "✓ Batch inference completed successfully"
else
    echo "✗ Batch inference failed"
    exit 1
fi

echo "Batch inference completed!"
echo "Results saved to: $OUTPUT_FOLDER" 
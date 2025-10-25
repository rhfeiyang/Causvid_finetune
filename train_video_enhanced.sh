#!/bin/bash

# Enhanced Video Regression Training Script for CausVid
# This script provides convenient training with enhanced experiment management

set -e  # Exit on any error

# Configuration
CONFIG_PATH=${CONFIG_PATH:-"configs/wan_video_regression.yaml"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"experiments"}
WANDB_PROJECT=${WANDB_PROJECT:-"causvid_video_regression"}

# Training parameters
DATASET_BASE_PATH=${DATASET_BASE_PATH:-"/home/coder/code/video_sketch/data/custom_sketch0618/trunc_compress81_sample"}
METADATA_PATH=${METADATA_PATH:-"/home/coder/code/video_sketch/data/custom_sketch0618/metadata_detailed.csv"}
NUM_FRAMES=${NUM_FRAMES:-81}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-2e-6}
NUM_EPOCHS=${NUM_EPOCHS:-10}
SEED=${SEED:-42}

# Debug mode
DEBUG=${DEBUG:-false}
NO_WANDB=${NO_WANDB:-false}

echo "======================================================"
echo "CausVid Video Regression Training (Enhanced)"
echo "======================================================"
echo "Config: $CONFIG_PATH"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Dataset: $DATASET_BASE_PATH"
echo "Metadata: $METADATA_PATH"
echo "Frames: $NUM_FRAMES, Resolution: ${HEIGHT}x${WIDTH}"
echo "Batch size: $BATCH_SIZE, LR: $LR, Epochs: $NUM_EPOCHS"
echo "Debug: $DEBUG, No WandB: $NO_WANDB"
echo "======================================================"

# Build command arguments
ARGS=(
    --config_path "$CONFIG_PATH"
    --experiment_dir "$EXPERIMENT_DIR"
    --wandb_project "$WANDB_PROJECT"
    --dataset_base_path "$DATASET_BASE_PATH"
    --metadata_path "$METADATA_PATH"
    --num_frames "$NUM_FRAMES"
    --height "$HEIGHT"
    --width "$WIDTH"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --num_epochs "$NUM_EPOCHS"
    --seed "$SEED"
)

# Add optional flags
if [ "$DEBUG" = "true" ]; then
    ARGS+=(--debug)
    echo "Debug mode enabled"
fi

if [ "$NO_WANDB" = "true" ]; then
    ARGS+=(--no_wandb)
    echo "WandB logging disabled"
fi

# Add any additional arguments passed to the script
ARGS+=("$@")

echo "Running command: python causvid/train_video.py ${ARGS[*]}"
echo "======================================================"

# Set up environment for better tqdm display
export PYTHONUNBUFFERED=1

# Check if we should use accelerate or direct python
if command -v accelerate &> /dev/null && [ -z "$DISABLE_ACCELERATE" ]; then
    echo "Using accelerate launch..."
    
    # Create default accelerate config if it doesn't exist
    if [ ! -f "accelerate_config.yaml" ]; then
        echo "Creating default accelerate config..."
        cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    fi
    
    accelerate launch --config_file accelerate_config.yaml causvid/train_video.py "${ARGS[@]}"
else
    echo "Using direct python execution..."
    python causvid/train_video.py "${ARGS[@]}"
fi

echo "======================================================"
echo "Training completed!"
echo "======================================================" 
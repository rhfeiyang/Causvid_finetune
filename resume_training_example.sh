#!/bin/bash

# Example script for resuming CausVid distillation training
# 
# Usage examples:
#
# 1. Resume from the latest checkpoint in an experiment directory:
#    bash resume_training_example.sh /path/to/experiment_dir
#
# 2. Resume from a specific epoch:
#    bash resume_training_example.sh /path/to/experiment_dir --resume_epoch 10
#
# 3. Continue training with modified parameters:
#    bash resume_training_example.sh /path/to/experiment_dir --num_epochs 200 --lr 1e-6

set -e

# Check if experiment directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_directory> [additional_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep100_bs1_ratio5_sketch_captions_slurm333"
    echo "  $0 /path/to/experiment --resume_epoch 50 --num_epochs 200"
    echo "  $0 /path/to/experiment --lr 5e-7 --num_epochs 150"
    exit 1
fi

EXPERIMENT_DIR="$1"
shift  # Remove first argument, keep the rest

# Validate experiment directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Experiment directory does not exist: $EXPERIMENT_DIR"
    exit 1
fi

# Check for checkpoints
CHECKPOINT_COUNT=$(find "$EXPERIMENT_DIR" -name "checkpoint_epoch_*" -type d | wc -l)
if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
    echo "Error: No checkpoints found in $EXPERIMENT_DIR"
    echo "Expected to find directories like checkpoint_epoch_001, checkpoint_epoch_002, etc."
    exit 1
fi

echo "Found $CHECKPOINT_COUNT checkpoint(s) in $EXPERIMENT_DIR"

# Set default values that can be overridden by command line args
DEFAULT_ARGS=(
    --config_path "/home/coder/code/video_sketch/libs/CausVid/configs/wan_causal_dmd_finetune.yaml"
    --data_path "/home/coder/code/video_sketch/data/caption_gen/sketch_captions_quickdraw.txt"
    --batch_size 1
    --lr 1e-5
    --num_epochs 100
    --dataset_repeat 1
    --dfake_gen_update_ratio 5
    --log_iters 200
    --experiment_name "causvid_distillation_resume"
    --resume_from "$EXPERIMENT_DIR"
)

# Combine default args with user-provided args
ALL_ARGS=("${DEFAULT_ARGS[@]}" "$@")

echo "Resuming training from: $EXPERIMENT_DIR"
echo "Additional arguments: $*"
echo ""
echo "Starting training..."

# Run the training script
python /home/coder/code/video_sketch/libs/CausVid/causvid/train_distillation_finetune.py "${ALL_ARGS[@]}" 
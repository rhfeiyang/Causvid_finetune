#!/bin/bash

dataset_base_path=$1
metadata_path=$2
lr=${3:-2e-6}
num_epochs=${4:-10}

echo "Starting video regression training with Accelerate..."
echo "Config: $CONFIG_PATH"

cd /home/coder/code/video_sketch/libs/CausVid

accelerate launch --config_file /home/coder/code/video_sketch/accelerate_config_7gpu.yaml causvid/train_ode_finetune.py --config_path configs/wan_causal_ode_finetune.yaml \
    --dataset_base_path $dataset_base_path \
    --metadata_path $metadata_path \
    --lr $lr \
    --num_epochs $num_epochs
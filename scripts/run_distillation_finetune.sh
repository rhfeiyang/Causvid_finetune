#!/bin/bash

# dataset_base_path=$1
# metadata_path=$2
lr=${1:-2e-6}
num_epochs=${2:-10}
data_path=${3:-None}
data_repeat=${4:-1}
resume_from=${5:-None}

cd /home/coder/code/video_sketch/libs/CausVid

# Add current directory to Python path so causvid module can be imported
export PYTHONPATH="/home/coder/code/video_sketch/libs/CausVid:$PYTHONPATH"

if [ "$resume_from" != "None" ]; then
    resume_arg="--resume_from $resume_from"
else
    resume_arg=""
fi

if [ "$data_path" == "None" ]; then
    echo "data_path is None"
    exit 1
fi

accelerate launch --config_file /home/coder/code/video_sketch/libs/CausVid/accelerate_config.yaml --num_processes 7 causvid/train_distillation_finetune.py --config_path configs/wan_causal_dmd_finetune.yaml \
    --data_path $data_path \
    --generator_ckpt /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr2e-6_ep15_f81_480x832_custom_sketch0618_tr_metadata_detailed_noanimal_slurm315/checkpoint_epoch_009/model_epoch_009.pt \
    --fake_score_lora_ckpt /home/coder/code/video_sketch/finetune/wan/train/experiments/Wan2.1-T2V-1.3B_lora/Wan2.1-T2V-1.3B_lora_dit_r32_lr1e-4_ep10_f81_480x832_custom_sketch0618_tr_metadata_detailed_noanimal_slurm307/ckpt/epoch-1.safetensors \
    --real_score_lora_ckpt /home/coder/code/video_sketch/finetune/wan/train/experiments/Wan2.1-T2V-1.3B_lora/Wan2.1-T2V-1.3B_lora_dit_r32_lr1e-4_ep10_f81_480x832_custom_sketch0618_tr_metadata_detailed_noanimal_slurm307/ckpt/epoch-1.safetensors \
    --real_score_type 1.3B \
    --lr $lr \
    --num_epochs $num_epochs \
    --epoch_save_interval 4 \
    --dataset_repeat $data_repeat \
    $resume_arg
    
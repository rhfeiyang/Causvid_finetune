# CausVid Distillation Resume Training Guide

This document explains how to use the newly implemented resume training functionality in the CausVid distillation training script.

## Overview

The resume training feature allows you to continue training from previously saved checkpoints, which is essential for:
- Recovering from interrupted training runs
- Extending training with more epochs
- Fine-tuning with different hyperparameters
- Checkpoint-based experimentation

## Feature Implementation

### Key Components Added

1. **Checkpoint Loading Method**: `load_checkpoint()` method in `DistillationTrainer` class
2. **Command Line Arguments**: `--resume_from` and `--resume_epoch` parameters
3. **Training State Recovery**: Restores model weights, optimizer states, and training progress
4. **Multi-GPU Support**: Properly handles distributed training scenarios

### Checkpoint Structure

The training script saves checkpoints in the following structure:
```
experiment_directory/
├── checkpoint_epoch_001/
│   └── model_epoch_001.pt
├── checkpoint_epoch_002/
│   └── model_epoch_002.pt
├── checkpoint_epoch_003/
│   └── model_epoch_003.pt
└── ...
```

Each checkpoint file contains:
- Generator model state dict
- Critic model state dict
- Generator optimizer state dict
- Critic optimizer state dict
- Training step and epoch information
- Configuration parameters

## Usage Examples

### 1. Resume from Latest Checkpoint

Resume training from the most recent checkpoint in an experiment directory:

```bash
python train_distillation_finetune.py \
    --config_path /path/to/config.yaml \
    --data_path /path/to/data.txt \
    --resume_from /path/to/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep100_bs1_ratio5_sketch_captions_slurm333 \
    --num_epochs 200
```

### 2. Resume from Specific Epoch

Resume training from a specific epoch checkpoint:

```bash
python train_distillation_finetune.py \
    --config_path /path/to/config.yaml \
    --data_path /path/to/data.txt \
    --resume_from /path/to/experiment_directory \
    --resume_epoch 50 \
    --num_epochs 200
```

### 3. Resume with Modified Parameters

Continue training with different hyperparameters:

```bash
python train_distillation_finetune.py \
    --config_path /path/to/config.yaml \
    --data_path /path/to/data.txt \
    --resume_from /path/to/experiment_directory \
    --lr 5e-7 \
    --num_epochs 150 \
    --batch_size 2
```

### 4. Using the Convenience Script

A helper script `resume_training_example.sh` is provided for easier usage:

```bash
# Resume from latest checkpoint
bash resume_training_example.sh /path/to/experiment_directory

# Resume from specific epoch with additional parameters
bash resume_training_example.sh /path/to/experiment_directory --resume_epoch 10 --lr 1e-6

# Extend training with more epochs
bash resume_training_example.sh /path/to/experiment_directory --num_epochs 200
```

## Command Line Parameters

### Resume-Specific Parameters

- `--resume_from <path>`: Path to the experiment directory containing checkpoints
- `--resume_epoch <int>`: Specific epoch to resume from (optional, defaults to latest)

### Example with All Parameters

```bash
python train_distillation_finetune.py \
    --config_path "/home/coder/code/video_sketch/libs/CausVid/configs/wan_causal_dmd_finetune.yaml" \
    --data_path "/home/coder/code/video_sketch/data/caption_gen/sketch_captions_quickdraw.txt" \
    --batch_size 1 \
    --lr 1e-5 \
    --num_epochs 100 \
    --dataset_repeat 1 \
    --dfake_gen_update_ratio 5 \
    --log_iters 200 \
    --experiment_name "causvid_distillation_continued" \
    --resume_from "/path/to/previous_experiment" \
    --resume_epoch 25
```

## Technical Details

### Checkpoint Discovery

The system automatically discovers available checkpoints by:
1. Scanning the experiment directory for `checkpoint_epoch_*` folders
2. Extracting epoch numbers from folder names
3. Verifying the existence of corresponding `.pt` files
4. Sorting checkpoints by epoch number

### State Recovery

When resuming training:
1. **Model States**: Both generator and critic models are restored to their saved states
2. **Optimizer States**: Learning rates, momentum, and other optimizer parameters are preserved
3. **Training Progress**: Training resumes from the next epoch after the loaded checkpoint
4. **Multi-GPU Sync**: In distributed training, the start epoch is broadcast to all processes

### Error Handling

The system follows a fail-fast approach:
- Validates experiment directory existence
- Checks for available checkpoints
- Program exits immediately if checkpoint files are missing or corrupted
- Program exits immediately if checkpoint folder names are invalid
- Provides detailed logging before exit for debugging

## Best Practices

### 1. Experiment Management

- Use descriptive experiment names that include key parameters
- Maintain consistent directory structure across experiments
- Keep original config files alongside checkpoints

### 2. Checkpoint Strategy

- Save checkpoints regularly (default: every epoch)
- Monitor disk space for large models
- Consider cleanup strategies for old checkpoints

### 3. Parameter Updates

When resuming with modified parameters:
- Learning rate changes are generally safe
- Batch size changes may affect convergence
- Architecture changes are not supported

### 4. Distributed Training

- Ensure all processes can access the checkpoint directory
- Use shared storage for multi-node setups
- Verify checkpoint loading succeeds on all processes

## Troubleshooting

### Common Issues

1. **No Checkpoints Found**
   - Verify the experiment directory path
   - Check that checkpoint folders follow the `checkpoint_epoch_XXX` naming convention
   - Ensure `.pt` files exist within checkpoint folders

2. **Checkpoint Loading Fails**
   - Check for model architecture mismatches
   - Verify PyTorch version compatibility
   - Review error logs for specific loading issues
   - Program will exit immediately on any loading error

3. **Performance Issues**
   - Loading large checkpoints may take time
   - Consider checkpoint compression for storage optimization
   - Monitor memory usage during checkpoint loading

### Debug Tips

- Enable detailed logging with appropriate verbosity levels
- Test checkpoint loading before long training runs
- Validate training progress after resuming

## Integration with Existing Workflows

The resume functionality integrates seamlessly with:
- Wandb logging (maintains run continuity)
- SLURM job scheduling (handle job preemption)
- Automated training pipelines
- Experiment tracking systems

## Examples for Your Use Case

Based on your example experiment directory `causvid_distillation_realLora32-fakeLora32_lr1e-5_ep100_bs1_ratio5_sketch_captions_slurm333`, here are specific usage examples:

```bash
# Resume from this specific experiment
python train_distillation_finetune.py \
    --config_path "/home/coder/code/video_sketch/libs/CausVid/configs/wan_causal_dmd_finetune.yaml" \
    --data_path "/home/coder/code/video_sketch/data/caption_gen/sketch_captions_quickdraw.txt" \
    --resume_from "/path/to/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep100_bs1_ratio5_sketch_captions_slurm333" \
    --num_epochs 200

# Or using the convenience script
bash resume_training_example.sh /path/to/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep100_bs1_ratio5_sketch_captions_slurm333 --num_epochs 200
```

This implementation provides a robust, efficient, and user-friendly way to resume training from checkpoints while maintaining all the benefits of the original training pipeline. 
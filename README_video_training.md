# Video Regression Training for CausVid

This extension to CausVid enables training on video datasets directly, using VAE-encoded video latents as supervision targets instead of ODE trajectories.

## Overview

The video regression training approach:
1. **Loads video files** and corresponding text prompts from a CSV metadata file
2. **Encodes videos** to latent space using the VAE
3. **Trains the generator** to predict clean latents from noisy inputs at random timesteps
4. **Reuses CausVid's infrastructure** for distributed training, logging, and model management

## Key Components

### 1. VideoRegression (`causvid/video_regression.py`)
- Inherits from `ODERegression` 
- Implements `video_loss()` method that encodes videos to latents and computes regression loss
- Maintains compatibility with CausVid's training infrastructure

### 2. VideoRegressionDataset (`causvid/video_data.py`)
- Loads video files and metadata from CSV
- Handles video preprocessing (resizing, normalization, frame sampling)
- Supports temporal division constraints for model compatibility

### 3. VideoTrainer (`causvid/train_video.py`)
- Reuses the Trainer architecture from `train_ode.py`
- Integrates video dataset and regression loss
- Maintains distributed training and logging capabilities

## Usage

### 1. Prepare Your Dataset

Create a CSV metadata file with `video` and `prompt` columns:
```csv
video,prompt
bird.mp4,"Draw a sketch of bird step by step..."
car.mp4,"Draw a sketch of car step by step..."
```

Place video files in a directory structure like:
```
data/custom_sketch0618/
├── metadata_detailed.csv
└── trunc_compress81_sample/
    ├── bird.mp4
    ├── car.mp4
    └── ...
```

### 2. Configure Training

Edit `configs/wan_video_regression.yaml`:
```yaml
# Data paths
dataset_base_path: "data/custom_sketch0618/trunc_compress81_sample"
metadata_path: "data/custom_sketch0618/metadata_detailed.csv"

# Video settings
num_frames: 81
height: 480
width: 832

# Training settings
batch_size: 1
lr: 1e-4
max_steps: 10000
```

### 3. Run Training

#### Single GPU:
```bash
cd libs/CausVid
python causvid/train_video.py --config_path configs/wan_video_regression.yaml
```

#### Multi-GPU:
```bash
cd libs/CausVid
NPROC_PER_NODE=4 ./run_video_training.sh
```

#### Custom configuration:
```bash
CONFIG_PATH="path/to/your/config.yaml" NPROC_PER_NODE=2 ./run_video_training.sh
```

## Configuration Options

### Data Configuration
- `dataset_base_path`: Directory containing video files
- `metadata_path`: Path to CSV metadata file
- `num_frames`: Number of frames to load from each video
- `height`, `width`: Target video resolution
- `time_division_factor`: Temporal division constraint (default: 4)

### Training Configuration
- `batch_size`: Training batch size (start with 1 for video training)
- `lr`: Learning rate (recommended: 1e-4)
- `max_steps`: Total training steps
- `save_interval`: Steps between model saves
- `generator_task`: "causal_video" or "bidirectional_video"

### Model Configuration
- `generator_ckpt`: Path to pretrained generator (optional)
- `denoising_step_list`: Timesteps for regression training
- `num_frame_per_block`: Frames per block for causal generation

## Key Differences from ODE Training

| Aspect | ODE Training | Video Training |
|--------|-------------|----------------|
| **Input Data** | Precomputed ODE trajectories | Raw video files + prompts |
| **Target** | Clean latent from trajectory end | VAE-encoded video latents |
| **Loss Function** | `generator_loss()` from ODERegression | `video_loss()` from VideoRegression |
| **Data Processing** | Load trajectory tensors | Load videos → VAE encode → latents |
| **Memory Usage** | Lower (precomputed latents) | Higher (video loading + VAE encoding) |

## Architecture Benefits

### Reuses CausVid Infrastructure
- ✅ Distributed training with FSDP
- ✅ Gradient checkpointing
- ✅ Mixed precision training
- ✅ WandB logging and metrics
- ✅ Model saving and loading

### Maintains Compatibility
- ✅ Same model architectures (WAN, etc.)
- ✅ Same training configurations
- ✅ Same evaluation pipelines
- ✅ Easy switching between ODE and video training

## Memory and Performance Tips

### For Large Videos:
- Start with `batch_size: 1`
- Enable `gradient_checkpointing: true`
- Use `mixed_precision: true`
- Consider reducing `num_frames` if memory is limited

### For Better Performance:
- Preprocess videos to target resolution
- Use SSD storage for video files
- Increase `num_workers` in dataloader if I/O bound
- Monitor GPU utilization and adjust batch size accordingly

## Troubleshooting

### Common Issues:

1. **Out of Memory:**
   - Reduce `batch_size` to 1
   - Reduce `num_frames`
   - Enable gradient checkpointing
   - Use smaller video resolution

2. **Video Loading Errors:**
   - Check video file paths in metadata CSV
   - Ensure video formats are supported (mp4, avi, mov, etc.)
   - Verify video files are not corrupted

3. **Training Instability:**
   - Lower learning rate (try 5e-5)
   - Adjust `denoising_step_list` 
   - Check gradient norms in logs

### Monitoring Training:

Key metrics to watch in WandB:
- `generator_loss`: Main training loss
- `target_latent_mean/std`: Statistics of encoded video latents
- `pred_latent_mean/std`: Statistics of model predictions
- `loss_at_time_*`: Loss breakdown by timestep ranges

## Extension Points

This implementation can be easily extended for:
- **Different video encoders**: Replace VAE with other encoders
- **Multi-modal inputs**: Add image conditioning, control signals
- **Different loss functions**: L1, perceptual, adversarial losses
- **Data augmentation**: Temporal crops, color jittering, etc.
- **Progressive training**: Start with fewer frames, gradually increase

The modular design allows easy customization while maintaining the robust CausVid training infrastructure. 
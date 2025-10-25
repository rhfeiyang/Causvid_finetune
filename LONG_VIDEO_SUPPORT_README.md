# Enhanced Features in upload_checkpoint_results_to_wandb.py

This document describes the enhancements made to support:
1. Long video inference results with the `seed0_long3-3` directory structure pattern
2. Negative epoch checkpoints with the `checkpoint_epoch_-001` directory structure pattern

## Overview

The upload script now automatically detects and processes:

1. **Long video inference results** that use autoregressive generation with multiple rollouts and frame overlaps. These results are generated using the `--num_rollout` and `--num_overlap_frames` parameters in the comprehensive inference script.

2. **Negative epoch checkpoints** that may be generated during training processes (e.g., during warmup phases or when using negative step numbering).

## Directory Structure Support

### Supported Directory Patterns
```
experiment_path/
├── checkpoint_epoch_-001/    # Negative epoch support
│   ├── checkpoint_epoch_-001_inference/
│   │   └── validation_prompts_detailed/
│   │       └── seed0_long3-3/
│   │           ├── batch_summary.txt
│   │           ├── bird_oval_shape/
│   │           │   ├── bird_oval_shape.mp4
│   │           │   ├── bird_oval_shape_config.json
│   │           │   └── bird_oval_shape_trajectory.png
│   │           └── open_book_vertical/
│   │               └── ...
│   └── model_epoch_-001.pt
├── checkpoint_epoch_001/     # Positive epoch (existing)
│   ├── checkpoint_epoch_001_inference/
│   │   └── validation_prompts_detailed/
│   │       └── seed0_long3-3/
│   │           ├── batch_summary.txt
│   │           ├── bird_oval_shape/
│   │           │   ├── bird_oval_shape.mp4
│   │           │   ├── bird_oval_shape_config.json
│   │           │   └── bird_oval_shape_trajectory.png
│   │           └── open_book_vertical/
│   │               └── ...
│   └── model_epoch_001.pt
├── checkpoint_epoch_003/
└── ...
```

### Pattern Explanation
- `checkpoint_epoch_-001`: Negative epoch -1 (e.g., during warmup phase)
- `seed0_long3-3` means:
  - `seed0`: Random seed 0
  - `long3-3`: 3 rollouts with 3 overlap frames
  - This generates longer videos through autoregressive generation

## Implementation Details

### 1. Enhanced Epoch Detection

The epoch extraction functions now support negative epochs using updated regex patterns:

```python
# Old pattern (positive only): r'checkpoint_epoch_(\d+)'
# New pattern (positive and negative): r'checkpoint_epoch_(-?\d+)'
# Supports: checkpoint_epoch_001, checkpoint_epoch_-001
```

### 2. Enhanced Directory Discovery

The `find_seed_dirs()` function now uses recursive search to find seed directories even when nested under intermediate directories like `validation_prompts_detailed/`:

```python
# Recursive search finds: validation_prompts_detailed/seed0_long3-3/
# Even when structure has multiple levels of nesting
```

### 3. Enhanced Parameter Extraction

The `find_video_files()` function now extracts long video parameters from directory names:

```python
# Extracts: seed=0, rollout=3, overlap=3, is_long=True
dir_name = "seed0_long3-3"
long_match = re.search(r'_long(\d+)-(\d+)', dir_name)
```

### 4. Enhanced Run Naming

WandB run names now include long video information:

**Short video:** `experiment_epoch5_prompt01_cat_draw_seed42_steps50`
**Long video:** `experiment_epoch5_prompt01_cat_draw_seed42_steps50_long3-3`

### 5. Enhanced Tagging

Long videos get additional tags for easy filtering:
- `long_video` (vs `short_video`)
- `rollout:3`
- `overlap:3`

### 6. Enhanced Configuration

WandB config includes long video parameters:
```python
config = {
    # ... existing parameters ...
    "is_long_video": True,
    "num_rollout": 3,
    "num_overlap_frames": 3,
}
```

## Usage Examples

### Upload all videos (including long videos)
```bash
python upload_checkpoint_results_to_wandb.py /path/to/experiment
```

### Upload specific epochs with long videos
```bash
python upload_checkpoint_results_to_wandb.py /path/to/experiment --epochs 5 10 15
```

### Filter results in WandB

You can now filter long vs short videos using tags:
- `long_video`: All long videos
- `short_video`: All short videos  
- `rollout:3`: Videos with 3 rollouts
- `overlap:3`: Videos with 3 overlap frames

## Backward Compatibility

All existing functionality remains unchanged:
- Short videos (single rollout) work exactly as before
- Old directory structures are still supported
- Existing run naming and configuration structure is preserved

## Benefits

1. **Easy Identification**: Long videos are clearly marked in WandB
2. **Parameter Tracking**: Rollout and overlap parameters are tracked
3. **Filtering**: Easy to compare short vs long video results
4. **Analysis**: Can analyze the effect of different rollout/overlap settings

## Configuration Parameters

The following parameters are now tracked for long videos:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `is_long_video` | Whether this is a long video | `True`/`False` |
| `num_rollout` | Number of autoregressive rollouts | `3` |
| `num_overlap_frames` | Frames overlapping between rollouts | `3` |

These parameters are extracted both from:
1. Directory names (primary source)
2. Config files (fallback)

## Debug Information

The script now logs when long video parameters are detected:
```
INFO - Detected long video parameters from directory checkpoint_epoch_003_inference_seed0_long3-3: rollout=3, overlap=3
``` 
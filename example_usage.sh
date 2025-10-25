#!/bin/bash

# Example Usage of Enhanced CausVid Video Training
# This script demonstrates different ways to run the enhanced training

echo "======================================================"
echo "CausVid Enhanced Training Examples"
echo "======================================================"

# Set up common paths (adjust these for your setup)
EXAMPLE_DATASET="/home/coder/code/video_sketch/data/custom_sketch0618/trunc_compress81_sample"
EXAMPLE_METADATA="/home/coder/code/video_sketch/data/custom_sketch0618/metadata_detailed.csv"

echo "Available examples:"
echo "1. Basic training (default settings)"
echo "2. Quick debug run"
echo "3. Custom parameters"
echo "4. Small resolution test"
echo "5. Multi-epoch training"
echo ""

read -p "Select example (1-5) or 'q' to quit: " choice

case $choice in
    1)
        echo "Running basic training with default settings..."
        ./train_video_enhanced.sh
        ;;
    2)
        echo "Running quick debug test (no WandB, small settings)..."
        echo "This will show tqdm progress bars in action!"
        DEBUG=true \
        NO_WANDB=true \
        NUM_FRAMES=9 \
        HEIGHT=128 \
        WIDTH=128 \
        NUM_EPOCHS=2 \
        ./train_video_enhanced.sh
        ;;
    3)
        echo "Running with custom parameters..."
        ./train_video_enhanced.sh \
            --dataset_base_path "$EXAMPLE_DATASET" \
            --metadata_path "$EXAMPLE_METADATA" \
            --lr 1e-5 \
            --batch_size 1 \
            --num_epochs 5 \
            --wandb_project "causvid_custom_test"
        ;;
    4)
        echo "Running small resolution test for memory-constrained systems..."
        HEIGHT=320 \
        WIDTH=512 \
        NUM_FRAMES=33 \
        BATCH_SIZE=1 \
        NUM_EPOCHS=2 \
        ./train_video_enhanced.sh
        ;;
    5)
        echo "Running longer training with checkpointing..."
        NUM_EPOCHS=20 \
        LR=5e-6 \
        ./train_video_enhanced.sh \
            --dataset_base_path "$EXAMPLE_DATASET" \
            --metadata_path "$EXAMPLE_METADATA"
        ;;
    q|Q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please select 1-5 or 'q'."
        exit 1
        ;;
esac

echo "======================================================"
echo "Example completed!"
echo "Check experiments/ directory for results."
echo "======================================================" 
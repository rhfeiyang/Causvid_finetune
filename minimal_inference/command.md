set prompt_file_path  "/home/coder/code/video_sketch/data/lion/captions.txt"
set prompt_file_path "/home/coder/code/video_sketch/data/custom_sketch0618/validation_prompts_detailed.txt"
# Autoregressive 3-step 5-second Video Generation


python minimal_inference/autoregressive_inference.py --config_path configs/wan_causal_dmd.yaml --output_folder experiments/ar_5s  --prompt_file_path $prompt_file_path

# Autoregressive 3-step long Video Generation

python minimal_inference/longvideo_autoregressive_inference.py --config_path configs/wan_causal_dmd.yaml --output_folder experiments/ar_long --prompt_file_path $prompt_file_path
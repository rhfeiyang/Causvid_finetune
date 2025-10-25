import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="/home/coder/code/video_sketch/libs/CausVid/configs/wan_causal_ode_finetune.yaml")
parser.add_argument("--checkpoint_folder", type=str, default="/home/coder/code/video_sketch/libs/CausVid/pretrained/tianweiy/CausVid/autoregressive_checkpoint")
parser.add_argument("--output_folder", type=str, default="experiments/ar_5s")
parser.add_argument("--prompt_file_path", type=str, default="/home/coder/code/video_sketch/data/lion/captions.txt")

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

dataset = TextDataset(args.prompt_file_path)

sampled_noise = torch.randn(
    [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)

os.makedirs(args.output_folder, exist_ok=True)

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{prompts[0][:50]}.mp4"), fps=16)

from causvid.models.wan.wan_wrapper import WanVAEWrapper
from causvid.util import launch_distributed_job
import torch.distributed as dist
import imageio.v3 as iio
from tqdm import tqdm
import argparse
import torch
import json
import math
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

torch.set_grad_enabled(False)


def video_to_numpy(video_path):
    """
    Reads a video file and returns a NumPy array containing all frames.

    :param video_path: Path to the video file.
    :return: NumPy array of shape (num_frames, height, width, channels)
    """
    return iio.imread(video_path, plugin="pyav")  # Reads the entire video as a NumPy array


def load_video_info(info_path):
    """
    Load video information from either JSON or CSV format.
    
    :param info_path: Path to the info file (JSON or CSV)
    :return: Dictionary with video paths as keys and prompts as values
    """
    if info_path.lower().endswith('.csv'):
        # Load CSV format
        df = pd.read_csv(info_path)
        # Assume columns are 'video' and 'prompt'
        if 'video' not in df.columns or 'prompt' not in df.columns:
            raise ValueError("CSV file must contain 'video' and 'prompt' columns")
        return dict(zip(df['video'], df['prompt']))
    else:
        # Load JSON format (default behavior)
        with open(info_path, "r") as f:
            return json.load(f)


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def preprocess_video_frames(video_array, height, width):
    """
    Preprocess video frames by resizing to target dimensions.
    
    :param video_array: NumPy array of shape (num_frames, height, width, channels)
    :param height: Target height
    :param width: Target width
    :return: Preprocessed video tensor of shape (channels, num_frames, height, width)
    """
    # Create transform for resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    frames = []
    for frame in video_array:
        # Convert numpy array to PIL Image
        frame_pil = Image.fromarray(frame).convert('RGB')
        # Apply transforms
        frame_tensor = transform(frame_pil)
        frames.append(frame_tensor)
    
    # Stack frames: [T, C, H, W]
    video_tensor = torch.stack(frames, dim=0)
    
    # Rearrange to [C, T, H, W] to match VAE encode expectation
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    
    return video_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_folder", type=str,
                        help="Path to the folder containing input videos.")
    parser.add_argument("--output_latent_folder", type=str,
                        help="Path to the folder where output latents will be saved.")
    parser.add_argument("--info_path", type=str,
                        help="Path to the info file containing video metadata (JSON or CSV format). "
                             "For CSV, must have 'video' and 'prompt' columns.")
    parser.add_argument("--height", type=int, default=480,
                        help="Target height for video frames (default: 480)")
    parser.add_argument("--width", type=int, default=832,
                        help="Target width for video frames (default: 832)")

    args = parser.parse_args()

    # Step 1: Setup the environment
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Step 2: Create the generator
    launch_distributed_job()
    device = torch.cuda.current_device()

    video_info = load_video_info(args.info_path)

    model = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    video_paths = sorted(list(video_info.keys()))

    os.makedirs(args.output_latent_folder, exist_ok=True)

    for index in tqdm(range(int(math.ceil(len(video_paths) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        global_index = index * dist.get_world_size() + dist.get_rank()
        if global_index >= len(video_paths):
            break

        video_path = video_paths[global_index]
        video_file_name = video_path.split('/')[-1].split('.')[0]
        prompt = video_info[video_path]

        # Load video frames
        array = video_to_numpy(os.path.join(args.input_video_folder, video_path))
        if array is None:
            print(f"Failed to read video: {video_path}")
            continue

        # Preprocess video with resize
        video_tensor = preprocess_video_frames(array, args.height, args.width)
        video_tensor = video_tensor.to(device=device, dtype=torch.bfloat16)
        
        encoded_latents = encode(model, [video_tensor]).transpose(2, 1)

        torch.save(
            {prompt: encoded_latents.cpu().detach()},
            os.path.join(args.output_latent_folder, f"{video_file_name}.pt")
        )
    dist.barrier()


if __name__ == "__main__":
    main()

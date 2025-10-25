from causvid.ode_regression import ODERegression
from causvid.models import get_diffusion_wrapper, get_text_encoder_wrapper, get_vae_wrapper
import torch.nn.functional as F
from typing import Tuple
from torch import nn
import torch


class VideoRegression(ODERegression):
    def __init__(self, args, device):
        """
        Initialize the VideoRegression module.
        This class inherits from ODERegression but processes video data instead of ODE trajectories.
        The video is encoded to latent space using VAE and used as supervision target.
        """
        super().__init__(args, device)
        print("VideoRegression initialized, inheriting from ODERegression")
 
    def video_loss(self, video_tensor: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate predictions from noisy latents and compute regression loss against clean video latents.
        Input:
            - video_tensor: a tensor containing video frames [batch_size, num_frames, num_channels, height, width].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss breakdown.
        """
        # Step 1: Encode video to latent space using VAE
        batch_size, num_frames, num_channels, height, width = video_tensor.shape
        
        with torch.no_grad():
            # Convert video tensor to the format expected by WanVAE.encode
            # WanVAE.encode expects a list of videos, each with shape [C, T, H, W]
            videos_list = []
            for i in range(batch_size):
                # video_tensor[i] has shape [F, C, H, W], need to transpose to [C, F, H, W]
                video_ctfw = video_tensor[i].permute(1, 0, 2, 3)  # [C, T, H, W]
                videos_list.append(video_ctfw)
            
            # Encode all videos to latents
            latent_list = self.vae.encode(videos_list)
            
            # Convert back to tensor format: [B, F, C_latent, H_latent, W_latent]
            # Each latent in latent_list has shape [C_latent, T, H_latent, W_latent]
            target_latents = []
            for latent_ctfw in latent_list:
                # Transpose back to [T, C_latent, H_latent, W_latent]
                latent_tchw = latent_ctfw.permute(1, 0, 2, 3)
                target_latents.append(latent_tchw)
            
            # Stack to get [B, T, C_latent, H_latent, W_latent]
            target_latent = torch.stack(target_latents, dim=0)
            target_latent = target_latent.to(device=self.device, dtype=self.dtype)

        num_frames_latent = target_latent.shape[1]
        # Step 2: Prepare generator input by adding noise
        # Randomly sample timestep for each frame
        timestep = torch.randint(
            0, len(self.denoising_step_list), 
            [batch_size, num_frames_latent],
            device=self.device, 
            dtype=torch.long
        )
        
        # Process timestep according to task type
        timestep = self._process_timestep(timestep)
        
        # Get actual timestep values
        actual_timestep = self.denoising_step_list[timestep]
        
        # Add noise to clean latents
        noise = torch.randn_like(target_latent)
        noisy_input = self.scheduler.add_noise(
            target_latent.flatten(0, 1),
            noise.flatten(0, 1),
            actual_timestep.flatten(0, 1)
        ).detach().unflatten(0, (batch_size, num_frames_latent)).type_as(target_latent)

        # Step 3: Run generator on noisy input
        pred_latent = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=actual_timestep
        )

        # Step 4: Compute regression loss (only on non-zero timesteps)
        mask = actual_timestep != 0
        loss = F.mse_loss(
            pred_latent[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_latent, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": actual_timestep.float().mean(dim=1).detach(),
            "target_latent_mean": target_latent.mean().detach(),
            "target_latent_std": target_latent.std().detach(),
            "pred_latent_mean": pred_latent.mean().detach(),
            "pred_latent_std": pred_latent.std().detach(),
        }

        return loss, log_dict 
from causvid.video_data import VideoRegressionDataset
from causvid.video_regression import VideoRegression
from causvid.models import get_block_class
from collections import defaultdict
from causvid.util import set_seed, init_logging_folder
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from omegaconf import OmegaConf
import argparse
import torch
import time
import os
import sys
import logging
from tqdm import tqdm

# Add paths for experiment utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from experiment_utils import setup_experiment


def generate_causvid_exp_name(args):
    """
    Generate experiment name based on CausVid training parameters.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        list: List of name components (excluding timestamp)
    """
    components = [args.experiment_name]
    
    # Model info
    if hasattr(args, 'model_name') and args.model_name:
        components.append(args.model_name)
    
    # Task info
    if hasattr(args, 'generator_task'):
        components.append(args.generator_task)
    
    # Training info
    if hasattr(args, 'lr'):
        lr_str = f"lr{args.lr:.0e}".replace('-0', '-')  # e.g., lr2e-6
        components.append(lr_str)
    if hasattr(args, 'num_epochs'):
        components.append(f"ep{args.num_epochs}")
    
    # Data info
    if hasattr(args, 'num_frames') and args.num_frames > 0:
        components.append(f"f{args.num_frames}")
    if hasattr(args, 'height') and hasattr(args, 'width') and args.height and args.width:
        components.append(f"{args.height}x{args.width}")
    
    # Dataset info (use basename of dataset path)
    if hasattr(args, 'dataset_base_path') and args.dataset_base_path:
        dataset_name = os.path.basename(os.path.dirname(args.dataset_base_path.rstrip('/')))
        dataset_name_sub = os.path.basename(args.dataset_base_path.rstrip('/'))
        dataset_name = f"{dataset_name}_{dataset_name_sub}" if dataset_name != dataset_name_sub else dataset_name
        # Shorten common dataset names
        if len(dataset_name) > 20:
            dataset_name = dataset_name[:20]
        components.append(dataset_name)

    if hasattr(args, 'metadata_path') and args.metadata_path:
        metadata_name = os.path.basename(args.metadata_path)
        metadata_name = os.path.splitext(metadata_name)[0]
        components.append(metadata_name)
    
    return components


class VideoTrainer:
    def __init__(self, config, exp_dir=None, wandb_run=None):
        self.config = config
        self.exp_dir = exp_dir
        self.wandb_run = wandb_run

        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision="bf16" if config.mixed_precision else "no",
            gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
            log_with="wandb" if not getattr(config, "no_wandb", False) and wandb_run is not None else None,
            project_dir=self.exp_dir
        )

        self.device = self.accelerator.device
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.is_main_process = self.accelerator.is_main_process

        # Set seed
        accelerate_set_seed(config.seed)

        # Initialize logging
        if self.is_main_process:
            output_folder = self.exp_dir
            os.makedirs(output_folder, exist_ok=True)
            
            # Initialize wandb if available and not disabled
            if not getattr(config, "no_wandb", False) and wandb_run is not None:
                self.accelerator.init_trackers(
                    project_name=getattr(config, "wandb_project", "causvid_video_regression"),
                    config=dict(config),
                    init_kwargs={"wandb": {"name": getattr(config, "wandb_name", "sketch_finetune")}}
                )

        # Step 2: Initialize the model
        self.regression_model = VideoRegression(config, device=self.device)

        # Step 3: Initialize the dataloader
        dataset = VideoRegressionDataset(
            dataset_base_path=config.dataset_base_path,
            metadata_path=config.metadata_path,
            num_frames=getattr(config, "num_frames", 81),
            height=getattr(config, "height", 480),
            width=getattr(config, "width", 832),
            time_division_factor=getattr(config, "time_division_factor", 4),
            time_division_remainder=getattr(config, "time_division_remainder", 1),
            repeat=getattr(config, "dataset_repeat", 1)
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=getattr(config, "num_workers", 4),
            drop_last=True
        )

        # Step 4: Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            [param for param in self.regression_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999))
        )

        # Step 5: Prepare everything with Accelerator
        (
            self.regression_model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.regression_model,
            self.optimizer,
            dataloader
        )

        # Keep VAE on CPU/original device (not trained)
        # self.regression_model.vae remains unprepared

        self.step = 0
        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.previous_time = None

        if self.is_main_process:
            dataset_repeat = getattr(config, "dataset_repeat", 1)
            original_size = len(dataset) // dataset_repeat
            logging.info(f"Initialized VideoTrainer with {original_size} videos (repeated {dataset_repeat}x = {len(dataset)} samples)")
            logging.info(f"Model: {config.model_name}")
            logging.info(f"Batch size: {config.batch_size}")
            logging.info(f"Learning rate: {config.lr}")
            logging.info(f"Device: {self.device}")
            logging.info(f"Mixed precision: {config.mixed_precision}")
            logging.info(f"Experiment directory: {self.exp_dir}")

    def save(self, epoch=None):
        if self.is_main_process:
            logging.info("Saving model checkpoint...")
            # Unwrap model for saving
            unwrapped_generator = self.accelerator.unwrap_model(self.regression_model.generator)
            
            # Use experiment directory if available
            output_folder = self.exp_dir
            
            # Save with epoch info only
            if epoch is not None:
                checkpoint_dir = os.path.join(output_folder, f"checkpoint_epoch_{epoch:03d}")
                filename = f"model_epoch_{epoch:03d}.pt"
            else:
                checkpoint_dir = os.path.join(output_folder, "checkpoint_final")
                filename = "model.pt"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save state dict
            state_dict = {
                "generator": unwrapped_generator.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "config": dict(self.config)
            }
            torch.save(state_dict, os.path.join(checkpoint_dir, filename))
            logging.info(f"Model saved to {os.path.join(checkpoint_dir, filename)}")

    def train_one_step(self, batch):
        """Train on a single batch"""
        self.regression_model.generator.eval()  # prevent any randomness (e.g. dropout)
        self.regression_model.text_encoder.eval()

        text_prompts = batch["prompts"]
        video_tensor = batch["video_tensor"]

        # Ensure video_tensor is in correct format: [B, T, C, H, W]
        if video_tensor.dim() == 5:
            pass  # Already correct [B, T, C, H, W]
        elif video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(1)  # Add temporal dimension
        else:
            raise ValueError(f"Unexpected video tensor shape: {video_tensor.shape}")

        video_tensor = video_tensor.to(device=self.device, dtype=self.dtype)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.regression_model.text_encoder(
                text_prompts=text_prompts)

        # Step 3: Train the generator using video regression loss
        with self.accelerator.accumulate(self.regression_model.generator):
            generator_loss, log_dict = self.regression_model.video_loss(
                video_tensor=video_tensor,
                conditional_dict=conditional_dict
            )

            # Backpropagation
            self.accelerator.backward(generator_loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.regression_model.generator.parameters(), 
                    self.max_grad_norm
                )
            
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Store loss for progress bar display
        loss_value = generator_loss.item()
        self._last_loss = loss_value

        # Step 4: Logging
        if self.is_main_process:
            log_data = {
                "generator_loss": loss_value,
                "target_latent_mean": log_dict["target_latent_mean"].item(),
                "target_latent_std": log_dict["target_latent_std"].item(),
                "pred_latent_mean": log_dict["pred_latent_mean"].item(),
                "pred_latent_std": log_dict["pred_latent_std"].item(),
                "step": self.step,
                "epoch": self.step // len(self.dataloader)
            }
            
            # Add timestep breakdown
            unnormalized_loss = log_dict["unnormalized_loss"]
            timestep = log_dict["timestep"]
            
            loss_breakdown = defaultdict(list)
            for index, t in enumerate(timestep):
                loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                    unnormalized_loss[index].item())

            for key_t in loss_breakdown.keys():
                log_data[f"loss_at_time_{key_t}"] = sum(loss_breakdown[key_t]) / len(loss_breakdown[key_t])

            # Log to wandb if available
            if not getattr(self.config, "no_wandb", False):
                self.accelerator.log(log_data, step=self.step)
                
                # Also log to external wandb if provided
                if self.wandb_run is not None:
                    self.wandb_run.log(log_data, step=self.step)
            
            # Reduced logging frequency for cleaner output with tqdm
            if self.step % 50 == 0:
                logging.info(f"Step {self.step}: Loss = {loss_value:.6f}")
        
        return loss_value

    def train(self):
        num_epochs = self.config["num_epochs"]
        global_step = 0
        
        if self.is_main_process:
            logging.info(f"Starting training for {num_epochs} epochs")
            
        # Create overall training progress bar for main process only
        if self.is_main_process:
            total_steps = num_epochs * len(self.dataloader)
            overall_pbar = tqdm(
                total=total_steps, 
                desc="Training Progress", 
                position=0,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
        for epoch in range(num_epochs):
            if self.is_main_process:
                logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
                
                # Create epoch progress bar for main process only
                epoch_pbar = tqdm(
                    self.dataloader, 
                    desc=f"Epoch {epoch+1}/{num_epochs}",
                    position=1,
                    leave=False,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
                )
                dataloader_iter = epoch_pbar
            else:
                dataloader_iter = self.dataloader
                
            epoch_losses = []
            epoch_start_time = time.time()
            
            for step, batch in enumerate(dataloader_iter):
                self.step = global_step
                step_start_time = time.time()
                
                with self.accelerator.autocast():
                    loss_value = self.train_one_step(batch)
                
                step_end_time = time.time()
                step_time = step_end_time - step_start_time
                
                if self.is_main_process:
                    # Track epoch losses
                    epoch_losses.append(loss_value)
                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    
                    # Update epoch progress bar with metrics
                    epoch_pbar.set_postfix({
                        'Loss': f'{loss_value:.6f}',
                        'Avg': f'{avg_loss:.6f}',
                        'Time': f'{step_time:.2f}s'
                    })
                    
                    # Update overall progress bar
                    overall_pbar.update(1)
                    overall_pbar.set_postfix({
                        'Epoch': f'{epoch+1}/{num_epochs}',
                        'Loss': f'{loss_value:.6f}',
                        'Step': f'{step+1}/{len(self.dataloader)}'
                    })
                    
                    # Log timing
                    current_time = time.time()
                    if self.previous_time is None:
                        self.previous_time = current_time
                    else:
                        time_per_step = current_time - self.previous_time
                        log_data = {"time_per_step": time_per_step}
                        
                        if not getattr(self.config, "no_wandb", False):
                            self.accelerator.log(log_data, step=self.step)
                            
                            if self.wandb_run is not None:
                                self.wandb_run.log(log_data, step=self.step)
                        self.previous_time = current_time
                        
                global_step += 1
                
            # Close epoch progress bar and log summary
            if self.is_main_process:
                epoch_pbar.close()
                
                # Log epoch summary
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                
                tqdm.write(f"✅ Epoch {epoch+1} completed in {epoch_duration:.2f}s | Avg Loss: {avg_epoch_loss:.6f}")
                logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s, Average Loss: {avg_epoch_loss:.6f}")
                
            # Save checkpoint at the end of each epoch
            self.save(epoch=epoch)
            
        # Close overall progress bar and log completion
        if self.is_main_process:
            overall_pbar.close()
            tqdm.write("🎉 Training completed successfully!")
            logging.info("Training completed successfully!")
            
        # End tracking
        if not getattr(self.config, "no_wandb", False):
            self.accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="CausVid Video Regression Training")
    parser.add_argument("--config_path", type=str, default="/home/coder/code/video_sketch/libs/CausVid/configs/wan_causal_ode_finetune.yaml",
                        help="Path to configuration file")
    parser.add_argument("--no_save", action="store_true", help="Disable model saving")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with reduced dataset size")
    
    # Experiment management
    parser.add_argument("--experiment_dir", type=str, default="experiments",
                        help="Base directory for experiments")
    parser.add_argument("--experiment_name", type=str, default="causvid_finetune")
    
    # Commonly changed parameters
    parser.add_argument("--dataset_base_path", type=str, 
                        default="/home/coder/code/video_sketch/data/custom_sketch0618/trunc_compress81_sample",
                        help="Override dataset base path")
    parser.add_argument("--metadata_path", type=str, 
                        default="/home/coder/code/video_sketch/data/custom_sketch0618/metadata_detailed.csv",
                        help="Override metadata CSV path")
    parser.add_argument("--num_frames", type=int, default=81, help="Override number of frames per video")
    parser.add_argument("--height", type=int, default=480, help="Override video height")
    parser.add_argument("--width", type=int, default=832, help="Override video width")
    parser.add_argument("--batch_size", type=int, default=1, help="Override batch size")
    parser.add_argument("--lr", type=float, default=2.0e-06, help="Override learning rate")
    parser.add_argument("--wandb_name", type=str, default="causvid_finetune", help="Override wandb run name")
    parser.add_argument("--seed", type=int, default=0, help="Override random seed")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--dataset_repeat", type=int, default=100, help="Number of times to repeat the dataset per epoch")
    parser.add_argument("--generator_ckpt", type=str,
                        default="/home/coder/code/video_sketch/libs/CausVid/pretrained/tianweiy/CausVid/autoregressive_checkpoint/model.pt")
    args = parser.parse_args()

    if args.debug:
        args.no_wandb = True

    # Load config and override with command-line arguments
    config = OmegaConf.load(args.config_path)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})

    # Setup experiment with enhanced logging and management
    # try:
    exp_dir, wandb_run = setup_experiment(
        args, 
        base_output_dir=args.experiment_dir,
        project_name=args.experiment_name if not args.no_wandb else None,
        exp_name_func=generate_causvid_exp_name
    )
    
    # If WandB is disabled, set wandb_run to None
    if args.no_wandb:
        wandb_run = None
        
    # Update config with experiment directory
    config.output_folder = exp_dir
    
    logging.info("Starting CausVid video regression training...")
    logging.info(f"Experiment directory: {exp_dir}")
    logging.info(f"Dataset: {args.dataset_base_path}")
    logging.info(f"Dataset repeat: {args.dataset_repeat}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Epochs: {args.num_epochs}")
    logging.info(f"Frames: {args.num_frames}")
    logging.info(f"Resolution: {args.height}x{args.width}")
        
    # except Exception as e:
    #     logging.warning(f"Failed to setup experiment management: {e}")
    #     logging.warning("Falling back to basic directory creation")
    #     exp_dir = args.output_folder
    #     wandb_run = None
    #     os.makedirs(exp_dir, exist_ok=True)

    # Create trainer and start training
    trainer = VideoTrainer(config, exp_dir=exp_dir, wandb_run=wandb_run)
    
    # try:
    trainer.train()
    
    # Log training completion
    logging.info("Training completed successfully")
            
    # except Exception as e:
    #     logging.error(f"Training failed: {e}")
    #     if wandb_run is not None:
    #         try:
    #             wandb_run.log({"training/failed": True, "error_message": str(e)})
    #             wandb_run.finish()
    #         except Exception as wandb_e:
    #             logging.warning(f"Failed to finish WandB run: {wandb_e}")
    #     raise


if __name__ == "__main__":
    main() 
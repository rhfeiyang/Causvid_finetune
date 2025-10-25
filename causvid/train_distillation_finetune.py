from causvid.data import ODERegressionLMDBDataset, TextDataset
from causvid.models import get_block_class
from causvid.util import set_seed, init_logging_folder, cycle
from causvid.dmd import DMD
from collections import defaultdict
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
from accelerate.utils import InitProcessGroupKwargs
# Add paths for experiment utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from experiment_utils import setup_experiment
from datetime import timedelta

# Optimize NCCL settings for better stability and performance
# os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')  # Use blocking wait to avoid timeouts
# os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # Better error handling
# os.environ.setdefault('NCCL_DEBUG', 'WARN')  # Moderate debugging level

def generate_causvid_distillation_exp_name(args):
    """
    Generate experiment name based on CausVid distillation training parameters.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        list: List of name components (excluding timestamp)
    """
    components = [args.experiment_name]
    
    # Model info
    if hasattr(args, 'model_name') and args.model_name:
        components.append(args.model_name)
    
    # Distillation info
    if hasattr(args, 'distillation_loss'):
        components.append(args.distillation_loss)
    
    # LoRA info
    lora_info = []
    if hasattr(args, 'real_score_lora_ckpt') and args.real_score_lora_ckpt:
        real_lora_rank = getattr(args, 'real_score_lora_rank', 32)
        lora_info.append(f"realLora{real_lora_rank}")
    if hasattr(args, 'fake_score_lora_ckpt') and args.fake_score_lora_ckpt:
        fake_lora_rank = getattr(args, 'fake_score_lora_rank', 32)
        lora_info.append(f"fakeLora{fake_lora_rank}")
    if lora_info:
        components.append("-".join(lora_info))
    
    # Training info
    if hasattr(args, 'lr'):
        lr_str = f"lr{args.lr:.0e}".replace('-0', '-')  # e.g., lr2e-6
        components.append(lr_str)
    if hasattr(args, 'num_epochs'):
        components.append(f"ep{args.num_epochs}")
    if hasattr(args, 'batch_size'):
        components.append(f"bs{args.batch_size}")
    
    # Update ratio info
    if hasattr(args, 'dfake_gen_update_ratio'):
        components.append(f"ratio{args.dfake_gen_update_ratio}")
    
    # Dataset repeat info
    if hasattr(args, 'dataset_repeat') and args.dataset_repeat > 1:
        components.append(f"repeat{args.dataset_repeat}")
    
    # Data info
    if hasattr(args, 'data_path') and args.data_path:
        data_name = os.path.basename(args.data_path.rstrip('/'))
        if len(data_name) > 15:
            data_name = data_name[:15]
        components.append(data_name)
    
    # Simulation type
    if hasattr(args, 'backward_simulation'):
        sim_type = "backward" if args.backward_simulation else "forward"
        components.append(sim_type)
    
    return components


class DistillationTrainer:
    def __init__(self, config, exp_dir=None, wandb_run=None):
        self.config = config
        self.exp_dir = exp_dir
        self.wandb_run = wandb_run

        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision="bf16" if config.mixed_precision else "no",
            gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
            log_with="wandb" if not getattr(config, "no_wandb", False) and wandb_run is not None else None,
            project_dir=self.exp_dir,
            kwargs_handlers=[InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=7200))],
        )

        self.device = self.accelerator.device
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.is_main_process = self.accelerator.is_main_process

        # Set seed
        accelerate_set_seed(config.seed)

        # Initialize logging
        # if self.is_main_process:
        #     output_folder = self.exp_dir
        #     os.makedirs(output_folder, exist_ok=True)
        #
        #     # Initialize wandb if available and not disabled
        #     if not getattr(config, "no_wandb", False) and wandb_run is not None:
        #         self.accelerator.init_trackers(
        #             project_name=getattr(config, "wandb_project", "causvid_distillation"),
        #             config=dict(config),
        #             init_kwargs={"wandb": {"entity": "video-sketch" ,"name": getattr(config, "wandb_name", "distillation_training")}}
        #         )

        # Step 2: Initialize the distillation model
        if config.distillation_loss == "dmd":
            self.distillation_model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distillation loss type")

        # Step 3: Initialize optimizers
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999))
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999))
        )

        # Step 4: Initialize dataloader
        self.backward_simulation = getattr(config, "backward_simulation", True)

        if self.backward_simulation:
            dataset = TextDataset(config.data_path, repeat=getattr(config, "dataset_repeat", 1))
        else:
            dataset = ODERegressionLMDBDataset(
                config.data_path, max_pair=int(1e8), repeat=getattr(config, "dataset_repeat", 1))
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=getattr(config, "num_workers", 4),
            drop_last=True
        )

        # Step 5: Prepare everything with Accelerator
        # (
        #     self.distillation_model.generator,
        #     self.distillation_model.fake_score,
        #     self.generator_optimizer,
        #     self.critic_optimizer,
        #     self.dataloader
        # ) = self.accelerator.prepare(
        #     self.distillation_model.generator,
        #     self.distillation_model.fake_score,
        #     self.generator_optimizer,
        #     self.critic_optimizer,
        #     dataloader
        # )
        #
        # # Keep other components on device but not prepared
        # self.distillation_model.real_score = self.distillation_model.real_score.to(self.device)
        # self.distillation_model.text_encoder = self.distillation_model.text_encoder.to(self.device)
        (
            self.distillation_model,
            self.generator_optimizer,
            self.critic_optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.distillation_model,
            self.generator_optimizer,
            self.critic_optimizer,
            dataloader
        )
        # if not config.no_visualize:
        #     self.distillation_model.vae = self.distillation_model.vae.to(
        #         device=self.device, dtype=self.dtype)
        self.distillation_model = self.distillation_model.to(dtype=self.dtype)
        if hasattr(self.distillation_model, 'module'):
            # Unwrap components from FSDP if needed
            self.distillation_model.vae = self.distillation_model.module.vae
            self.distillation_model.generator = self.distillation_model.module.generator
            self.distillation_model.fake_score = self.distillation_model.module.fake_score
            self.distillation_model.real_score = self.distillation_model.module.real_score.to(self.device)
            self.distillation_model.text_encoder = self.distillation_model.module.text_encoder.to(self.device)
            self.distillation_model.generator_loss = self.distillation_model.module.generator_loss
            self.distillation_model.critic_loss = self.distillation_model.module.critic_loss

        self.dataloader_cycle = cycle(self.dataloader)
        self.step = 0
        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.previous_time = None

        if self.is_main_process:
            dataset_repeat = getattr(config, "dataset_repeat", 1)
            original_size = len(dataset) // dataset_repeat
            logging.info(f"Initialized DistillationTrainer with {original_size} samples (repeated {dataset_repeat}x = {len(dataset)} samples)")
            logging.info(f"Model: {config.model_name}")
            logging.info(f"Batch size: {config.batch_size}")
            logging.info(f"Learning rate: {config.lr}")
            logging.info(f"Device: {self.device}")
            logging.info(f"Mixed precision: {config.mixed_precision}")
            logging.info(f"Backward simulation: {self.backward_simulation}")
            logging.info(f"Experiment directory: {self.exp_dir}")
            logging.info(f"Generator param count: {sum(p.numel() for p in self.distillation_model.generator.parameters())}")
            logging.info(f"Generator param requires_grad count: {sum(p.numel() for p in self.distillation_model.generator.parameters() if p.requires_grad)}")

    def save(self, step=None, epoch=None):
        if self.is_main_process:
            logging.info("Saving model checkpoint...")
            # Unwrap models for saving
            unwrapped_generator = self.accelerator.unwrap_model(self.distillation_model.generator)
            unwrapped_critic = self.accelerator.unwrap_model(self.distillation_model.fake_score)
            
            # Use experiment directory if available
            output_folder = self.exp_dir
            
            # Save with step or epoch info
            if epoch is not None:
                checkpoint_dir = os.path.join(output_folder, f"checkpoint_epoch_{epoch:03d}")
                filename = f"model_epoch_{epoch:03d}.pt"
            elif step is not None:
                checkpoint_dir = os.path.join(output_folder, f"checkpoint_step_{step:06d}")
                filename = f"model_step_{step:06d}.pt"
            else:
                checkpoint_dir = os.path.join(output_folder, "checkpoint_final")
                filename = "model.pt"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save state dict
            state_dict = {
                "generator": unwrapped_generator.state_dict(),
                "critic": unwrapped_critic.state_dict(),
                "generator_optimizer": self.generator_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
                "config": dict(self.config)
            }
            torch.save(state_dict, os.path.join(checkpoint_dir, filename))
            logging.info(f"Model saved to {os.path.join(checkpoint_dir, filename)}")

    def load_checkpoint(self, resume_from_dir, resume_epoch=None):
        """
        Load checkpoint for continue training
        
        Args:
            resume_from_dir: Directory containing the experiment checkpoints
            resume_epoch: Specific epoch to resume from (if None, will find the latest)
        
        Returns:
            start_epoch: The epoch to start training from
        """
        start_epoch = 0
        
        if self.is_main_process:
            logging.info(f"Looking for checkpoints in: {resume_from_dir}")
        
        # Find available checkpoints
        checkpoint_files = []
        if os.path.exists(resume_from_dir):
            for item in os.listdir(resume_from_dir):
                if item.startswith("checkpoint_epoch_"):
                    epoch_num = int(item.split("_")[-1])
                    checkpoint_path = os.path.join(resume_from_dir, item, f"model_epoch_{epoch_num:03d}.pt")
                    if os.path.exists(checkpoint_path):
                        checkpoint_files.append((epoch_num, checkpoint_path))
        
            if not checkpoint_files:
                logging.warning(f"No valid checkpoints found in {resume_from_dir}")
                start_epoch = 0
            else:
                # Sort by epoch number and find the target checkpoint
                checkpoint_files.sort(key=lambda x: x[0])
                
                if resume_epoch is not None:
                    # Find specific epoch
                    target_checkpoint = None
                    for epoch_num, checkpoint_path in checkpoint_files:
                        if epoch_num == resume_epoch:
                            target_checkpoint = (epoch_num, checkpoint_path)
                            break
                    if target_checkpoint is None:
                        logging.error(f"Checkpoint for epoch {resume_epoch} not found")
                        start_epoch = 0
                    else:
                        epoch_num, checkpoint_path = target_checkpoint
                        start_epoch = self._load_checkpoint_file(checkpoint_path, epoch_num)
                else:
                    # Use latest checkpoint
                    epoch_num, checkpoint_path = checkpoint_files[-1]
                    start_epoch = self._load_checkpoint_file(checkpoint_path, epoch_num)
                    print(f"Resuming from latest checkpoint at epoch {epoch_num}, path: {checkpoint_path}")
        
        # Broadcast start_epoch to all processes
        if torch.distributed.is_initialized():
            start_epoch_tensor = torch.tensor(start_epoch, device=self.device)
            torch.distributed.broadcast(start_epoch_tensor, src=0)
            start_epoch = start_epoch_tensor.item()
        
        return start_epoch
    
    def _load_checkpoint_file(self, checkpoint_path, epoch_num):
        """Helper method to load checkpoint file"""
        logging.info(f"Loading checkpoint from epoch {epoch_num}: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load generator state
        generator_state_dict = checkpoint["generator"]
        if hasattr(self.distillation_model, 'module'):
            # Handle wrapped model (FSDP/DDP)
            self.distillation_model.module.generator.load_state_dict(generator_state_dict)
        else:
            self.distillation_model.generator.load_state_dict(generator_state_dict)
        
        # Load critic state
        critic_state_dict = checkpoint["critic"]
        if hasattr(self.distillation_model, 'module'):
            self.distillation_model.module.fake_score.load_state_dict(critic_state_dict)
        else:
            self.distillation_model.fake_score.load_state_dict(critic_state_dict)
        
        # Load optimizer states
        if "generator_optimizer" in checkpoint:
            self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        logging.info(f"Successfully loaded checkpoint from epoch {epoch_num}")
        return epoch_num + 1  # Start from next epoch

    def train_one_step(self, batch=None):
        """Train on a single step"""
        self.distillation_model.generator.eval()  # prevent any randomness (e.g. dropout)
        self.distillation_model.fake_score.eval()
        self.distillation_model.real_score.eval()
        self.distillation_model.text_encoder.eval()

        TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
        VISUALIZE = self.step % self.config.log_iters == 0 and not self.config.no_visualize

        # Step 1: Get the next batch of text prompts
        if batch is None:
            # Fallback to cycle for compatibility
            if not self.backward_simulation:
                batch = next(self.dataloader_cycle)
                text_prompts = batch["prompts"]
                clean_latent = batch["ode_latent"][:, -1].to(
                    device=self.device, dtype=self.dtype)
            else:
                text_prompts = next(self.dataloader_cycle)
                clean_latent = None
        else:
            # Use provided batch (for epoch-based training)
            if not self.backward_simulation:
                text_prompts = batch["prompts"]
                clean_latent = batch["ode_latent"][:, -1].to(
                    device=self.device, dtype=self.dtype)
            else:
                text_prompts = batch
                clean_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.distillation_model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.distillation_model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        generator_loss = None
        generator_log_dict = {}
        if TRAIN_GENERATOR:
            with self.accelerator.accumulate(self.distillation_model.generator):
                generator_loss, generator_log_dict = self.distillation_model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent
                )

                # Backpropagation
                self.accelerator.backward(generator_loss)
                
                if self.accelerator.sync_gradients:
                    generator_grad_norm = self.accelerator.clip_grad_norm_(
                        self.distillation_model.generator.parameters(), 
                        self.max_grad_norm
                    )
                else:
                    generator_grad_norm = torch.tensor(0.0)
                
                self.generator_optimizer.step()
                self.generator_optimizer.zero_grad()

        # Step 4: Train the critic
        with self.accelerator.accumulate(self.distillation_model.fake_score):
            critic_loss, critic_log_dict = self.distillation_model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent
            )

            # Backpropagation
            self.accelerator.backward(critic_loss)
            
            if self.accelerator.sync_gradients:
                critic_grad_norm = self.accelerator.clip_grad_norm_(
                    self.distillation_model.fake_score.parameters(),
                    self.max_grad_norm
                )
            else:
                critic_grad_norm = torch.tensor(0.0)
            
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

        # Step 5: Logging
        if self.is_main_process:
            log_data = {
                "critic_loss": critic_loss.item(),
                "critic_grad_norm": critic_grad_norm.item() if hasattr(critic_grad_norm, 'item') else critic_grad_norm,
                "step": self.step,
                "epoch": getattr(self, 'current_epoch', 0)
            }

            if TRAIN_GENERATOR and generator_loss is not None:
                log_data.update({
                    "generator_loss": generator_loss.item(),
                    "generator_grad_norm": generator_grad_norm.item() if hasattr(generator_grad_norm, 'item') else generator_grad_norm,
                })
                if "dmdtrain_gradient_norm" in generator_log_dict:
                    log_data["dmdtrain_gradient_norm"] = generator_log_dict["dmdtrain_gradient_norm"].item()

            if VISUALIZE:
                self.add_visualization(generator_log_dict, critic_log_dict, log_data, text_prompts)

            # Log to wandb if available
            if not getattr(self.config, "no_wandb", False):
                self.accelerator.log(log_data, step=self.step)
                
                # Also log to external wandb if provided
                if self.wandb_run is not None:
                    self.wandb_run.log(log_data, step=self.step)
            
            # Reduced logging frequency for cleaner output
            if self.step % 50 == 0:
                logging.info(f"Step {self.step}: Critic Loss = {critic_loss.item():.6f}")
        
        return critic_loss.item()  # Return critic loss as main loss metric

    def add_visualization(self, generator_log_dict, critic_log_dict, log_data, text_prompts=None):
        """Add visualization to logging data"""
        from causvid.util import prepare_for_saving
        import re
        
        # Caption processing functions from upload_checkpoint_results_to_wandb.py
        def sanitize_filename(text: str, max_length: int = 50) -> str:
            """Clean text as safe filename"""
            # Remove or replace unsafe characters
            safe_text = re.sub(r'[^\w\s\-_]', '', text)
            safe_text = re.sub(r'[\s]+', '_', safe_text)
            safe_text = re.sub(r'[_\-]+', '_', safe_text)
            safe_text = safe_text.strip('_-')
            
            # Limit length
            if len(safe_text) > max_length:
                safe_text = safe_text[:max_length].rstrip('_-')
            
            # Ensure not empty
            if not safe_text:
                safe_text = "video"
            
            return safe_text

        def extract_caption_keywords(prompt: str, max_words: int = 3) -> str:
            """Extract keywords from prompt as concise caption"""
            # Remove common meaningless words
            stop_words = {'a', 'an', 'the', 'of', 'step', 'by', 'draw', 'sketch', 'process', 
                          'first', 'then', 'next', 'finally', 'and', 'or', 'for', 'with', 'in', 'on'}
            
            # Tokenize and filter
            words = re.findall(r'\b\w+\b', prompt.lower())
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Take first few keywords
            keywords = filtered_words[:max_words]
            
            if not keywords:
                # If no keywords, use original method
                return sanitize_filename(prompt, max_length=20)
            
            # Combine keywords
            caption = '_'.join(keywords)
            return sanitize_filename(caption, max_length=30)
        
        # Generate captions from text prompts if available
        captions = []
        if text_prompts and len(text_prompts) > 0:
            for prompt in text_prompts:
                if isinstance(prompt, str) and prompt.strip():
                    caption = extract_caption_keywords(prompt)
                    captions.append(caption)
                else:
                    captions.append("unknown")
        
        # Create a combined caption for batch visualization
        if captions:
            if len(captions) == 1:
                batch_caption = captions[0]
            else:
                # For multiple prompts, show first few and count
                display_captions = captions[:3]  # Show first 3
                if len(captions) > 3:
                    batch_caption = f"{', '.join(display_captions)}... ({len(captions)} total)"
                else:
                    batch_caption = ', '.join(display_captions)
        else:
            batch_caption = "training_step"
        
        if 'critictrain_latent' in critic_log_dict:
            critictrain_latent, critictrain_noisy_latent, critictrain_pred_image = map(
                lambda x: self.distillation_model.vae.decode_to_pixel(x).squeeze(1),
                [critic_log_dict['critictrain_latent'], 
                 critic_log_dict['critictrain_noisy_latent'],
                 critic_log_dict['critictrain_pred_image']]
            )

            log_data.update({
                "critictrain_latent": prepare_for_saving(critictrain_latent, caption=f"Critic Training Clean Latent: {batch_caption}"),
                "critictrain_noisy_latent": prepare_for_saving(critictrain_noisy_latent, caption=f"Critic Training Noisy Latent: {batch_caption}"),
                "critictrain_pred_image": prepare_for_saving(critictrain_pred_image, caption=f"Critic Training Predicted: {batch_caption}")
            })

        if "dmdtrain_clean_latent" in generator_log_dict:
            (dmdtrain_clean_latent, dmdtrain_noisy_latent, 
             dmdtrain_pred_real_image, dmdtrain_pred_fake_image) = map(
                lambda x: self.distillation_model.vae.decode_to_pixel(x).squeeze(1),
                [generator_log_dict['dmdtrain_clean_latent'], 
                 generator_log_dict['dmdtrain_noisy_latent'],
                 generator_log_dict['dmdtrain_pred_real_image'], 
                 generator_log_dict['dmdtrain_pred_fake_image']]
            )

            log_data.update({
                "dmdtrain_clean_latent": prepare_for_saving(dmdtrain_clean_latent, caption=f"DMD Training Clean Latent: {batch_caption}"),
                "dmdtrain_noisy_latent": prepare_for_saving(dmdtrain_noisy_latent, caption=f"DMD Training Noisy Latent: {batch_caption}"),
                "dmdtrain_pred_real_image": prepare_for_saving(dmdtrain_pred_real_image, caption=f"DMD Training Real Predicted: {batch_caption}"),
                "dmdtrain_pred_fake_image": prepare_for_saving(dmdtrain_pred_fake_image, caption=f"DMD Training Fake Predicted: {batch_caption}")
            })

    def train(self, start_epoch=0):
        num_epochs = getattr(self.config, "num_epochs", 10)
        global_step = start_epoch * len(self.dataloader)
        
        if self.is_main_process:
            if start_epoch > 0:
                logging.info(f"Resuming training from epoch {start_epoch+1} for {num_epochs} total epochs")
            else:
                logging.info(f"Starting training for {num_epochs} epochs")
            
        # Create overall training progress bar for main process only
        if self.is_main_process:
            total_steps = num_epochs * len(self.dataloader)
            overall_pbar = tqdm(
                total=total_steps, 
                desc="Training Progress", 
                position=0,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                initial=global_step  # Start from resumed position
            )
            
        for epoch in range(start_epoch, num_epochs):
            # Set current epoch for logging
            self.current_epoch = epoch
            
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
                
                # with self.accelerator.autocast():
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
                        timing_log_data = {
                            "time_per_step": time_per_step,
                            "epoch": epoch
                        }
                        
                        if not getattr(self.config, "no_wandb", False):
                            self.accelerator.log(timing_log_data, step=self.step)
                            
                            if self.wandb_run is not None:
                                self.wandb_run.log(timing_log_data, step=self.step)
                        self.previous_time = current_time
                        
                global_step += 1
                
                # Save checkpoint periodically (within epoch)
                # if not getattr(self.config, "no_save", False) and global_step % self.config.log_iters == 0:
                #     self.save(step=global_step)
                #     if global_step % (self.config.log_iters * 5) == 0:  # Less frequent GPU cache clearing
                #         torch.cuda.empty_cache()
                
            # Close epoch progress bar and log summary
            if self.is_main_process:
                epoch_pbar.close()
                
                # Log epoch summary
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                
                tqdm.write(f"✅ Epoch {epoch+1} completed in {epoch_duration:.2f}s | Avg Loss: {avg_epoch_loss:.6f}")
                logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s, Average Loss: {avg_epoch_loss:.6f}")
                
                # Log epoch summary to wandb
                epoch_summary_data = {
                    "epoch_duration": epoch_duration,
                    "epoch_avg_loss": avg_epoch_loss,
                    "epoch_completed": epoch + 1,
                    "epoch": epoch
                }
                
                if not getattr(self.config, "no_wandb", False):
                    self.accelerator.log(epoch_summary_data, step=global_step - 1)
                    
                    if self.wandb_run is not None:
                        self.wandb_run.log(epoch_summary_data, step=global_step - 1)
                
            # Save checkpoint at the end of each epoch
            if not getattr(self.config, "no_save", False) and (epoch + 1) % getattr(self.config, "epoch_save_interval", 1) == 0:
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
    parser = argparse.ArgumentParser(description="CausVid Distillation Training")
    parser.add_argument("--config_path", type=str, 
                        default="/home/coder/code/video_sketch/libs/CausVid/configs/wan_causal_dmd_finetune.yaml",
                        help="Path to configuration file")
    parser.add_argument("--no_save", action="store_true", help="Disable model saving")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--epoch_save_interval", type=int, default=1, help="Epoch interval for saving checkpoints")
    # Experiment management
    parser.add_argument("--experiment_dir", type=str, default="experiments",
                        help="Base directory for experiments")
    parser.add_argument("--experiment_name", type=str, default="causvid_distillation")
    
    # Commonly changed parameters
    parser.add_argument("--data_path", type=str, required=False,
                        default="/home/coder/code/video_sketch/data/sketch_distill/lmdb",
                        help="Override data path")
    parser.add_argument("--batch_size", type=int, default=1, help="Override batch size")
    parser.add_argument("--lr", type=float, default=2.0e-06, help="Override learning rate")
    parser.add_argument("--wandb_name", type=str, default="distillation_training", help="Override wandb run name")
    parser.add_argument("--seed", type=int, default=0, help="Override random seed")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch")
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=5, help="Generator update ratio")
    parser.add_argument("--log_iters", type=int, default=200, help="Logging interval")
    parser.add_argument("--generator_ckpt", type=str, default=None, help="Generator checkpoint")
    parser.add_argument("--real_score_type", type=str, default="14B")
    parser.add_argument("--real_score_ckpt", type=str, default=None, help="Real score checkpoint")

    parser.add_argument("--fake_score_type", type=str, default="1.3B")

    parser.add_argument("--fake_score_ckpt", type=str, default=None, help="Fake score checkpoint")

    # LoRA parameters for real_score
    parser.add_argument("--real_score_lora_ckpt", type=str, default=None, 
                        help="Path to real_score LoRA checkpoint file (.safetensors or .pth)")
    parser.add_argument("--real_score_lora_base_model", type=str, default="model",
                        help="Which model component to inject LoRA into for real_score (e.g., 'model' or 'dit')")
    parser.add_argument("--real_score_lora_rank", type=int, default=32,
                        help="LoRA rank for real_score (overrides config)")

    # LoRA parameters for fake_score
    parser.add_argument("--fake_score_lora_ckpt", type=str, default=None,
                        help="Path to fake_score LoRA checkpoint file (.safetensors or .pth)")
    parser.add_argument("--fake_score_lora_base_model", type=str, default="model",
                        help="Which model component to inject LoRA into for fake_score (e.g., 'model' or 'dit')")
    parser.add_argument("--fake_score_lora_rank", type=int, default=32,
                        help="LoRA rank for fake_score (overrides config)")

    # Resume training parameters
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to experiment directory to resume training from")
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="Specific epoch to resume from (if not specified, will use latest checkpoint)")

    args = parser.parse_args()

    if args.debug:
        args.no_wandb = True
        args.num_epochs = 1

    # Load config and override with command-line arguments
    config = OmegaConf.load(args.config_path)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})

    # Setup experiment with enhanced logging and management
    exp_dir, wandb_run = setup_experiment(
        args, 
        base_output_dir=args.experiment_dir,
        project_name=args.experiment_name if not args.no_wandb else None,
        exp_name_func=generate_causvid_distillation_exp_name
    )
    
    # If WandB is disabled, set wandb_run to None
    if args.no_wandb:
        wandb_run = None
        
    # Update config with experiment directory
    config.output_folder = exp_dir
    
    logging.info("Starting CausVid distillation training...")
    logging.info(f"Experiment directory: {exp_dir}")
    logging.info(f"Data path: {args.data_path}")
    logging.info(f"Dataset repeat: {args.dataset_repeat}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Epochs: {args.num_epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Generator update ratio: {args.dfake_gen_update_ratio}")

    # Log LoRA configuration if used
    if args.real_score_lora_ckpt:
        logging.info(f"Real Score LoRA: {args.real_score_lora_ckpt}")
        logging.info(f"Real Score LoRA base model: {args.real_score_lora_base_model}")
        logging.info(f"Real Score LoRA rank: {args.real_score_lora_rank}")
    if args.fake_score_lora_ckpt:
        logging.info(f"Fake Score LoRA: {args.fake_score_lora_ckpt}")
        logging.info(f"Fake Score LoRA base model: {args.fake_score_lora_base_model}")
        logging.info(f"Fake Score LoRA rank: {args.fake_score_lora_rank}")

    # Create trainer and start training
    trainer = DistillationTrainer(config, exp_dir=exp_dir, wandb_run=wandb_run)
    
    # Handle resume training
    start_epoch = 0
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Resume directory does not exist: {args.resume_from}")
        
        logging.info(f"Attempting to resume training from: {args.resume_from}")
        start_epoch = trainer.load_checkpoint(args.resume_from, args.resume_epoch)
        
        if start_epoch == 0:
            logging.warning("Failed to load checkpoint, starting from scratch")
        else:
            logging.info(f"Successfully resumed from epoch {start_epoch}")
    
    trainer.train(start_epoch=start_epoch)
    
    # Log training completion
    logging.info("Training completed successfully")


if __name__ == "__main__":
    main()

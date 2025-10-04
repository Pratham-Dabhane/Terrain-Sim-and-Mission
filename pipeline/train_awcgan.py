"""
Training script for CLIP-conditioned aWCGAN (Wasserstein GAN with Gradient Penalty)
Optimized for low VRAM (GTX 1650, 4GB) with gradient checkpointing and memory-efficient training.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
from datetime import datetime
from tqdm import tqdm
import json

# Local imports
from data import DESDataModule, create_synthetic_data
from clip_encoder import CLIPTextEncoderWithProcessor
from models_awcgan import CLIPConditionedGenerator, CLIPConditionedCritic, gradient_penalty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWCGANTrainer:
    """
    Trainer for CLIP-conditioned aWCGAN with memory optimization for low VRAM.
    
    Args:
        generator: Generator model
        critic: Critic model
        clip_encoder: CLIP text encoder
        dataloader: Training data loader
        device: Training device
        config: Training configuration dictionary
    """
    
    def __init__(
        self,
        generator: CLIPConditionedGenerator,
        critic: CLIPConditionedCritic,
        clip_encoder: CLIPTextEncoderWithProcessor,
        dataloader: DataLoader,
        device: str,
        config: dict
    ):
        self.generator = generator
        self.critic = critic
        self.clip_encoder = clip_encoder
        self.dataloader = dataloader
        self.device = device
        self.config = config
        
        # Setup optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config['lr_g'],
            betas=(config['beta1'], config['beta2'])
        )
        
        self.optimizer_c = optim.Adam(
            self.critic.parameters(),
            lr=config['lr_c'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = {
            'g_loss': [],
            'c_loss': [],
            'gp_loss': [],
            'clip_similarity': []
        }
        
        # Sample prompts for monitoring
        self.sample_prompts = [
            "mountainous terrain with steep peaks",
            "rolling hills with gentle slopes",
            "desert landscape with sand dunes",
            "river valley with water flowing",
            "volcanic terrain with craters",
            "forest covered hills",
            "coastal cliffs and beaches",
            "arctic tundra landscape"
        ]
        
        # Create output directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['sample_dir'], exist_ok=True)
        
        logger.info(f"Trainer initialized with {len(dataloader)} batches per epoch")
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        if hasattr(self.generator, 'gradient_checkpointing_enable'):
            self.generator.gradient_checkpointing_enable()
        if hasattr(self.critic, 'gradient_checkpointing_enable'):
            self.critic.gradient_checkpointing_enable()
    
    def clear_cuda_cache(self):
        """Clear CUDA cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def compute_clip_similarity_loss(self, generated_heightmaps: torch.Tensor, clip_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute CLIP similarity loss to encourage semantic consistency.
        
        Args:
            generated_heightmaps: Generated heightmaps (B, 1, H, W)
            clip_embeddings: Original CLIP embeddings (B, clip_dim)
            
        Returns:
            torch.Tensor: CLIP similarity loss
        """
        # Convert heightmaps to pseudo-RGB for CLIP image encoder
        # (This is a simplified approach - in practice, you might want to use a separate image encoder)
        pseudo_rgb = generated_heightmaps.repeat(1, 3, 1, 1)
        
        # Resize to CLIP input size (224x224)
        pseudo_rgb = nn.functional.interpolate(pseudo_rgb, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Note: For full implementation, you would encode these images with CLIP image encoder
        # and compute cosine similarity with text embeddings
        # For now, we'll use a placeholder loss
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def train_critic_step(self, real_heightmaps: torch.Tensor, prompts: list) -> dict:
        """
        Train critic for one step.
        
        Args:
            real_heightmaps: Real heightmaps (B, 1, H, W)
            prompts: List of terrain prompts
            
        Returns:
            dict: Critic training metrics
        """
        batch_size = real_heightmaps.size(0)
        
        # Zero gradients
        self.optimizer_c.zero_grad()
        
        # Encode prompts
        with torch.no_grad():
            clip_embeddings = self.clip_encoder.encode_text_with_enhancement(prompts)
        
        # Generate fake heightmaps
        noise = torch.randn(batch_size, self.config['noise_dim'], device=self.device)
        with torch.no_grad():
            fake_heightmaps = self.generator(noise, clip_embeddings)
        
        # Critic scores
        real_scores = self.critic(real_heightmaps, clip_embeddings)
        fake_scores = self.critic(fake_heightmaps.detach(), clip_embeddings)
        
        # Wasserstein loss
        w_loss = torch.mean(fake_scores) - torch.mean(real_scores)
        
        # Gradient penalty
        gp = gradient_penalty(self.critic, real_heightmaps, fake_heightmaps, clip_embeddings, self.device)
        
        # Total critic loss
        c_loss = w_loss + self.config['lambda_gp'] * gp
        
        # Backward pass
        c_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer_c.step()
        
        return {
            'c_loss': c_loss.item(),
            'w_loss': w_loss.item(),
            'gp_loss': gp.item(),
            'real_score': torch.mean(real_scores).item(),
            'fake_score': torch.mean(fake_scores).item()
        }
    
    def train_generator_step(self, prompts: list) -> dict:
        """
        Train generator for one step.
        
        Args:
            prompts: List of terrain prompts
            
        Returns:
            dict: Generator training metrics
        """
        batch_size = len(prompts)
        
        # Zero gradients
        self.optimizer_g.zero_grad()
        
        # Encode prompts
        clip_embeddings = self.clip_encoder.encode_text_with_enhancement(prompts)
        
        # Generate fake heightmaps
        noise = torch.randn(batch_size, self.config['noise_dim'], device=self.device)
        fake_heightmaps = self.generator(noise, clip_embeddings)
        
        # Critic scores for fake samples
        fake_scores = self.critic(fake_heightmaps, clip_embeddings)
        
        # Generator loss (negative critic score)
        g_loss = -torch.mean(fake_scores)
        
        # CLIP similarity loss (optional, for semantic consistency)
        if self.config['lambda_clip'] > 0:
            clip_loss = self.compute_clip_similarity_loss(fake_heightmaps, clip_embeddings)
            g_loss += self.config['lambda_clip'] * clip_loss
        
        # Backward pass
        g_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer_g.step()
        
        return {
            'g_loss': g_loss.item(),
            'fake_score': torch.mean(fake_scores).item()
        }
    
    def generate_samples(self, prompts: list, save_path: str = None) -> torch.Tensor:
        """
        Generate sample heightmaps for monitoring.
        
        Args:
            prompts: List of prompts to generate
            save_path: Optional path to save samples
            
        Returns:
            torch.Tensor: Generated heightmaps
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Encode prompts
            clip_embeddings = self.clip_encoder.encode_text_with_enhancement(prompts)
            
            # Generate samples
            noise = torch.randn(len(prompts), self.config['noise_dim'], device=self.device)
            samples = self.generator(noise, clip_embeddings)
            
            # Convert to [0, 1] range for visualization
            samples = (samples + 1) / 2
            
            if save_path:
                self.save_sample_grid(samples, prompts, save_path)
        
        self.generator.train()
        return samples
    
    def save_sample_grid(self, samples: torch.Tensor, prompts: list, save_path: str):
        """Save a grid of generated samples"""
        batch_size = samples.size(0)
        grid_size = int(np.ceil(np.sqrt(batch_size)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        if grid_size == 1:
            axes = [axes]
        elif grid_size > 1:
            axes = axes.flatten()
        
        for i in range(batch_size):
            heightmap = samples[i, 0].cpu().numpy()
            
            ax = axes[i] if grid_size > 1 else axes[0]
            im = ax.imshow(heightmap, cmap='terrain', vmin=0, vmax=1)
            ax.set_title(prompts[i][:30] + "..." if len(prompts[i]) > 30 else prompts[i], fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide unused subplots
        for i in range(batch_size, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Samples saved to {save_path}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_c_state_dict': self.optimizer_c.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_c.load_state_dict(checkpoint['optimizer_c_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.generator.train()
        self.critic.train()
        
        epoch_metrics = {'g_loss': [], 'c_loss': [], 'gp_loss': []}
        
        pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, real_heightmaps in enumerate(pbar):
            real_heightmaps = real_heightmaps.to(self.device)
            batch_size = real_heightmaps.size(0)
            
            # Generate random prompts for this batch
            # In practice, you would have paired text-heightmap data
            prompts = np.random.choice(self.sample_prompts, size=batch_size, replace=True).tolist()
            
            # Train critic multiple times per generator update
            for _ in range(self.config['n_critic']):
                critic_metrics = self.train_critic_step(real_heightmaps, prompts)
                epoch_metrics['c_loss'].append(critic_metrics['c_loss'])
                epoch_metrics['gp_loss'].append(critic_metrics['gp_loss'])
            
            # Train generator
            generator_metrics = self.train_generator_step(prompts)
            epoch_metrics['g_loss'].append(generator_metrics['g_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{generator_metrics['g_loss']:.4f}",
                'C_loss': f"{critic_metrics['c_loss']:.4f}",
                'GP': f"{critic_metrics['gp_loss']:.4f}"
            })
            
            self.global_step += 1
            
            # Clear cache periodically to prevent OOM
            if batch_idx % 10 == 0:
                self.clear_cuda_cache()
        
        # Update training history
        for key in epoch_metrics:
            if epoch_metrics[key]:
                self.training_history[key].append(np.mean(epoch_metrics[key]))
        
        # Generate samples
        if epoch % self.config['sample_interval'] == 0:
            sample_path = os.path.join(self.config['sample_dir'], f'samples_epoch_{epoch}.png')
            self.generate_samples(self.sample_prompts[:8], sample_path)
        
        logger.info(f"Epoch {epoch} completed - G_loss: {np.mean(epoch_metrics['g_loss']):.4f}, "
                   f"C_loss: {np.mean(epoch_metrics['c_loss']):.4f}")
    
    def train(self, num_epochs: int, resume_from: str = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        start_epoch = self.current_epoch + 1
        
        logger.info(f"Starting training from epoch {start_epoch} to {num_epochs}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            try:
                self.train_epoch(epoch)
                
                # Save checkpoint
                if epoch % self.config['checkpoint_interval'] == 0:
                    self.save_checkpoint(epoch)
                
                self.current_epoch = epoch
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("CUDA out of memory! Clearing cache and continuing...")
                    self.clear_cuda_cache()
                    continue
                else:
                    raise e
        
        # Save final checkpoint
        self.save_checkpoint(num_epochs, is_best=True)
        logger.info("Training completed!")


def create_training_config():
    """Create default training configuration"""
    return {
        # Model parameters
        'noise_dim': 128,
        'clip_dim': 512,
        'image_size': 256,
        
        # Training parameters
        'batch_size': 1,  # Very small for GTX 1650
        'num_epochs': 100,
        'lr_g': 0.0001,
        'lr_c': 0.0002,
        'beta1': 0.0,
        'beta2': 0.9,
        
        # WGAN-GP parameters
        'n_critic': 5,
        'lambda_gp': 10.0,
        'lambda_clip': 0.1,
        
        # Memory optimization
        'gradient_checkpointing': True,
        'mixed_precision': True,
        
        # Logging and saving
        'checkpoint_interval': 5,
        'sample_interval': 2,
        'checkpoint_dir': 'checkpoints',
        'sample_dir': 'samples',
        'log_interval': 50
    }


def main():
    """Main training function"""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Configuration
    config = create_training_config()
    
    # Create synthetic data if needed
    data_dir = "data/synthetic_heightmaps"
    if not os.path.exists(data_dir):
        logger.info("Creating synthetic training data...")
        create_synthetic_data(data_dir, num_samples=200)
    
    # Setup data
    datamodule = DESDataModule(
        data_dir=data_dir,
        batch_size=config['batch_size'],
        image_size=config['image_size']
    )
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    
    # Initialize models
    clip_encoder = CLIPTextEncoderWithProcessor(
        embedding_dim=config['clip_dim'],
        device=device
    )
    
    generator = CLIPConditionedGenerator(
        noise_dim=config['noise_dim'],
        clip_dim=config['clip_dim'],
        output_size=config['image_size']
    ).to(device)
    
    critic = CLIPConditionedCritic(
        input_size=config['image_size'],
        clip_dim=config['clip_dim']
    ).to(device)
    
    # Initialize trainer
    trainer = AWCGANTrainer(
        generator=generator,
        critic=critic,
        clip_encoder=clip_encoder,
        dataloader=train_dataloader,
        device=device,
        config=config
    )
    
    # Enable memory optimizations
    if config['gradient_checkpointing']:
        trainer.enable_gradient_checkpointing()
    
    # Train
    try:
        trainer.train(num_epochs=config['num_epochs'])
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
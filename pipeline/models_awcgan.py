"""
CLIP-conditioned Wasserstein GAN (aWCGAN) for Terrain Generation
Generator and Critic networks with FiLM/AdaIN modulation for text conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer for conditioning.
    
    Args:
        in_channels: Input feature channels
        condition_dim: Conditioning vector dimension
    """
    
    def __init__(self, in_channels: int, condition_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.condition_dim = condition_dim
        
        # Linear layers for scale and bias
        self.scale_transform = nn.Linear(condition_dim, in_channels)
        self.bias_transform = nn.Linear(condition_dim, in_channels)
        
        # Initialize with identity transformation
        nn.init.ones_(self.scale_transform.weight)
        nn.init.zeros_(self.scale_transform.bias)
        nn.init.zeros_(self.bias_transform.weight)
        nn.init.zeros_(self.bias_transform.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.
        
        Args:
            x: Input features (B, C, H, W)
            condition: Conditioning vector (B, condition_dim)
            
        Returns:
            torch.Tensor: Modulated features
        """
        batch_size = x.size(0)
        
        # Generate scale and bias
        scale = self.scale_transform(condition).view(batch_size, self.in_channels, 1, 1)
        bias = self.bias_transform(condition).view(batch_size, self.in_channels, 1, 1)
        
        # Apply modulation
        return x * scale + bias


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer for style conditioning.
    
    Args:
        in_channels: Input feature channels
        condition_dim: Conditioning vector dimension
    """
    
    def __init__(self, in_channels: int, condition_dim: int):
        super().__init__()
        self.in_channels = in_channels
        
        # Style transformation
        self.style_transform = nn.Linear(condition_dim, in_channels * 2)
        nn.init.normal_(self.style_transform.weight, 0.02)
        nn.init.zeros_(self.style_transform.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN conditioning.
        
        Args:
            x: Input features (B, C, H, W)
            condition: Conditioning vector (B, condition_dim)
            
        Returns:
            torch.Tensor: Normalized and modulated features
        """
        batch_size, channels = x.size(0), x.size(1)
        
        # Instance normalization
        x_mean = x.view(batch_size, channels, -1).mean(dim=2, keepdim=True).unsqueeze(3)
        x_std = x.view(batch_size, channels, -1).std(dim=2, keepdim=True).unsqueeze(3)
        x_normalized = (x - x_mean) / (x_std + 1e-8)
        
        # Generate style parameters
        style_params = self.style_transform(condition).view(batch_size, channels, 2)
        scale = style_params[:, :, 0].unsqueeze(2).unsqueeze(3)
        bias = style_params[:, :, 1].unsqueeze(2).unsqueeze(3)
        
        return x_normalized * (1 + scale) + bias


class ConditionalGeneratorBlock(nn.Module):
    """
    Generator block with conditional modulation.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        condition_dim: CLIP embedding dimension
        upsample: Whether to upsample
        modulation_type: 'film' or 'adain'
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        upsample: bool = True,
        modulation_type: str = 'film'
    ):
        super().__init__()
        
        self.upsample = upsample
        
        # Main convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Normalization
        self.norm = nn.InstanceNorm2d(out_channels, affine=False)
        
        # Conditional modulation
        if modulation_type == 'film':
            self.modulation = FiLM(out_channels, condition_dim)
        elif modulation_type == 'adain':
            self.modulation = AdaIN(out_channels, condition_dim)
        else:
            raise ValueError(f"Unknown modulation type: {modulation_type}")
        
        # Activation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Optional upsampling
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass with conditioning"""
        if self.upsample:
            x = self.upsample_layer(x)
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.modulation(x, condition)
        x = self.activation(x)
        
        return x


class CLIPConditionedGenerator(nn.Module):
    """
    CLIP-conditioned Generator for terrain heightmaps.
    
    Args:
        noise_dim: Dimension of input noise vector
        clip_dim: CLIP embedding dimension
        output_size: Output heightmap size (default: 256)
        base_channels: Base number of channels (scaled by depth)
        modulation_type: 'film' or 'adain'
    """
    
    def __init__(
        self,
        noise_dim: int = 128,
        clip_dim: int = 512,
        output_size: int = 256,
        base_channels: int = 256,
        modulation_type: str = 'film'
    ):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.clip_dim = clip_dim
        self.output_size = output_size
        
        # Calculate number of upsampling blocks needed
        self.num_blocks = int(np.log2(output_size)) - 2  # Start from 4x4
        
        # Initial projection
        self.initial_size = 4
        self.initial_projection = nn.Linear(
            noise_dim + clip_dim, 
            base_channels * self.initial_size * self.initial_size
        )
        
        # Generator blocks
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(self.num_blocks):
            out_channels = max(base_channels // (2 ** (i + 1)), 16)
            
            block = ConditionalGeneratorBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                condition_dim=clip_dim,
                upsample=True,
                modulation_type=modulation_type
            )
            
            self.blocks.append(block)
            current_channels = out_channels
        
        # Final output layer
        self.output_conv = nn.Conv2d(current_channels, 1, kernel_size=3, padding=1)
        self.output_activation = nn.Tanh()  # Output in [-1, 1]
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Generator initialized: {noise_dim}+{clip_dim} -> {output_size}x{output_size}")
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, noise: torch.Tensor, clip_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate heightmap from noise and CLIP embedding.
        
        Args:
            noise: Random noise vector (B, noise_dim)
            clip_embedding: CLIP text embedding (B, clip_dim)
            
        Returns:
            torch.Tensor: Generated heightmap (B, 1, H, W)
        """
        batch_size = noise.size(0)
        
        # Concatenate noise and CLIP embedding
        combined_input = torch.cat([noise, clip_embedding], dim=1)
        
        # Initial projection
        x = self.initial_projection(combined_input)
        x = x.view(batch_size, -1, self.initial_size, self.initial_size)
        
        # Apply generator blocks with conditioning
        for block in self.blocks:
            x = block(x, clip_embedding)
        
        # Final output
        x = self.output_conv(x)
        x = self.output_activation(x)
        
        return x


class SpectralNorm(nn.Module):
    """Spectral normalization for stable training"""
    
    def __init__(self, module: nn.Module, name: str = 'weight'):
        super().__init__()
        self.module = module
        self.name = name
        
        # Get weight tensor
        weight = getattr(module, name)
        
        # Initialize u vector
        self.register_buffer('u', torch.randn(weight.size(0)))
        
    def forward(self, *args, **kwargs):
        weight = getattr(self.module, self.name)
        
        # Power iteration
        u = self.u
        v = F.normalize(torch.mv(weight.t(), u), dim=0, eps=1e-12)
        u = F.normalize(torch.mv(weight, v), dim=0, eps=1e-12)
        
        # Spectral norm
        sigma = torch.dot(u, torch.mv(weight, v))
        
        # Normalize weight
        setattr(self.module, self.name, weight / sigma)
        
        # Update u
        self.u.copy_(u)
        
        return self.module(*args, **kwargs)


class CLIPConditionedCritic(nn.Module):
    """
    CLIP-conditioned Critic (Discriminator) for WGAN-GP.
    
    Args:
        input_size: Input heightmap size
        clip_dim: CLIP embedding dimension
        base_channels: Base number of channels
    """
    
    def __init__(
        self,
        input_size: int = 256,
        clip_dim: int = 512,
        base_channels: int = 16
    ):
        super().__init__()
        
        self.input_size = input_size
        self.clip_dim = clip_dim
        
        # Calculate number of downsampling blocks
        self.num_blocks = int(np.log2(input_size)) - 2  # End at 4x4
        
        # Initial convolution (no normalization for first layer)
        self.initial_conv = nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1)
        
        # Downsampling blocks
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(self.num_blocks - 1):
            out_channels = min(current_channels * 2, 512)
            
            block = nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
            self.blocks.append(block)
            current_channels = out_channels
        
        # CLIP conditioning layer
        self.clip_projection = nn.Sequential(
            nn.Linear(clip_dim, current_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final layers
        final_size = 4
        self.final_conv = nn.Conv2d(
            current_channels, current_channels, 
            kernel_size=final_size, stride=1, padding=0
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(current_channels + clip_dim, current_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(current_channels, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Critic initialized: {input_size}x{input_size}+{clip_dim} -> 1")
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, heightmap: torch.Tensor, clip_embedding: torch.Tensor) -> torch.Tensor:
        """
        Evaluate heightmap authenticity conditioned on CLIP embedding.
        
        Args:
            heightmap: Input heightmap (B, 1, H, W)
            clip_embedding: CLIP text embedding (B, clip_dim)
            
        Returns:
            torch.Tensor: Critic score (B, 1)
        """
        batch_size = heightmap.size(0)
        
        # Initial convolution
        x = self.initial_conv(heightmap)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        # Downsampling blocks
        for block in self.blocks:
            x = block(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        # Global average pooling
        x = x.view(batch_size, -1)
        
        # Concatenate with CLIP embedding
        combined = torch.cat([x, clip_embedding], dim=1)
        
        # Output
        output = self.output_head(combined)
        
        return output


def gradient_penalty(critic, real_samples, fake_samples, clip_embeddings, device):
    """
    Calculate gradient penalty for WGAN-GP.
    
    Args:
        critic: Critic network
        real_samples: Real heightmaps
        fake_samples: Generated heightmaps
        clip_embeddings: CLIP embeddings
        device: Device to run on
        
    Returns:
        torch.Tensor: Gradient penalty loss
    """
    batch_size = real_samples.size(0)
    
    # Random interpolation factor
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolated samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Critic output on interpolated samples
    critic_output = critic(interpolated, clip_embeddings)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Gradient penalty
    gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return penalty


def test_models():
    """Test function for aWCGAN models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    try:
        # Initialize models
        generator = CLIPConditionedGenerator(
            noise_dim=128,
            clip_dim=512,
            output_size=256
        ).to(device)
        
        critic = CLIPConditionedCritic(
            input_size=256,
            clip_dim=512
        ).to(device)
        
        print(f"Models initialized on {device}")
        
        # Test inputs
        noise = torch.randn(batch_size, 128, device=device)
        clip_embedding = torch.randn(batch_size, 512, device=device)
        
        # Generator forward pass
        fake_heightmaps = generator(noise, clip_embedding)
        print(f"Generator output shape: {fake_heightmaps.shape}")
        print(f"Generator output range: [{fake_heightmaps.min():.3f}, {fake_heightmaps.max():.3f}]")
        
        # Critic forward pass
        critic_score = critic(fake_heightmaps, clip_embedding)
        print(f"Critic output shape: {critic_score.shape}")
        print(f"Critic score range: [{critic_score.min():.3f}, {critic_score.max():.3f}]")
        
        # Test gradient penalty
        real_heightmaps = torch.randn_like(fake_heightmaps)
        gp = gradient_penalty(critic, real_heightmaps, fake_heightmaps, clip_embedding, device)
        print(f"Gradient penalty: {gp.item():.4f}")
        
        print("aWCGAN models test passed!")
        
    except Exception as e:
        print(f"aWCGAN models test failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_models()
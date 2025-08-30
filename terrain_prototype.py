#!/usr/bin/env python3
"""
Generative AI Terrain Prototype
A prototype that generates 2.5D terrain from text prompts using GAN and Diffusion models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os
import warnings
warnings.filterwarnings("ignore")

# Try to import required libraries, provide fallbacks if not available
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers.utils import load_image
    DIFFUSION_AVAILABLE = True
except ImportError:
    print("Warning: diffusers library not available. Install with: pip install diffusers transformers")
    DIFFUSION_AVAILABLE = False

try:
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False


class SimpleStyleGAN2Generator:
    """
    A simplified StyleGAN2 generator for terrain heightmap generation.
    This is a basic implementation since the full StyleGAN2-ADA requires significant resources.
    """
    
    def __init__(self, latent_dim=512, output_size=256):
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # Simple generator architecture
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_size * output_size),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.generator.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def generate(self, num_samples=1):
        """Generate terrain heightmaps from random latent vectors."""
        if not TORCH_AVAILABLE:
            # Fallback: generate random heightmap
            return self._generate_random_heightmap(num_samples)
        
        with torch.no_grad():
            # Generate random latent vectors
            z = torch.randn(num_samples, self.latent_dim)
            
            # Generate heightmaps
            heightmaps = self.generator(z)
            heightmaps = heightmaps.view(num_samples, self.output_size, self.output_size)
            
            # Convert to numpy
            heightmaps = heightmaps.cpu().numpy()
            
            return heightmaps
    
    def _generate_random_heightmap(self, num_samples=1):
        """Fallback method to generate random heightmaps when PyTorch is not available."""
        heightmaps = []
        for _ in range(num_samples):
            # Generate Perlin-like noise for terrain
            size = self.output_size
            heightmap = np.random.rand(size, size)
            
            # Apply smoothing to make it more terrain-like
            from scipy.ndimage import gaussian_filter
            try:
                heightmap = gaussian_filter(heightmap, sigma=2.0)
            except ImportError:
                # Simple smoothing if scipy not available
                heightmap = self._simple_smooth(heightmap)
            
            heightmaps.append(heightmap)
        
        return np.array(heightmaps)
    
    def _simple_smooth(self, heightmap):
        """Simple smoothing function when scipy is not available."""
        size = heightmap.shape[0]
        smoothed = heightmap.copy()
        
        # Simple 3x3 smoothing kernel
        for i in range(1, size-1):
            for j in range(1, size-1):
                smoothed[i, j] = (
                    heightmap[i-1, j-1] + heightmap[i-1, j] + heightmap[i-1, j+1] +
                    heightmap[i, j-1] + heightmap[i, j] + heightmap[i, j+1] +
                    heightmap[i+1, j-1] + heightmap[i+1, j] + heightmap[i+1, j+1]
                ) / 9.0
        
        return smoothed


def prompt_to_heightmap_gan(prompt, size=256):
    """
    Generate a base terrain heightmap using GAN from text prompt.
    
    Args:
        prompt (str): Text description of the terrain
        size (int): Size of the output heightmap
    
    Returns:
        numpy.ndarray: Grayscale heightmap (0-1 range)
    """
    print(f"Generating heightmap for prompt: '{prompt}'")
    
    # Initialize the generator
    generator = SimpleStyleGAN2Generator(output_size=size)
    
    # Generate heightmap
    heightmap = generator.generate(num_samples=1)[0]
    
    # Apply terrain-specific modifications based on prompt
    heightmap = _apply_prompt_modifications(heightmap, prompt)
    
    print("Heightmap generation completed!")
    return heightmap


def _apply_prompt_modifications(heightmap, prompt):
    """Apply terrain modifications based on the text prompt."""
    prompt_lower = prompt.lower()
    
    # Mountain modifications
    if 'mountain' in prompt_lower or 'peak' in prompt_lower:
        heightmap = _add_mountains(heightmap)
    
    # Valley modifications
    if 'valley' in prompt_lower or 'canyon' in prompt_lower:
        heightmap = _add_valleys(heightmap)
    
    # River modifications
    if 'river' in prompt_lower or 'water' in prompt_lower:
        heightmap = _add_rivers(heightmap)
    
    # Forest modifications
    if 'forest' in prompt_lower or 'trees' in prompt_lower:
        heightmap = _add_forest_roughness(heightmap)
    
    # Desert modifications
    if 'desert' in prompt_lower or 'sand' in prompt_lower:
        heightmap = _add_desert_features(heightmap)
    
    return heightmap


def _add_mountains(heightmap):
    """Add mountain features to the heightmap."""
    size = heightmap.shape[0]
    
    # Create mountain peaks
    for _ in range(3):
        center_x = np.random.randint(size//4, 3*size//4)
        center_y = np.random.randint(size//4, 3*size//4)
        radius = np.random.randint(size//8, size//4)
        height = np.random.uniform(0.3, 0.8)
        
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < radius:
                    factor = 1 - (dist / radius)
                    heightmap[i, j] += height * factor * factor
    
    return np.clip(heightmap, 0, 1)


def _add_valleys(heightmap):
    """Add valley features to the heightmap."""
    size = heightmap.shape[0]
    
    # Create valleys
    for _ in range(2):
        start_x = np.random.randint(0, size)
        start_y = np.random.randint(0, size)
        end_x = np.random.randint(0, size)
        end_y = np.random.randint(0, size)
        
        # Create a valley path
        steps = 50
        for step in range(steps + 1):
            t = step / steps
            x = int(start_x + t * (end_x - start_x))
            y = int(start_y + t * (end_y - start_y))
            
            if 0 <= x < size and 0 <= y < size:
                # Create valley depression
                for dx in range(-size//8, size//8):
                    for dy in range(-size//8, size//8):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < size//8:
                                factor = 1 - (dist / (size//8))
                                heightmap[nx, ny] -= 0.2 * factor * factor
    
    return np.clip(heightmap, 0, 1)


def _add_rivers(heightmap):
    """Add river features to the heightmap."""
    size = heightmap.shape[0]
    
    # Create river paths
    for _ in range(2):
        start_x = np.random.randint(0, size)
        start_y = np.random.randint(0, size)
        end_x = np.random.randint(0, size)
        end_y = np.random.randint(0, size)
        
        # Create a meandering river
        steps = 100
        for step in range(steps + 1):
            t = step / steps
            x = int(start_x + t * (end_x - start_x))
            y = int(start_y + t * (end_y - start_y))
            
            # Add some randomness to the path
            x += np.random.randint(-size//16, size//16)
            y += np.random.randint(-size//16, size//16)
            
            if 0 <= x < size and 0 <= y < size:
                # Create river depression
                for dx in range(-size//16, size//16):
                    for dy in range(-size//16, size//16):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < size//16:
                                factor = 1 - (dist / (size//16))
                                heightmap[nx, ny] -= 0.3 * factor * factor
    
    return np.clip(heightmap, 0, 1)


def _add_forest_roughness(heightmap):
    """Add forest-like roughness to the heightmap."""
    size = heightmap.shape[0]
    
    # Add small random variations
    noise = np.random.rand(size, size) * 0.1
    heightmap += noise
    
    return np.clip(heightmap, 0, 1)


def _add_desert_features(heightmap):
    """Add desert-like features to the heightmap."""
    size = heightmap.shape[0]
    
    # Add sand dunes
    for _ in range(5):
        center_x = np.random.randint(0, size)
        center_y = np.random.randint(0, size)
        radius = np.random.randint(size//16, size//8)
        height = np.random.uniform(0.1, 0.3)
        
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < radius:
                    factor = 1 - (dist / radius)
                    heightmap[i, j] += height * factor * factor
    
    return np.clip(heightmap, 0, 1)


def refine_with_diffusion(heightmap, prompt):
    """
    Refine the heightmap using Stable Diffusion with ControlNet.
    
    Args:
        heightmap (numpy.ndarray): Input heightmap
        prompt (str): Text prompt for refinement
    
    Returns:
        numpy.ndarray: Refined terrain image
    """
    print("Refining heightmap with diffusion model...")
    
    if not DIFFUSION_AVAILABLE:
        print("Diffusion model not available, using basic enhancement instead.")
        return _basic_enhancement(heightmap, prompt)
    
    try:
        # Convert heightmap to depth map format
        depth_map = _heightmap_to_depth_map(heightmap)
        
        # Load ControlNet model for depth
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16
        )
        
        # Load Stable Diffusion pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        
        # Generate refined image
        refined_image = pipe(
            prompt=f"realistic terrain, {prompt}, high quality, detailed",
            image=depth_map,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        # Convert PIL image to numpy array
        refined_array = np.array(refined_image)
        
        print("Diffusion refinement completed!")
        return refined_array
        
    except Exception as e:
        print(f"Error in diffusion refinement: {e}")
        print("Falling back to basic enhancement...")
        return _basic_enhancement(heightmap, prompt)


def _heightmap_to_depth_map(heightmap):
    """Convert heightmap to depth map format for ControlNet."""
    # Normalize to 0-255 range
    depth_map = (heightmap * 255).astype(np.uint8)
    
    # Create PIL Image
    depth_image = Image.fromarray(depth_map, mode='L')
    
    # Resize to 512x512 (typical SD input size)
    depth_image = depth_image.resize((512, 512), Image.LANCZOS)
    
    return depth_image


def _basic_enhancement(heightmap, prompt):
    """Basic enhancement when diffusion model is not available."""
    print("Applying basic terrain enhancement...")
    
    # Apply some basic filters to enhance the terrain
    enhanced = heightmap.copy()
    
    # Add some contrast
    enhanced = np.power(enhanced, 0.8)
    
    # Add some texture
    noise = np.random.rand(*enhanced.shape) * 0.05
    enhanced += noise
    
    # Normalize
    enhanced = np.clip(enhanced, 0, 1)
    
    # Convert to RGB for visualization
    rgb_terrain = np.stack([enhanced] * 3, axis=-1)
    
    # Apply color mapping based on height
    rgb_terrain = _apply_terrain_colors(rgb_terrain, enhanced)
    
    return (rgb_terrain * 255).astype(np.uint8)


def _apply_terrain_colors(rgb_terrain, heightmap):
    """Apply realistic terrain colors based on height."""
    # Define color schemes for different height ranges
    colors = {
        'water': [0.2, 0.4, 0.8],      # Blue for low areas
        'sand': [0.9, 0.8, 0.6],       # Sand for low-mid areas
        'grass': [0.3, 0.6, 0.2],      # Green for mid areas
        'forest': [0.2, 0.4, 0.1],     # Dark green for mid-high areas
        'rock': [0.5, 0.5, 0.5],       # Gray for high areas
        'snow': [0.9, 0.9, 0.9]        # White for highest areas
    }
    
    # Apply colors based on height
    for i in range(heightmap.shape[0]):
        for j in range(heightmap.shape[1]):
            height = heightmap[i, j]
            
            if height < 0.2:
                color = colors['water']
            elif height < 0.35:
                color = colors['sand']
            elif height < 0.5:
                color = colors['grass']
            elif height < 0.7:
                color = colors['forest']
            elif height < 0.85:
                color = colors['rock']
            else:
                color = colors['snow']
            
            rgb_terrain[i, j] = color
    
    return rgb_terrain


def visualize_terrain(terrain_image, heightmap, prompt, save_path=None):
    """
    Visualize the generated terrain.
    
    Args:
        terrain_image (numpy.ndarray): The final terrain image
        heightmap (numpy.ndarray): The original heightmap
        prompt (str): The input prompt
        save_path (str): Optional path to save the visualization
    """
    print("Creating terrain visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Generative AI Terrain: "{prompt}"', fontsize=16, fontweight='bold')
    
    # Plot 1: Original heightmap
    axes[0].imshow(heightmap, cmap='terrain', aspect='equal')
    axes[0].set_title('Generated Heightmap', fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Final terrain image
    if len(terrain_image.shape) == 3:
        axes[1].imshow(terrain_image)
    else:
        axes[1].imshow(terrain_image, cmap='terrain')
    axes[1].set_title('Final Terrain (2.5D)', fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: 3D surface plot
    ax3d = fig.add_subplot(133, projection='3d')
    y, x = np.mgrid[0:heightmap.shape[0]:1, 0:heightmap.shape[1]:1]
    ax3d.plot_surface(x, y, heightmap, cmap='terrain', alpha=0.8)
    ax3d.set_title('3D Terrain View', fontweight='bold')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Height')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to run the terrain generation prototype."""
    print("=" * 60)
    print("Generative AI Terrain Prototype")
    print("=" * 60)
    
    # Example prompts
    example_prompts = [
        "mountainous terrain with rivers and valleys",
        "desert landscape with sand dunes",
        "forest terrain with rolling hills",
        "coastal cliffs with rocky outcrops",
        "alpine landscape with snow-capped peaks"
    ]
    
    print("\nExample terrain prompts:")
    for i, prompt in enumerate(example_prompts, 1):
        print(f"{i}. {prompt}")
    
    print("\n" + "-" * 60)
    
    # Get user input
    user_prompt = input("Enter your terrain description (or press Enter for default): ").strip()
    
    if not user_prompt:
        user_prompt = "mountainous terrain with rivers and valleys"
        print(f"Using default prompt: '{user_prompt}'")
    
    print(f"\nGenerating terrain for: '{user_prompt}'")
    print("-" * 60)
    
    try:
        # Step 1: Generate base heightmap using GAN
        print("\nStep 1: Generating base heightmap...")
        heightmap = prompt_to_heightmap_gan(user_prompt, size=256)
        
        # Step 2: Refine with diffusion model
        print("\nStep 2: Refining with diffusion model...")
        refined_terrain = refine_with_diffusion(heightmap, user_prompt)
        
        # Step 3: Visualize results
        print("\nStep 3: Creating visualization...")
        visualize_terrain(refined_terrain, heightmap, user_prompt, 
                         save_path="terrain_output.png")
        
        print("\n" + "=" * 60)
        print("Terrain generation completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during terrain generation: {e}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    main()

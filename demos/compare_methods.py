#!/usr/bin/env python3
"""
Compare GAN vs Diffusion methods for terrain generation
Shows the difference between basic GAN output and refined diffusion output.
"""

import numpy as np
import matplotlib.pyplot as plt
from terrain_prototype import (
    prompt_to_heightmap_gan,
    refine_with_diffusion,
    _basic_enhancement,
    _apply_terrain_colors
)

def compare_terrain_methods():
    """Compare different terrain generation methods."""
    print("=" * 70)
    print("GAN vs Diffusion Terrain Generation Comparison")
    print("=" * 70)
    
    # Test prompt
    prompt = "mountainous terrain with rivers and valleys"
    size = 128
    
    print(f"\nGenerating terrain for: '{prompt}'")
    print(f"Terrain size: {size}x{size}")
    
    # Method 1: Basic GAN generation
    print("\n" + "="*50)
    print("METHOD 1: Basic GAN Generation")
    print("="*50)
    
    heightmap = prompt_to_heightmap_gan(prompt, size=size)
    
    # Method 2: Basic enhancement (fallback when diffusion not available)
    print("\n" + "="*50)
    print("METHOD 2: Basic Enhancement")
    print("="*50)
    
    basic_enhanced = _basic_enhancement(heightmap, prompt)
    
    # Method 3: Try diffusion refinement
    print("\n" + "="*50)
    print("METHOD 3: Diffusion Refinement")
    print("="*50)
    
    try:
        diffusion_refined = refine_with_diffusion(heightmap, prompt)
        diffusion_available = True
        print("✓ Diffusion refinement completed!")
    except Exception as e:
        print(f"⚠️  Diffusion refinement failed: {e}")
        print("   Using basic enhancement instead...")
        diffusion_refined = basic_enhanced
        diffusion_available = False
    
    # Create comparison visualization
    print("\n" + "="*50)
    print("Creating Comparison Visualization")
    print("="*50)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Terrain Generation Methods Comparison: "{prompt}"', fontsize=16, fontweight='bold')
    
    # Row 1: Heightmaps
    axes[0, 0].imshow(heightmap, cmap='terrain', aspect='equal')
    axes[0, 0].set_title('1. GAN Heightmap', fontweight='bold')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].imshow(heightmap, cmap='terrain', aspect='equal')
    axes[0, 1].set_title('2. Enhanced Heightmap', fontweight='bold')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].imshow(heightmap, cmap='terrain', aspect='equal')
    axes[0, 2].set_title('3. Diffusion Heightmap', fontweight='bold')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Final terrain images
    if len(basic_enhanced.shape) == 3:
        axes[1, 0].imshow(basic_enhanced)
    else:
        axes[1, 0].imshow(basic_enhanced, cmap='terrain')
    axes[1, 0].set_title('1. GAN Output (Basic)', fontweight='bold')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].grid(True, alpha=0.3)
    
    if len(basic_enhanced.shape) == 3:
        axes[1, 1].imshow(basic_enhanced)
    else:
        axes[1, 1].imshow(basic_enhanced, cmap='terrain')
    axes[1, 1].set_title('2. Enhanced Output', fontweight='bold')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].grid(True, alpha=0.3)
    
    if len(diffusion_refined.shape) == 3:
        axes[1, 2].imshow(diffusion_refined)
    else:
        axes[1, 2].imshow(diffusion_refined, cmap='terrain')
    axes[1, 2].set_title('3. Diffusion Output', fontweight='bold')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison
    save_path = "terrain_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"✓ GAN Generation: Basic heightmap with terrain features")
    print(f"✓ Basic Enhancement: Color-coded terrain with realistic colors")
    
    if diffusion_available:
        print(f"✓ Diffusion Refinement: High-quality realistic terrain")
        print(f"   - Uses Stable Diffusion 1.5 + ControlNet")
        print(f"   - Converts heightmap to depth map")
        print(f"   - Applies realistic textures and details")
    else:
        print(f"⚠️  Diffusion Refinement: Not available")
        print(f"   - Install with: pip install diffusers transformers accelerate")
        print(f"   - Requires ~2GB of model downloads")
    
    print(f"\n📊 Quality Comparison:")
    print(f"   GAN Output: Basic elevation data")
    print(f"   Enhanced: Realistic colors and basic textures")
    print(f"   Diffusion: Photorealistic terrain with fine details")
    
    return heightmap, basic_enhanced, diffusion_refined

def show_terrain_features():
    """Show how different terrain features are generated."""
    print("\n" + "=" * 70)
    print("TERRAIN FEATURE GENERATION")
    print("=" * 70)
    
    # Test different prompts
    test_prompts = [
        "mountainous terrain",
        "desert landscape", 
        "forest with rivers",
        "coastal cliffs"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Terrain Feature Generation Examples', fontsize=16, fontweight='bold')
    
    for i, prompt in enumerate(test_prompts):
        row = i // 2
        col = i % 2
        
        print(f"\nGenerating: {prompt}")
        heightmap = prompt_to_heightmap_gan(prompt, size=64)
        
        axes[row, col].imshow(heightmap, cmap='terrain', aspect='equal')
        axes[row, col].set_title(f'{prompt.title()}', fontweight='bold')
        axes[row, col].set_xlabel('X')
        axes[row, col].set_ylabel('Y')
        axes[row, col].grid(True, alpha=0.3)
        
        print(f"  ✓ Heightmap generated: {heightmap.shape}")
        print(f"  ✓ Height range: {np.min(heightmap):.3f} to {np.max(heightmap):.3f}")
    
    plt.tight_layout()
    
    # Save features
    save_path = "terrain_features.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFeatures saved to: {save_path}")
    
    plt.show()

def main():
    """Main comparison function."""
    print("Welcome to the Terrain Generation Methods Comparison!")
    print("\nThis script will show you the difference between:")
    print("1. Basic GAN generation")
    print("2. Enhanced terrain with colors")
    print("3. Diffusion-refined realistic terrain")
    
    try:
        # Run comparison
        heightmap, basic, diffusion = compare_terrain_methods()
        
        # Show terrain features
        show_terrain_features()
        
        print("\n" + "=" * 70)
        print("🎉 Comparison completed successfully!")
        print("=" * 70)
        
        print("\nTo run the full interactive prototype:")
        print("python terrain_prototype.py")
        
        print("\nTo run demos with diffusion:")
        print("python demo.py")
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()

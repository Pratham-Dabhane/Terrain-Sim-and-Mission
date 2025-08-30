#!/usr/bin/env python3
"""
GAN + Enhancement Pipeline Explanation
Shows step-by-step how terrain is generated from text prompts.
"""

import numpy as np
import matplotlib.pyplot as plt
from terrain_prototype import (
    prompt_to_heightmap_gan,
    _basic_enhancement,
    _apply_prompt_modifications
)

def explain_pipeline():
    """Explain the terrain generation pipeline step by step."""
    print("=" * 80)
    print("GAN + Enhancement Pipeline Explanation")
    print("=" * 80)
    
    prompt = "mountainous terrain with rivers and valleys"
    size = 128
    
    print(f"\n🎯 Example Prompt: '{prompt}'")
    print(f"📏 Terrain Size: {size}x{size} pixels")
    
    # Step 1: Text Analysis
    print("\n" + "="*60)
    print("STEP 1: Text Analysis & Feature Detection")
    print("="*60)
    
    prompt_lower = prompt.lower()
    detected_features = []
    
    if 'mountain' in prompt_lower or 'peak' in prompt_lower:
        detected_features.append("🏔️ Mountains")
    if 'valley' in prompt_lower or 'canyon' in prompt_lower:
        detected_features.append("🏞️ Valleys")
    if 'river' in prompt_lower or 'water' in prompt_lower:
        detected_features.append("🌊 Rivers")
    if 'forest' in prompt_lower or 'trees' in prompt_lower:
        detected_features.append("🌲 Forest")
    if 'desert' in prompt_lower or 'sand' in prompt_lower:
        detected_features.append("🏜️ Desert")
    
    print("Detected terrain features:")
    for feature in detected_features:
        print(f"  ✓ {feature}")
    
    # Step 2: GAN Generation
    print("\n" + "="*60)
    print("STEP 2: GAN Generation (Base Heightmap)")
    print("="*60)
    
    print("Generating base heightmap using neural network...")
    heightmap = prompt_to_heightmap_gan(prompt, size=size)
    
    print(f"✓ Base heightmap generated: {heightmap.shape}")
    print(f"✓ Height range: {np.min(heightmap):.3f} to {np.max(heightmap):.3f}")
    print(f"✓ Data type: {heightmap.dtype}")
    
    # Step 3: Feature Application
    print("\n" + "="*60)
    print("STEP 3: Feature Application")
    print("="*60)
    
    print("Applying detected terrain features...")
    modified_heightmap = _apply_prompt_modifications(heightmap.copy(), prompt)
    
    # Calculate changes
    changes = modified_heightmap - heightmap
    print(f"✓ Features applied successfully")
    print(f"✓ Maximum height change: {np.max(changes):.3f}")
    print(f"✓ Minimum height change: {np.min(changes):.3f}")
    
    # Step 4: Color Enhancement
    print("\n" + "="*60)
    print("STEP 4: Color Enhancement")
    print("="*60)
    
    print("Applying realistic terrain colors...")
    enhanced_terrain = _basic_enhancement(modified_heightmap, prompt)
    
    print(f"✓ Colors applied successfully")
    print(f"✓ Output shape: {enhanced_terrain.shape}")
    print(f"✓ Color range: {np.min(enhanced_terrain)} to {np.max(enhanced_terrain)}")
    
    # Create visualization
    print("\n" + "="*60)
    print("Creating Pipeline Visualization")
    print("="*60)
    
    create_pipeline_visualization(heightmap, modified_heightmap, enhanced_terrain, prompt)
    
    return heightmap, modified_heightmap, enhanced_terrain

def create_pipeline_visualization(original, modified, enhanced, prompt):
    """Create a visualization showing each step of the pipeline."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Terrain Generation Pipeline: "{prompt}"', fontsize=16, fontweight='bold')
    
    # Row 1: Heightmaps
    axes[0, 0].imshow(original, cmap='terrain', aspect='equal')
    axes[0, 0].set_title('1. Base GAN Heightmap', fontweight='bold')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].imshow(modified, cmap='terrain', aspect='equal')
    axes[0, 1].set_title('2. After Feature Application', fontweight='bold')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].imshow(modified, cmap='terrain', aspect='equal')
    axes[0, 2].set_title('3. Feature-Enhanced Heightmap', fontweight='bold')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Final outputs
    axes[1, 0].imshow(original, cmap='terrain', aspect='equal')
    axes[1, 0].set_title('4. Original (for comparison)', fontweight='bold')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].grid(True, alpha=0.3)
    
    if len(enhanced.shape) == 3:
        axes[1, 1].imshow(enhanced)
    else:
        axes[1, 1].imshow(enhanced, cmap='terrain')
    axes[1, 1].set_title('5. Final Colored Terrain', fontweight='bold')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 3D view
    ax3d = fig.add_subplot(2, 3, 6, projection='3d')
    y, x = np.mgrid[0:modified.shape[0]:1, 0:modified.shape[1]:1]
    ax3d.plot_surface(x, y, modified, cmap='terrain', alpha=0.8)
    ax3d.set_title('6. 3D Terrain View', fontweight='bold')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Height')
    
    plt.tight_layout()
    
    # Save visualization
    save_path = "pipeline_explanation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Pipeline visualization saved to: {save_path}")
    
    # Show plot
    print("Displaying pipeline visualization...")
    plt.show()
    
    # Close plot
    plt.close()

def explain_gan_architecture():
    """Explain the GAN architecture used."""
    print("\n" + "=" * 80)
    print("GAN Architecture Details")
    print("=" * 80)
    
    print("🤖 Neural Network Structure:")
    print("  • Input: Random latent vector (512 dimensions)")
    print("  • Layer 1: 512 → 512 neurons (LeakyReLU)")
    print("  • Layer 2: 512 → 1024 neurons (LeakyReLU)")
    print("  • Layer 3: 1024 → 2048 neurons (LeakyReLU)")
    print("  • Output: 2048 → (128×128) heightmap (Sigmoid)")
    
    print("\n🔧 Key Features:")
    print("  • LeakyReLU activation prevents dead neurons")
    print("  • Sigmoid output ensures heights between 0-1")
    print("  • Random latent vectors create unique terrains")
    print("  • Weight initialization follows StyleGAN2 principles")
    
    print("\n📊 Output Properties:")
    print("  • Grayscale heightmap (0 = lowest, 1 = highest)")
    print("  • Smooth, continuous elevation changes")
    print("  • Realistic terrain-like patterns")
    print("  • Configurable size (64x64, 128x128, 256x256)")

def explain_enhancement_pipeline():
    """Explain the enhancement pipeline."""
    print("\n" + "=" * 80)
    print("Enhancement Pipeline Details")
    print("=" * 80)
    
    print("🎨 Color Mapping:")
    print("  • Water (low): Blue [0.2, 0.4, 0.8]")
    print("  • Sand (low-mid): Sand [0.9, 0.8, 0.6]")
    print("  • Grass (mid): Green [0.3, 0.6, 0.2]")
    print("  • Forest (mid-high): Dark Green [0.2, 0.4, 0.1]")
    print("  • Rock (high): Gray [0.5, 0.5, 0.5]")
    print("  • Snow (highest): White [0.9, 0.9, 0.9]")
    
    print("\n🔧 Enhancement Steps:")
    print("  • Contrast adjustment (power law: 0.8)")
    print("  • Texture addition (5% random noise)")
    print("  • Height-based color assignment")
    print("  • RGB conversion and normalization")
    
    print("\n📈 Quality Improvements:")
    print("  • Realistic terrain appearance")
    print("  • Height-based color coding")
    print("  • Enhanced visual appeal")
    print("  • Professional-looking output")

def main():
    """Main explanation function."""
    print("Welcome to the Terrain Generation Pipeline Explanation!")
    print("\nThis script will show you exactly how:")
    print("1. Text prompts are analyzed")
    print("2. GAN generates base heightmaps")
    print("3. Terrain features are applied")
    print("4. Colors and enhancements are added")
    
    try:
        # Run pipeline explanation
        original, modified, enhanced = explain_pipeline()
        
        # Explain technical details
        explain_gan_architecture()
        explain_enhancement_pipeline()
        
        print("\n" + "=" * 80)
        print("🎉 Pipeline Explanation Completed!")
        print("=" * 80)
        
        print("\nTo run the interactive prototype:")
        print("python terrain_prototype.py")
        
        print("\nTo run the simple demo:")
        print("python simple_demo.py")
        
    except Exception as e:
        print(f"\n❌ Error during explanation: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()

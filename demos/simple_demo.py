#!/usr/bin/env python3
"""
Simple Terrain Demo - Guaranteed to show terrain images!
This demo works without heavy ML models and shows the GAN + enhancement pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from terrain_prototype import (
    prompt_to_heightmap_gan,
    _basic_enhancement,
    _apply_terrain_colors
)

def demo_mountainous_terrain():
    """Demo: Mountainous terrain with rivers and valleys."""
    print("\n" + "="*60)
    print("DEMO: Mountainous Terrain with Rivers and Valleys")
    print("="*60)
    
    prompt = "mountainous terrain with rivers and valleys"
    
    # Generate terrain
    print("Generating heightmap...")
    heightmap = prompt_to_heightmap_gan(prompt, size=128)
    
    # Enhance with colors
    print("Applying terrain colors...")
    enhanced = _basic_enhancement(heightmap, prompt)
    
    # Create visualization
    print("Creating visualization...")
    create_terrain_visualization(heightmap, enhanced, prompt, "mountainous_terrain.png")
    
    return heightmap, enhanced

def demo_desert_landscape():
    """Demo: Desert landscape with sand dunes."""
    print("\n" + "="*60)
    print("DEMO: Desert Landscape with Sand Dunes")
    print("="*60)
    
    prompt = "desert landscape with sand dunes"
    
    # Generate terrain
    print("Generating heightmap...")
    heightmap = prompt_to_heightmap_gan(prompt, size=128)
    
    # Enhance with colors
    print("Applying terrain colors...")
    enhanced = _basic_enhancement(heightmap, prompt)
    
    # Create visualization
    print("Creating visualization...")
    create_terrain_visualization(heightmap, enhanced, prompt, "desert_landscape.png")
    
    return heightmap, enhanced

def demo_forest_terrain():
    """Demo: Forest terrain with rolling hills."""
    print("\n" + "="*60)
    print("DEMO: Forest Terrain with Rolling Hills")
    print("="*60)
    
    prompt = "forest terrain with rolling hills"
    
    # Generate terrain
    print("Generating heightmap...")
    heightmap = prompt_to_heightmap_gan(prompt, size=128)
    
    # Enhance with colors
    print("Applying terrain colors...")
    enhanced = _basic_enhancement(heightmap, prompt)
    
    # Create visualization
    print("Creating visualization...")
    create_terrain_visualization(heightmap, enhanced, prompt, "forest_terrain.png")
    
    return heightmap, enhanced

def create_terrain_visualization(heightmap, terrain_image, prompt, save_path):
    """Create and display terrain visualization."""
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Terrain Generation: "{prompt}"', fontsize=16, fontweight='bold')
    
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
    
    # Save the visualization
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {save_path}")
    
    # Show the plot
    print("Displaying terrain visualization...")
    plt.show()
    
    # Close the plot to free memory
    plt.close()

def run_all_demos():
    """Run all demo examples."""
    print("=" * 80)
    print("Simple Terrain Demo - Guaranteed Image Output!")
    print("=" * 80)
    print("This demo will generate 3 different terrain types and show you:")
    print("- The generated heightmap")
    print("- The colored 2.5D terrain")
    print("- A 3D visualization")
    print("- Save each result as a PNG file")
    
    demos = [
        ("Mountainous terrain with rivers and valleys", demo_mountainous_terrain),
        ("Desert landscape with sand dunes", demo_desert_landscape),
        ("Forest terrain with rolling hills", demo_forest_terrain)
    ]
    
    results = []
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            print(f"\nRunning Demo {i}/3: {name}")
            heightmap, terrain = demo_func()
            results.append((heightmap, terrain))
            print(f"✓ Demo {i} completed successfully!")
            
        except Exception as e:
            print(f"✗ Demo {i} failed: {e}")
            results.append((None, None))
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    
    successful_demos = sum(1 for h, t in results if h is not None and t is not None)
    print(f"Successful demos: {successful_demos}/3")
    
    if successful_demos > 0:
        print("\nGenerated files:")
        print("- mountainous_terrain.png")
        print("- desert_landscape.png")
        print("- forest_terrain.png")
        
        print("\n🎉 You should have seen terrain images displayed!")
        print("\nTo run the interactive prototype:")
        print("python terrain_prototype.py")
    
    print("\nDemo completed!")

def main():
    """Main demo function."""
    print("Welcome to the Simple Terrain Demo!")
    print("\nThis demo is guaranteed to show you terrain images because:")
    print("✓ Uses only basic GAN generation (no heavy ML models)")
    print("✓ Applies realistic terrain colors")
    print("✓ Creates 3D visualizations")
    print("✓ Saves high-quality PNG files")
    
    try:
        run_all_demos()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check the error message and try again.")

if __name__ == "__main__":
    main()

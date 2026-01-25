"""Biome Visualization Demo

Demonstrates Phase 3 terrain analysis and biome mask generation.
This script generates a terrain, runs the full pipeline (macro + erosion + biome analysis),
and visualizes all terrain attributes and biome masks.

Usage:
    python biome_visualization_demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging

from pipeline.procedural_noise_utils import generate_procedural_heightmap, NoiseParams
from pipeline import terrain_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_biome_analysis(heightmap: np.ndarray, analysis: dict, output_dir: str = "Output/biome_demo"):
    """Create comprehensive visualization of terrain analysis results.
    
    Args:
        heightmap: The terrain heightmap
        analysis: Results from terrain_analysis.analyze_terrain()
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Side-by-side comparison of elevation and dominant biome
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(heightmap, cmap='terrain')
    axes[0].set_title('Terrain Elevation', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Create colormap for biomes
    biome_names = sorted(analysis['biome_masks'].keys())
    biome_colors = [
        [0.05, 0.05, 0.4],    # water - dark blue
        [0.3, 0.6, 0.3],      # forest - green
        [0.6, 0.7, 0.4],      # grassland - yellow-green
        [0.7, 0.3, 0.2],      # rock - brown-red
        [0.6, 0.6, 0.5],      # scree - gray-brown
        [0.9, 0.95, 1.0],     # snow - white
        [0.3, 0.5, 0.4],      # wetlands - teal
    ]
    
    n_biomes = len(biome_names)
    while len(biome_colors) < n_biomes:
        biome_colors.append([0.5, 0.5, 0.5])
    
    cmap = ListedColormap(biome_colors[:n_biomes])
    
    dominant_idx, dominant_strength = analysis['dominant_biome']
    im = axes[1].imshow(dominant_idx, cmap=cmap, vmin=0, vmax=n_biomes-1)
    axes[1].set_title('Dominant Biome Classification', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add colorbar with biome names
    cbar = plt.colorbar(im, ax=axes[1], ticks=range(n_biomes), fraction=0.046, pad=0.04)
    cbar.set_ticklabels([name.capitalize() for name in biome_names])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'elevation_vs_biomes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved elevation vs biomes comparison to {output_dir}")
    
    # 2. Individual biome masks grid
    n_biomes = len(biome_names)
    cols = 3
    rows = (n_biomes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle('Individual Biome Masks (Continuous [0,1])', fontsize=16, fontweight='bold')
    
    axes = axes.flatten() if n_biomes > 1 else [axes]
    
    for idx, biome_name in enumerate(biome_names):
        im = axes[idx].imshow(analysis['biome_masks'][biome_name], cmap='viridis', vmin=0, vmax=1)
        axes[idx].set_title(f'{biome_name.capitalize()}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_biomes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_biome_masks.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved individual biome masks to {output_dir}")
    
    # 3. Terrain attributes overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Terrain Attributes Used for Biome Classification', fontsize=16, fontweight='bold')
    
    im1 = axes[0, 0].imshow(analysis['elevation'], cmap='terrain')
    axes[0, 0].set_title('Elevation', fontsize=12)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 1].imshow(analysis['slope'], cmap='hot')
    axes[0, 1].set_title('Slope', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[0, 2].imshow(analysis['convexity'], cmap='RdYlGn')
    axes[0, 2].set_title('Convexity (Ridges/Peaks)', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im4 = axes[1, 0].imshow(analysis['concavity'], cmap='Blues')
    axes[1, 0].set_title('Concavity (Valleys/Basins)', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im5 = axes[1, 1].imshow(analysis['aspect'], cmap='hsv')
    axes[1, 1].set_title('Aspect (Slope Direction)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Normalize distance for visualization
    dist_vis = np.clip(analysis['distance_to_water'] / 50.0, 0, 1)
    im6 = axes[1, 2].imshow(dist_vis, cmap='Blues_r')
    axes[1, 2].set_title('Distance to Water', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'terrain_attributes_detailed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved terrain attributes overview to {output_dir}")
    
    # 4. Biome coverage statistics
    print("\n" + "="*60)
    print("BIOME COVERAGE STATISTICS")
    print("="*60)
    
    total_pixels = dominant_idx.size
    
    for idx, biome_name in enumerate(biome_names):
        count = np.sum(dominant_idx == idx)
        percentage = (count / total_pixels) * 100
        avg_strength = analysis['biome_masks'][biome_name].mean()
        max_strength = analysis['biome_masks'][biome_name].max()
        
        print(f"\n{biome_name.upper():15s}")
        print(f"  Dominant pixels: {count:6d} ({percentage:5.2f}%)")
        print(f"  Avg mask value:  {avg_strength:6.4f}")
        print(f"  Max mask value:  {max_strength:6.4f}")
    
    print("\n" + "="*60)


def main():
    """Main demo function."""
    
    print("="*60)
    print("BIOME VISUALIZATION DEMO - Phase 3")
    print("="*60)
    print("\nThis demo generates terrain with macro structure, erosion,")
    print("and biome analysis, then visualizes all results.\n")
    
    # Ensure biome analysis is enabled
    original_flag = terrain_analysis.ENABLE_BIOMES
    terrain_analysis.ENABLE_BIOMES = True
    
    try:
        # Generate terrain with full pipeline
        shape = (512, 512)
        seed = 42
        
        print(f"\nGenerating terrain: {shape[0]}x{shape[1]} pixels, seed={seed}")
        print("-" * 60)
        
        params = NoiseParams(
            scale=100.0,
            octaves=6,
            persistence=0.55,
            lacunarity=2.1,
            mountain_weight=0.7,
            valley_weight=0.3,
            river_strength=0.25,
            seed=seed
        )
        
        # Generate with debug outputs (this will trigger biome analysis)
        debug_dir = "Output/biome_demo/debug"
        heightmap = generate_procedural_heightmap(shape, params, debug_dir=debug_dir)
        
        print("\n" + "-" * 60)
        print("Running standalone terrain analysis...")
        print("-" * 60)
        
        # Run terrain analysis separately for visualization
        analysis = terrain_analysis.analyze_terrain(heightmap, debug_dir=None)
        
        print(f"\nTerrain analysis complete!")
        print(f"  - Computed {len(analysis['biome_masks'])} biome masks")
        print(f"  - Generated terrain attribute maps")
        
        # Create visualizations
        print("\n" + "-" * 60)
        print("Creating visualizations...")
        print("-" * 60)
        
        visualize_biome_analysis(heightmap, analysis)
        
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print(f"\nAll visualizations saved to: Output/biome_demo/")
        print(f"Debug outputs saved to: {debug_dir}")
        print("\nGenerated files:")
        print("  - elevation_vs_biomes.png       (side-by-side comparison)")
        print("  - individual_biome_masks.png    (all biome masks)")
        print("  - terrain_attributes_detailed.png (slope, curvature, etc.)")
        print("  - debug/terrain_attributes.png  (from pipeline)")
        print("  - debug/biome_masks.png         (from pipeline)")
        print("  - debug/dominant_biome.png      (from pipeline)")
        
    finally:
        # Restore original flag
        terrain_analysis.ENABLE_BIOMES = original_flag


if __name__ == "__main__":
    main()

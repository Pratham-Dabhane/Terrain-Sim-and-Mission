"""
Generate terrain using DEM-calibrated parameters from Grand Canyon

Usage:
    python generate_calibrated_terrain.py
    python generate_calibrated_terrain.py --seed 42
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline.procedural_noise_utils import generate_procedural_heightmap, NoiseParams
from pipeline.advanced_terrain_renderer import AdvancedTerrainRenderer

# CALIBRATED PARAMETERS from Grand Canyon DEM (n36_w113)
GRAND_CANYON_PARAMS = NoiseParams(
    scale=300.0,      # Calibrated from DEM
    octaves=6,
    persistence=0.5,
    lacunarity=2.0,
    seed=42,          # Default seed (can be changed)
    mountain_weight=0.80,
    valley_weight=0.20,
    river_strength=0.3,
    river_frequency=0.05
)


def main():
    parser = argparse.ArgumentParser(description="Generate terrain with DEM-calibrated parameters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for variation")
    parser.add_argument("--size", type=int, default=512, help="Terrain size (512 or 1024)")
    parser.add_argument("--interactive-3d", action="store_true", help="Launch interactive 3D viewer")
    parser.add_argument("--output", type=str, default="Output/calibrated_terrain", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENERATING GRAND CANYON-CALIBRATED TERRAIN")
    print("="*70)
    print(f"Parameters (from DEM calibration):")
    print(f"  Scale: {GRAND_CANYON_PARAMS.scale}")
    print(f"  Octaves: {GRAND_CANYON_PARAMS.octaves}")
    print(f"  Persistence: {GRAND_CANYON_PARAMS.persistence}")
    print(f"  Seed: {args.seed}")
    print(f"  Size: {args.size}x{args.size}")
    print()
    
    # Update seed
    params = NoiseParams(
        scale=GRAND_CANYON_PARAMS.scale,
        octaves=GRAND_CANYON_PARAMS.octaves,
        persistence=GRAND_CANYON_PARAMS.persistence,
        lacunarity=GRAND_CANYON_PARAMS.lacunarity,
        seed=args.seed,  # Use the user-provided seed
        mountain_weight=GRAND_CANYON_PARAMS.mountain_weight,
        valley_weight=GRAND_CANYON_PARAMS.valley_weight,
        river_strength=GRAND_CANYON_PARAMS.river_strength,
        river_frequency=GRAND_CANYON_PARAMS.river_frequency
    )
    
    # Generate heightmap
    print("Generating procedural heightmap...")
    heightmap = generate_procedural_heightmap(
        shape=(args.size, args.size),
        params=params,
        debug_dir=str(output_dir)
    )
    
    print(f"✓ Heightmap generated: range=[{heightmap.min():.3f}, {heightmap.max():.3f}]")
    
    # Save heightmap
    heightmap_path = output_dir / "heightmap.npy"
    np.save(heightmap_path, heightmap)
    print(f"✓ Saved heightmap: {heightmap_path}")
    
    # Save heightmap image
    plt.figure(figsize=(10, 10))
    plt.imshow(heightmap, cmap='terrain', interpolation='bilinear')
    plt.colorbar(label='Elevation')
    plt.title(f'Grand Canyon-Calibrated Terrain (seed={args.seed})')
    plt.axis('off')
    
    img_path = output_dir / "heightmap.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization: {img_path}")
    
    # Generate 3D mesh and save as image
    if args.interactive_3d:
        print("\nGenerating 3D visualization...")
        
        # Use advanced renderer for smooth mesh
        renderer = AdvancedTerrainRenderer()
        
        # Create dummy texture (same size as heightmap for simple visualization)
        enhanced_texture = np.stack([heightmap] * 3, axis=-1)  # RGB from heightmap
        
        # Save 3D render as image (not interactive)
        render_3d_path = output_dir / "terrain_3d_render.png"
        renderer.create_photorealistic_visualization(
            heightmap,
            enhanced_texture,
            terrain_prompt=f"Grand Canyon-Calibrated Terrain (seed={args.seed})",
            output_path=str(render_3d_path)
        )
        print(f"✓ Saved 3D render: {render_3d_path}")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"\nOutput saved to: {output_dir}")
    print("\nTo generate different variations, use different seeds:")
    print(f"  python generate_calibrated_terrain.py --seed 1 --interactive-3d")
    print(f"  python generate_calibrated_terrain.py --seed 2 --interactive-3d")
    print(f"  python generate_calibrated_terrain.py --seed 3 --interactive-3d")


if __name__ == "__main__":
    main()

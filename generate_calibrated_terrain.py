"""
Generate terrain using DEM-calibrated parameters from Grand Canyon

Usage:
    python generate_calibrated_terrain.py
    python generate_calibrated_terrain.py --seed 42
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path

from pipeline.procedural_noise_utils import generate_procedural_heightmap, NoiseParams
from pipeline.advanced_terrain_renderer import AdvancedTerrainRenderer
from pipeline.terrain_texture_mapper import colorize_by_elevation_and_slope

# CALIBRATED PARAMETERS from Grand Canyon DEM (n36_w113)
GRAND_CANYON_PARAMS = NoiseParams(
    scale=140.0,      # Lower scale to reduce broad banding and add terrain detail
    octaves=7,        # Slightly more multi-scale detail
    persistence=0.58, # Slightly stronger high-frequency contribution
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
    parser.add_argument("--remaster-top", action="store_true", help="Optional SD/ControlNet remaster for top texture")
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
        
        # Use terrain color mapping so top surface is not grayscale/dim.
        enhanced_texture = np.array(colorize_by_elevation_and_slope(heightmap))

        # Optional final realism pass: remaster the top texture with SD + ControlNet.
        if args.remaster_top:
            try:
                from pipeline.remaster_sd_controlnet import TerrainRemaster

                print("Applying optional SD/ControlNet top remaster...")
                remaster = TerrainRemaster(
                    controlnet_type="depth",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    use_fp16=False,
                    enable_cpu_offload=False,
                )
                remastered_image, _ = remaster.remaster_heightmap(
                    heightmap=heightmap,
                    prompt=f"realistic terrain texture, alpine rock and soil, seed {args.seed}",
                    num_inference_steps=16,
                    guidance_scale=6.0,
                    controlnet_conditioning_scale=0.65,
                    output_size=(args.size, args.size),
                    seed=args.seed,
                    preserve_geometry=True,
                )

                # Some diffusers builds can return a list instead of a single PIL image.
                if isinstance(remastered_image, list):
                    if len(remastered_image) == 0:
                        raise ValueError("Remaster returned an empty image list")
                    remastered_image = remastered_image[0]

                # Convert remaster output to a concrete image/array representation.
                if isinstance(remastered_image, np.ndarray):
                    remaster_arr = remastered_image
                else:
                    remaster_arr = np.array(remastered_image)

                # Normalize common remaster output ranges to uint8 safely.
                if remaster_arr.dtype != np.uint8:
                    remaster_arr = remaster_arr.astype(np.float32)
                    if remaster_arr.max() <= 1.0 and remaster_arr.min() >= 0.0:
                        remaster_arr = remaster_arr * 255.0
                    remaster_arr = np.clip(remaster_arr, 0.0, 255.0).astype(np.uint8)

                # Reject invalid/degenerate outputs (e.g., all-black or NaN/Inf collapsed).
                if (not np.isfinite(remaster_arr).all()) or remaster_arr.max() <= 2:
                    raise ValueError("Remaster produced invalid or near-empty texture")

                enhanced_texture = remaster_arr
                remaster_image_to_save = Image.fromarray(remaster_arr)

                remaster_path = output_dir / "top_texture_remastered.png"
                remaster_image_to_save.save(remaster_path)
                print(f"✓ Saved remastered top texture: {remaster_path}")
            except Exception as e:
                print(f"⚠ Remaster step skipped (fallback to procedural texture): {e}")
        
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

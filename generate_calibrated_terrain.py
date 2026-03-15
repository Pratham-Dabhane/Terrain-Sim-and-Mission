"""
Generate terrain using DEM-calibrated parameters from Grand Canyon

Usage:
    python generate_calibrated_terrain.py
    python generate_calibrated_terrain.py --seed 42
    python generate_calibrated_terrain.py --seed 2 --size 320 --interactive-3d --mission
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
from pipeline.cost_map import cost_map
from pipeline.planner import find_path, calculate_path_statistics, overlay_path_on_terrain

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
    parser.add_argument("--mission", action="store_true", help="Enable mission planning: click start/end on heightmap")
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
    
    # ══════════════════════════════════════════════════════════════
    # MISSION PLANNING: interactive start/end selection → A* path
    # ══════════════════════════════════════════════════════════════
    mission_path = []  # Will be populated if --mission is used
    if args.mission:
        print("\n" + "─"*50)
        print("MISSION PLANNING")
        print("─"*50)
        print("Click START point on the heightmap, then click END point.")
        print("(Close the window after clicking both points)")
        
        # Show heightmap and let user click start and end
        fig_pick, ax_pick = plt.subplots(figsize=(10, 10))
        ax_pick.imshow(heightmap, cmap='terrain', interpolation='bilinear')
        ax_pick.set_title('Click START point, then click END point\n(2 clicks total)', fontsize=14)
        ax_pick.axis('off')
        
        # Blocking call — waits for 2 clicks
        clicked_points = plt.ginput(2, timeout=120)
        plt.close(fig_pick)
        
        if len(clicked_points) < 2:
            print("⚠ Less than 2 points selected — skipping mission planning.")
        else:
            # ginput returns (x, y) = (col, row)
            start_col, start_row = int(clicked_points[0][0]), int(clicked_points[0][1])
            end_col, end_row = int(clicked_points[1][0]), int(clicked_points[1][1])
            
            # Clamp to valid range
            start_row = np.clip(start_row, 0, heightmap.shape[0] - 1)
            start_col = np.clip(start_col, 0, heightmap.shape[1] - 1)
            end_row = np.clip(end_row, 0, heightmap.shape[0] - 1)
            end_col = np.clip(end_col, 0, heightmap.shape[1] - 1)
            
            start = (int(start_row), int(start_col))
            goal = (int(end_row), int(end_col))
            
            print(f"  Start: row={start[0]}, col={start[1]}  (elevation={heightmap[start[0], start[1]]:.3f})")
            print(f"  Goal:  row={goal[0]}, col={goal[1]}  (elevation={heightmap[goal[0], goal[1]]:.3f})")
            
            # Generate cost map
            print("\nComputing cost map...")
            terrain_params = {
                'water_level': 0.15,
                'elevation_scale': 1.0,
                'roughness': 0.5,
                'biome_type': 'mountain'
            }
            cost_grid = cost_map(heightmap, terrain_params)
            
            cost_path = output_dir / "cost_map.png"
            plt.figure(figsize=(10, 10))
            plt.imshow(cost_grid, cmap='hot', interpolation='bilinear')
            plt.colorbar(label='Traversal Cost')
            plt.title('Cost Map')
            plt.axis('off')
            plt.savefig(cost_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved cost map: {cost_path}")
            
            # Run A* pathfinding
            print("Running A* pathfinding...")
            mission_path = find_path(cost_grid, start, goal)
            
            if not mission_path:
                print("⚠ No path found between the selected points!")
            else:
                # Print path statistics
                stats = calculate_path_statistics(mission_path, cost_grid, heightmap)
                print(f"\n{'─'*40}")
                print(f"  PATH FOUND")
                print(f"{'─'*40}")
                print(f"  Waypoints:        {stats['waypoints']}")
                print(f"  Total cost:       {stats['total_cost']:.2f}")
                print(f"  Elevation gain:   {stats['elevation_gain']:.4f}")
                print(f"  Elevation loss:   {stats['elevation_loss']:.4f}")
                print(f"  Start elevation:  {stats['start_elevation']:.3f}")
                print(f"  Goal elevation:   {stats['goal_elevation']:.3f}")
                print(f"  Path efficiency:  {stats['path_efficiency']:.3f}")
                print(f"{'─'*40}")
                
                # Save 2D path overlay
                path_2d_img = output_dir / "mission_path_2d.png"
                overlay_path_on_terrain(
                    heightmap, mission_path,
                    output_path=str(path_2d_img),
                    title=f"Mission Path (seed={args.seed})"
                )
                print(f"✓ Saved 2D path overlay: {path_2d_img}")
    
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
                    use_fp16=True,   # fp16 halves RAM usage; NaN was from autocast (now disabled)
                    enable_cpu_offload=torch.cuda.is_available(),  # Essential: frees VRAM by moving layers to CPU
                )

                # Stronger negative prompt to block map/diagram hallucinations.
                _neg = (
                    "abstract art, neon, glow, lava, fire, surreal patterns, checkerboard, cartoon, "
                    "text, symbols, high contrast posterization, "
                    "map, topographic map, diagram, labels, grid lines, illustration, "
                    "drawing, painting, watercolor, sketch"
                )

                remastered_image, _ = remaster.remaster_heightmap(
                    heightmap=heightmap,
                    prompt=(
                        "realistic satellite-style terrain texture, alpine rock, soil, sparse vegetation, "
                        "natural erosion patterns, physically plausible shading"
                    ),
                    negative_prompt=_neg,
                    num_inference_steps=28,
                    guidance_scale=12.0,         # High guidance = strong prompt adherence
                    controlnet_conditioning_scale=1.3,  # Strong depth conditioning
                    output_size=(args.size, args.size),
                    seed=args.seed,
                    preserve_geometry=True,
                )

                # generate() now always returns a single PIL image (not a list).
                # Convert to numpy for validation & saving.
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
                    # Save the failed output for debugging before raising.
                    _dbg = output_dir / "_debug_failed_remaster.png"
                    try:
                        Image.fromarray(remaster_arr).save(_dbg)
                        print(f"  [debug] Failed remaster saved to {_dbg}")
                    except Exception:
                        pass
                    raise ValueError("Remaster produced invalid or near-empty texture")

                # ── Structural quality gate ──────────────────────────────
                # Compare the luminance of the remastered output against
                # the original heightmap.  If the two are structurally
                # uncorrelated, the remaster has hallucinated (e.g. map,
                # diagram, abstract art) and should be rejected.
                from skimage.metrics import structural_similarity as ssim
                try:
                    import cv2 as _cv2
                    _lum = _cv2.cvtColor(remaster_arr, _cv2.COLOR_RGB2GRAY)
                    _hm_u8 = (np.clip(heightmap, 0, 1) * 255).astype(np.uint8)
                    # Resize heightmap to match remaster dimensions.
                    if _hm_u8.shape != _lum.shape:
                        _hm_u8 = _cv2.resize(_hm_u8, (_lum.shape[1], _lum.shape[0]),
                                             interpolation=_cv2.INTER_LINEAR)
                    _score = ssim(_hm_u8, _lum)
                    print(f"  Structural similarity (SSIM) to heightmap: {_score:.3f}")
                    if _score < 0.10:
                        # Save for debugging before rejecting.
                        _dbg = output_dir / "_debug_failed_remaster.png"
                        try:
                            Image.fromarray(remaster_arr).save(_dbg)
                            print(f"  [debug] Failed remaster saved to {_dbg}")
                        except Exception:
                            pass
                        raise ValueError(
                            f"Remaster failed quality gate: SSIM={_score:.3f} < 0.10 "
                            f"(output does not correlate with terrain structure)"
                        )
                except ImportError:
                    print("  ⚠ skimage not installed – skipping SSIM quality gate")
                # ─────────────────────────────────────────────────────────

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
            output_path=str(render_3d_path),
            mission_path=mission_path if mission_path else None
        )
        print(f"✓ Saved 3D render: {render_3d_path}")

        # Open interactive 3D viewer (drag to rotate, scroll to zoom, right-drag to pan)
        print("\n🎮 Opening interactive 3D viewer... (press 'q' to close)")
        renderer.create_interactive_photorealistic_visualization(
            heightmap,
            enhanced_texture,
            terrain_prompt=f"Grand Canyon-Calibrated Terrain (seed={args.seed})",
            mission_path=mission_path if mission_path else None
        )
    
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

"""DEM Calibration Demo

Demonstrates Phase 4: DEM-based parameter calibration.

This script:
1. Creates a synthetic "reference DEM" (or loads a real one)
2. Generates baseline procedural terrain
3. Computes and compares statistics
4. Calibrates parameters to match DEM
5. Shows before/after comparison

Usage:
    # With synthetic DEM (for testing)
    python dem_calibration_demo.py

    # With real DEM file
    python dem_calibration_demo.py --dem path/to/real_dem.tif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from pathlib import Path

from pipeline.procedural_noise_utils import generate_procedural_heightmap, NoiseParams
from pipeline import dem_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_dem(shape=(512, 512), seed=123) -> np.ndarray:
    """Create a synthetic DEM for testing (mountains with specific characteristics).
    
    This simulates a real-world DEM with known statistical properties.
    """
    logger.info(f"Creating synthetic reference DEM: {shape}")
    
    # Use a different seed and parameters than default to create statistical mismatch
    reference_params = NoiseParams(
        scale=120.0,        # Different from default 100
        octaves=7,          # Different from default 6
        persistence=0.58,   # Different from default 0.5
        lacunarity=2.0,
        seed=seed,
        mountain_weight=0.75,
        valley_weight=0.25,
        river_strength=0.15
    )
    
    dem = generate_procedural_heightmap(shape, reference_params, debug_dir=None)
    
    logger.info(f"✓ Synthetic DEM created: range=[{dem.min():.3f}, {dem.max():.3f}]")
    return dem


def visualize_comparison(
    dem: np.ndarray,
    dem_stats: dem_analysis.TerrainStatistics,
    before_terrain: np.ndarray,
    before_stats: dem_analysis.TerrainStatistics,
    after_terrain: np.ndarray,
    after_stats: dem_analysis.TerrainStatistics,
    output_dir: str
):
    """Create comprehensive before/after calibration visualization."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Three-way terrain comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('DEM Calibration: Terrain Comparison', fontsize=16, fontweight='bold')
    
    axes[0].imshow(dem, cmap='terrain')
    axes[0].set_title('Reference DEM', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(before_terrain, cmap='terrain')
    axes[1].set_title('Before Calibration', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(after_terrain, cmap='terrain')
    axes[2].set_title('After Calibration', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'terrain_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Statistical comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Comparison: DEM vs Generated', fontsize=16, fontweight='bold')
    
    # Elevation histograms
    axes[0, 0].stairs(dem_stats.elevation_histogram, dem_stats.elevation_bins,
                     label='Reference DEM', alpha=0.7, fill=True, color='blue', linewidth=2)
    axes[0, 0].stairs(before_stats.elevation_histogram, before_stats.elevation_bins,
                     label='Before Calibration', alpha=0.6, fill=True, color='red', linewidth=1.5)
    axes[0, 0].stairs(after_stats.elevation_histogram, after_stats.elevation_bins,
                     label='After Calibration', alpha=0.6, fill=True, color='green', linewidth=1.5)
    axes[0, 0].set_xlabel('Elevation', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Elevation Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Slope histograms
    axes[0, 1].stairs(dem_stats.slope_histogram, dem_stats.slope_bins,
                     label='Reference DEM', alpha=0.7, fill=True, color='blue', linewidth=2)
    axes[0, 1].stairs(before_stats.slope_histogram, before_stats.slope_bins,
                     label='Before Calibration', alpha=0.6, fill=True, color='red', linewidth=1.5)
    axes[0, 1].stairs(after_stats.slope_histogram, after_stats.slope_bins,
                     label='After Calibration', alpha=0.6, fill=True, color='green', linewidth=1.5)
    axes[0, 1].set_xlabel('Slope', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Slope Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Metric comparison
    metric_names = ['Elev\nMean', 'Elev\nStd', 'Slope\nMean', 'Drainage\nDensity', 'Roughness']
    dem_values = [
        dem_stats.elevation_mean,
        dem_stats.elevation_std,
        dem_stats.slope_mean,
        dem_stats.drainage_density,
        dem_stats.roughness
    ]
    before_values = [
        before_stats.elevation_mean,
        before_stats.elevation_std,
        before_stats.slope_mean,
        before_stats.drainage_density,
        before_stats.roughness
    ]
    after_values = [
        after_stats.elevation_mean,
        after_stats.elevation_std,
        after_stats.slope_mean,
        after_stats.drainage_density,
        after_stats.roughness
    ]
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    axes[1, 0].bar(x - width, dem_values, width, label='Reference DEM', alpha=0.8, color='blue')
    axes[1, 0].bar(x, before_values, width, label='Before', alpha=0.8, color='red')
    axes[1, 0].bar(x + width, after_values, width, label='After', alpha=0.8, color='green')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metric_names, fontsize=10)
    axes[1, 0].set_ylabel('Value', fontsize=11)
    axes[1, 0].set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Error reduction
    before_metrics = dem_analysis.compare_terrain_statistics(dem_stats, before_stats)
    after_metrics = dem_analysis.compare_terrain_statistics(dem_stats, after_stats)
    
    error_names = ['Elev\nMean', 'Slope\nMean', 'Drainage', 'Roughness', 'Overall']
    before_errors = [
        before_metrics['elevation_mean_error'],
        before_metrics['slope_mean_error'],
        before_metrics['drainage_density_error'],
        before_metrics['roughness_error'],
        before_metrics['overall_error']
    ]
    after_errors = [
        after_metrics['elevation_mean_error'],
        after_metrics['slope_mean_error'],
        after_metrics['drainage_density_error'],
        after_metrics['roughness_error'],
        after_metrics['overall_error']
    ]
    
    x_err = np.arange(len(error_names))
    axes[1, 1].bar(x_err - width/2, before_errors, width, label='Before', alpha=0.8, color='red')
    axes[1, 1].bar(x_err + width/2, after_errors, width, label='After', alpha=0.8, color='green')
    axes[1, 1].set_xticks(x_err)
    axes[1, 1].set_xticklabels(error_names, fontsize=10)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=11)
    axes[1, 1].set_title('Error Reduction (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved statistical comparison to {output_dir}")


def print_statistics_report(
    dem_stats: dem_analysis.TerrainStatistics,
    before_stats: dem_analysis.TerrainStatistics,
    after_stats: dem_analysis.TerrainStatistics,
    initial_params: NoiseParams,
    calibrated_params: NoiseParams
):
    """Print comprehensive statistics report."""
    
    print("\n" + "="*70)
    print("DEM CALIBRATION REPORT")
    print("="*70)
    
    print("\n📊 TERRAIN STATISTICS COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<20} {'DEM':>12} {'Before':>12} {'After':>12} {'Improvement':>12}")
    print("-" * 70)
    
    def print_metric(name, dem_val, before_val, after_val):
        before_error = abs(before_val - dem_val)
        after_error = abs(after_val - dem_val)
        improvement = ((before_error - after_error) / (before_error + 1e-9)) * 100
        print(f"{name:<20} {dem_val:>12.4f} {before_val:>12.4f} {after_val:>12.4f} {improvement:>11.1f}%")
    
    print_metric('Elevation Mean', dem_stats.elevation_mean, before_stats.elevation_mean, after_stats.elevation_mean)
    print_metric('Elevation Std', dem_stats.elevation_std, before_stats.elevation_std, after_stats.elevation_std)
    print_metric('Slope Mean', dem_stats.slope_mean, before_stats.slope_mean, after_stats.slope_mean)
    print_metric('Slope Std', dem_stats.slope_std, before_stats.slope_std, after_stats.slope_std)
    print_metric('Drainage Density', dem_stats.drainage_density, before_stats.drainage_density, after_stats.drainage_density)
    print_metric('Ridge Spacing', dem_stats.ridge_spacing, before_stats.ridge_spacing, after_stats.ridge_spacing)
    print_metric('Roughness', dem_stats.roughness, before_stats.roughness, after_stats.roughness)
    
    print("\n⚙️  PARAMETER ADJUSTMENTS")
    print("-" * 70)
    print(f"{'Parameter':<20} {'Initial':>12} {'Calibrated':>12} {'Change':>12}")
    print("-" * 70)
    
    def print_param(name, initial, calibrated):
        if isinstance(initial, (int, np.integer)):
            change = calibrated - initial
            print(f"{name:<20} {initial:>12} {calibrated:>12} {change:>+12}")
        else:
            change = calibrated - initial
            print(f"{name:<20} {initial:>12.2f} {calibrated:>12.2f} {change:>+12.2f}")
    
    print_param('Scale', initial_params.scale, calibrated_params.scale)
    print_param('Octaves', initial_params.octaves, calibrated_params.octaves)
    print_param('Persistence', initial_params.persistence, calibrated_params.persistence)
    print_param('Mountain Weight', initial_params.mountain_weight, calibrated_params.mountain_weight)
    print_param('Valley Weight', initial_params.valley_weight, calibrated_params.valley_weight)
    
    # Overall error comparison
    before_metrics = dem_analysis.compare_terrain_statistics(dem_stats, before_stats)
    after_metrics = dem_analysis.compare_terrain_statistics(dem_stats, after_stats)
    
    print("\n📈 OVERALL CALIBRATION QUALITY")
    print("-" * 70)
    print(f"Before calibration error: {before_metrics['overall_error']:.4f}")
    print(f"After calibration error:  {after_metrics['overall_error']:.4f}")
    
    error_reduction = ((before_metrics['overall_error'] - after_metrics['overall_error']) / 
                      (before_metrics['overall_error'] + 1e-9)) * 100
    print(f"Error reduction: {error_reduction:.1f}%")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="DEM Calibration Demo")
    parser.add_argument('--dem', type=str, help='Path to real DEM file (GeoTIFF, PNG, NPY)')
    parser.add_argument('--output', type=str, default='Output/dem_calibration',
                       help='Output directory for results')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of calibration iterations')
    parser.add_argument('--create-synthetic', action='store_true',
                       help='Create and save synthetic DEM for future use')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DEM CALIBRATION DEMO - Phase 4")
    print("="*70)
    print("\nThis demo shows how to calibrate procedural terrain parameters")
    print("to match real-world DEM statistics.\n")
    
    # Enable DEM calibration
    original_flag = dem_analysis.ENABLE_DEM_CALIBRATION
    dem_analysis.ENABLE_DEM_CALIBRATION = True
    
    try:
        # 1. Load or create reference DEM
        print("-" * 70)
        if args.dem:
            print(f"Loading reference DEM: {args.dem}")
            dem, dem_metadata = dem_analysis.load_dem(args.dem)
            print(f"✓ DEM loaded: {dem.shape}, source: {dem_metadata['source']}")
        else:
            print("Creating synthetic reference DEM...")
            dem = create_synthetic_dem(shape=(512, 512), seed=999)
            
            if args.create_synthetic:
                # Save synthetic DEM for future use
                synthetic_path = os.path.join(args.output, 'synthetic_reference_dem.npy')
                os.makedirs(args.output, exist_ok=True)
                np.save(synthetic_path, dem)
                print(f"✓ Synthetic DEM saved: {synthetic_path}")
        
        # 2. Compute DEM statistics
        print("\n" + "-" * 70)
        print("Computing reference DEM statistics...")
        dem_stats = dem_analysis.compute_terrain_statistics(dem)
        print(f"✓ DEM statistics:")
        print(f"  Elevation: mean={dem_stats.elevation_mean:.3f}, std={dem_stats.elevation_std:.3f}")
        print(f"  Slope: mean={dem_stats.slope_mean:.4f}, std={dem_stats.slope_std:.4f}")
        print(f"  Drainage density: {dem_stats.drainage_density:.3f}")
        print(f"  Ridge spacing: {dem_stats.ridge_spacing:.1f} pixels")
        print(f"  Roughness: {dem_stats.roughness:.5f}")
        
        # 3. Generate baseline (uncalibrated) terrain
        print("\n" + "-" * 70)
        print("Generating BASELINE procedural terrain (default parameters)...")
        
        initial_params = NoiseParams(
            scale=100.0,
            octaves=6,
            persistence=0.5,
            seed=42
        )
        
        before_terrain = generate_procedural_heightmap(dem.shape, initial_params, debug_dir=None)
        before_stats = dem_analysis.compute_terrain_statistics(before_terrain)
        
        before_metrics = dem_analysis.compare_terrain_statistics(dem_stats, before_stats)
        print(f"✓ Baseline generated")
        print(f"  Overall error vs DEM: {before_metrics['overall_error']:.4f}")
        
        # 4. Calibrate parameters
        print("\n" + "-" * 70)
        print(f"Running DEM calibration ({args.iterations} iterations)...")
        print("-" * 70)
        
        # Save DEM temporarily for calibration
        temp_dem_path = os.path.join(args.output, 'temp_reference_dem.npy')
        os.makedirs(args.output, exist_ok=True)
        np.save(temp_dem_path, dem)
        
        calibration_config = dem_analysis.CalibrationConfig(
            iterations=args.iterations,
            learning_rate=0.3
        )
        
        debug_dir = os.path.join(args.output, 'calibration_debug')
        calibrated_params, calibration_log = dem_analysis.calibrate_to_dem(
            temp_dem_path,
            initial_params,
            calibration_config=calibration_config,
            debug_dir=debug_dir
        )
        
        print("-" * 70)
        print(f"✓ Calibration complete!")
        
        # 5. Generate calibrated terrain
        print("\n" + "-" * 70)
        print("Generating CALIBRATED terrain with adjusted parameters...")
        
        after_terrain = generate_procedural_heightmap(dem.shape, calibrated_params, debug_dir=None)
        after_stats = dem_analysis.compute_terrain_statistics(after_terrain)
        
        after_metrics = dem_analysis.compare_terrain_statistics(dem_stats, after_stats)
        print(f"✓ Calibrated terrain generated")
        print(f"  Overall error vs DEM: {after_metrics['overall_error']:.4f}")
        
        # 6. Create visualizations
        print("\n" + "-" * 70)
        print("Creating comparison visualizations...")
        
        visualize_comparison(
            dem, dem_stats,
            before_terrain, before_stats,
            after_terrain, after_stats,
            args.output
        )
        
        # 7. Print comprehensive report
        print_statistics_report(dem_stats, before_stats, after_stats, initial_params, calibrated_params)
        
        # Clean up temporary DEM
        if os.path.exists(temp_dem_path):
            os.remove(temp_dem_path)
        
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print(f"\nOutput saved to: {args.output}")
        print("\nGenerated files:")
        print("  - terrain_comparison.png       (DEM vs before vs after)")
        print("  - statistical_comparison.png   (histograms and metrics)")
        print("  - calibration_debug/           (detailed debug outputs)")
        
    finally:
        # Restore original flag
        dem_analysis.ENABLE_DEM_CALIBRATION = original_flag


if __name__ == "__main__":
    main()

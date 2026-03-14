"""DEM (Digital Elevation Model) Analysis and Calibration

PHASE 4: Ground procedural terrain generation in real-world DEM statistics.

This module:
1. Loads DEM data from various formats (GeoTIFF, PNG, NPY)
2. Computes terrain statistics (elevation, slope, drainage, ridge spacing)
3. Compares generated terrain to reference DEM
4. Calibrates procedural parameters to match DEM statistics

Feature flag: ENABLE_DEM_CALIBRATION

IMPORTANT ASSUMPTIONS & LIMITATIONS:
- DEM calibration is OPTIONAL and non-destructive
- Procedural generation works independently without DEM
- Calibration nudges parameters toward DEM stats but doesn't copy DEM
- Parameters are adjusted iteratively, not overfit to single DEM
- Works best with DEMs that match the target terrain type (mountains, plains, etc.)
"""

import numpy as np
from scipy import ndimage, signal, stats
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# FEATURE FLAG
# ============================================================================
ENABLE_DEM_CALIBRATION = False  # Default off to preserve existing behavior


# ============================================================================
# DEM LOADING
# ============================================================================
def load_dem(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load DEM from various file formats.
    
    Supported formats:
    - GeoTIFF (.tif, .tiff) - requires rasterio
    - PNG/JPEG (.png, .jpg) - grayscale images
    - NumPy (.npy, .npz) - raw arrays
    
    Args:
        filepath: Path to DEM file
    
    Returns:
        (heightmap, metadata): Normalized heightmap [0,1] and metadata dict
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"DEM file not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    metadata = {'source': str(filepath), 'format': suffix}
    
    # GeoTIFF (requires rasterio)
    if suffix in ['.tif', '.tiff']:
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                heightmap = src.read(1).astype(np.float32)
                metadata.update({
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'resolution': src.res
                })
                logger.info(f"Loaded GeoTIFF: {src.width}x{src.height}, CRS={src.crs}")
        except ImportError:
            logger.error("rasterio not installed. Install with: pip install rasterio")
            raise ImportError("rasterio required for GeoTIFF support")
    
    # Image formats
    elif suffix in ['.png', '.jpg', '.jpeg']:
        from PIL import Image
        img = Image.open(filepath).convert('L')  # Grayscale
        heightmap = np.array(img, dtype=np.float32)
        metadata['size'] = img.size
        logger.info(f"Loaded image DEM: {img.width}x{img.height}")
    
    # NumPy formats
    elif suffix == '.npy':
        heightmap = np.load(filepath).astype(np.float32)
        logger.info(f"Loaded NumPy DEM: {heightmap.shape}")
    
    elif suffix == '.npz':
        data = np.load(filepath)
        # Try common keys
        for key in ['elevation', 'heightmap', 'dem', 'z', 'height']:
            if key in data:
                heightmap = data[key].astype(np.float32)
                break
        else:
            # Use first array
            heightmap = data[list(data.keys())[0]].astype(np.float32)
        logger.info(f"Loaded NPZ DEM: {heightmap.shape}")
    
    else:
        raise ValueError(f"Unsupported DEM format: {suffix}")
    
    # Normalize to [0, 1]
    if heightmap.size > 0:
        h_min, h_max = heightmap.min(), heightmap.max()
        if h_max > h_min:
            heightmap = (heightmap - h_min) / (h_max - h_min)
        else:
            logger.warning("DEM has constant elevation, setting to 0.5")
            heightmap = np.ones_like(heightmap) * 0.5
        
        metadata['original_range'] = (float(h_min), float(h_max))
        metadata['shape'] = heightmap.shape
    
    return heightmap, metadata


# ============================================================================
# TERRAIN STATISTICS
# ============================================================================
@dataclass
class TerrainStatistics:
    """Statistical description of terrain characteristics."""
    
    # Elevation statistics
    elevation_mean: float
    elevation_std: float
    elevation_histogram: np.ndarray
    elevation_bins: np.ndarray
    
    # Slope statistics
    slope_mean: float
    slope_std: float
    slope_histogram: np.ndarray
    slope_bins: np.ndarray
    
    # Drainage density (higher = more drainage channels)
    drainage_density: float
    
    # Ridge spacing (average distance between ridges in pixels)
    ridge_spacing: float
    
    # Roughness (elevation variance)
    roughness: float


def compute_terrain_statistics(heightmap: np.ndarray, num_bins: int = 50) -> TerrainStatistics:
    """Compute comprehensive terrain statistics.
    
    Args:
        heightmap: 2D elevation array [0, 1]
        num_bins: Number of bins for histograms
    
    Returns:
        TerrainStatistics object
    """
    # Elevation statistics
    elevation_mean = float(np.mean(heightmap))
    elevation_std = float(np.std(heightmap))
    elevation_hist, elevation_bins = np.histogram(heightmap, bins=num_bins, range=(0, 1))
    
    # Slope statistics
    dy, dx = np.gradient(heightmap.astype(np.float64))
    slope = np.sqrt(dx**2 + dy**2)
    slope_mean = float(np.mean(slope))
    slope_std = float(np.std(slope))
    slope_hist, slope_bins = np.histogram(slope, bins=num_bins)
    
    # Drainage density (flow accumulation proxy)
    drainage_density = _compute_drainage_density(heightmap)
    
    # Ridge spacing (characteristic wavelength)
    ridge_spacing = _compute_ridge_spacing(heightmap)
    
    # Roughness (local elevation variance)
    roughness = _compute_roughness(heightmap)
    
    logger.debug(f"Computed terrain stats: elev_mean={elevation_mean:.3f}, slope_mean={slope_mean:.4f}, "
                f"drainage={drainage_density:.3f}, ridge_spacing={ridge_spacing:.1f}px")
    
    return TerrainStatistics(
        elevation_mean=elevation_mean,
        elevation_std=elevation_std,
        elevation_histogram=elevation_hist,
        elevation_bins=elevation_bins,
        slope_mean=slope_mean,
        slope_std=slope_std,
        slope_histogram=slope_hist,
        slope_bins=slope_bins,
        drainage_density=drainage_density,
        ridge_spacing=ridge_spacing,
        roughness=roughness
    )


def _compute_drainage_density(heightmap: np.ndarray) -> float:
    """Estimate drainage density via flow accumulation.
    
    Higher values indicate more drainage channels/rivers.
    """
    # Simple D8 flow accumulation
    flow_accum = np.ones_like(heightmap, dtype=np.float32)
    
    # Sort pixels by elevation (high to low)
    h, w = heightmap.shape
    coords = [(i, j) for i in range(h) for j in range(w)]
    coords_sorted = sorted(coords, key=lambda c: -heightmap[c[0], c[1]])
    
    # Flow directions (8-connected)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for i, j in coords_sorted[:min(len(coords_sorted), 10000)]:  # Limit for performance
        # Find steepest descent neighbor
        min_neighbor = None
        min_elev = heightmap[i, j]
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                if heightmap[ni, nj] < min_elev:
                    min_elev = heightmap[ni, nj]
                    min_neighbor = (ni, nj)
        
        # Flow to lowest neighbor
        if min_neighbor:
            flow_accum[min_neighbor] += flow_accum[i, j]
    
    # Drainage density: fraction of pixels with significant flow
    threshold = np.percentile(flow_accum, 95)  # Top 5% of flow
    drainage_density = float(np.sum(flow_accum > threshold) / flow_accum.size)
    
    return drainage_density


def _compute_ridge_spacing(heightmap: np.ndarray) -> float:
    """Estimate average ridge spacing via autocorrelation.
    
    Returns characteristic wavelength in pixels.
    """
    # Use a sample for performance
    sample_size = min(256, min(heightmap.shape))
    h, w = heightmap.shape
    
    # Extract center sample
    i_start = (h - sample_size) // 2
    j_start = (w - sample_size) // 2
    sample = heightmap[i_start:i_start+sample_size, j_start:j_start+sample_size]
    
    # Compute 1D autocorrelation along x-axis
    autocorr = signal.correlate(sample.mean(axis=0), sample.mean(axis=0), mode='same')
    autocorr = autocorr / autocorr.max()
    
    # Find first minimum after central peak
    center = len(autocorr) // 2
    try:
        # Look for where autocorr drops below 0.5
        threshold_idx = np.where(autocorr[center:] < 0.5)[0]
        if len(threshold_idx) > 0:
            ridge_spacing = float(threshold_idx[0] * 2)  # Characteristic wavelength
        else:
            ridge_spacing = float(sample_size / 4)  # Default fallback
    except Exception:
        ridge_spacing = float(sample_size / 4)
    
    return ridge_spacing


def _compute_roughness(heightmap: np.ndarray, window_size: int = 5) -> float:
    """Compute terrain roughness as local elevation variance.
    
    Args:
        heightmap: Elevation array
        window_size: Size of local window for variance computation
    
    Returns:
        Mean local variance (roughness)
    """
    # Use generic filter for local variance
    def local_variance(values):
        return np.var(values)
    
    local_var = ndimage.generic_filter(heightmap, local_variance, size=window_size)
    roughness = float(np.mean(local_var))
    
    return roughness


# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================
def compare_terrain_statistics(
    dem_stats: TerrainStatistics,
    generated_stats: TerrainStatistics
) -> Dict[str, float]:
    """Compare DEM statistics with generated terrain.
    
    Args:
        dem_stats: Statistics from reference DEM
        generated_stats: Statistics from generated terrain
    
    Returns:
        Dictionary of metric_name -> difference/error
    """
    metrics = {}
    
    # Elevation metrics
    metrics['elevation_mean_error'] = abs(generated_stats.elevation_mean - dem_stats.elevation_mean)
    metrics['elevation_std_error'] = abs(generated_stats.elevation_std - dem_stats.elevation_std)
    
    # Slope metrics
    metrics['slope_mean_error'] = abs(generated_stats.slope_mean - dem_stats.slope_mean)
    metrics['slope_std_error'] = abs(generated_stats.slope_std - dem_stats.slope_std)
    
    # Histogram comparison (Wasserstein distance)
    metrics['elevation_histogram_distance'] = float(
        stats.wasserstein_distance(
            dem_stats.elevation_bins[:-1],
            generated_stats.elevation_bins[:-1],
            dem_stats.elevation_histogram,
            generated_stats.elevation_histogram
        )
    )
    
    metrics['slope_histogram_distance'] = float(
        stats.wasserstein_distance(
            dem_stats.slope_bins[:-1],
            generated_stats.slope_bins[:-1],
            dem_stats.slope_histogram,
            generated_stats.slope_histogram
        )
    )
    
    # Drainage and ridge metrics
    metrics['drainage_density_error'] = abs(generated_stats.drainage_density - dem_stats.drainage_density)
    metrics['ridge_spacing_error'] = abs(generated_stats.ridge_spacing - dem_stats.ridge_spacing)
    metrics['roughness_error'] = abs(generated_stats.roughness - dem_stats.roughness)
    
    # Overall score (lower is better)
    metrics['overall_error'] = (
        metrics['elevation_mean_error'] * 0.1 +
        metrics['elevation_std_error'] * 0.2 +
        metrics['slope_mean_error'] * 0.2 +
        metrics['drainage_density_error'] * 0.2 +
        metrics['roughness_error'] * 0.3
    )
    
    return metrics


# ============================================================================
# PARAMETER CALIBRATION
# ============================================================================
@dataclass
class CalibrationConfig:
    """Configuration for DEM-based parameter calibration."""
    
    # Number of calibration iterations
    iterations: int = 5
    
    # Learning rate for parameter adjustment (0.0 to 1.0)
    learning_rate: float = 0.3
    
    # Which parameters to adjust
    adjust_scale: bool = True
    adjust_octaves: bool = True
    adjust_persistence: bool = True
    adjust_weights: bool = True
    
    # Parameter bounds
    scale_min: float = 50.0
    scale_max: float = 300.0
    octaves_min: int = 4
    octaves_max: int = 10
    persistence_min: float = 0.3
    persistence_max: float = 0.7


def calibrate_to_dem(
    dem_filepath: str,
    initial_params: Any,  # NoiseParams from procedural_noise_utils
    calibration_config: Optional[CalibrationConfig] = None,
    debug_dir: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Calibrate procedural terrain parameters to match DEM statistics.
    
    This is an iterative optimization that:
    1. Loads reference DEM and computes its statistics
    2. Generates terrain with current parameters
    3. Compares statistics and adjusts parameters
    4. Repeats for N iterations
    
    Args:
        dem_filepath: Path to reference DEM
        initial_params: Starting NoiseParams
        calibration_config: Calibration settings
        debug_dir: Directory for debug outputs
    
    Returns:
        (calibrated_params, calibration_log): Tuned parameters and optimization log
    """
    if not ENABLE_DEM_CALIBRATION:
        logger.info("DEM calibration disabled, returning original parameters")
        return initial_params, {'calibration': 'disabled'}
    
    if calibration_config is None:
        calibration_config = CalibrationConfig()
    
    # Import here to avoid circular dependency
    from pipeline.procedural_noise_utils import generate_procedural_heightmap, NoiseParams
    
    logger.info(f"Starting DEM calibration: {calibration_config.iterations} iterations")
    
    # Load reference DEM
    dem_heightmap, dem_metadata = load_dem(dem_filepath)
    dem_stats = compute_terrain_statistics(dem_heightmap)
    
    logger.info(f"Reference DEM loaded: {dem_heightmap.shape}, "
               f"elev_mean={dem_stats.elevation_mean:.3f}, "
               f"slope_mean={dem_stats.slope_mean:.4f}")
    
    # Calibration log
    calibration_log = {
        'dem_metadata': dem_metadata,
        'dem_stats': dem_stats,
        'iterations': []
    }
    
    # Start with initial parameters
    current_params = NoiseParams(
        scale=initial_params.scale,
        octaves=initial_params.octaves,
        persistence=initial_params.persistence,
        lacunarity=initial_params.lacunarity,
        seed=initial_params.seed,
        mountain_weight=initial_params.mountain_weight,
        valley_weight=initial_params.valley_weight,
        river_strength=initial_params.river_strength,
        river_frequency=initial_params.river_frequency
    )
    
    best_params = current_params
    best_error = float('inf')
    
    # Iterative calibration
    for iteration in range(calibration_config.iterations):
        logger.info(f"Calibration iteration {iteration + 1}/{calibration_config.iterations}")
        
        # Generate terrain with current parameters
        # Use same shape as DEM for fair comparison
        generated_heightmap = generate_procedural_heightmap(
            shape=dem_heightmap.shape,
            params=current_params,
            debug_dir=None  # Don't clutter with intermediate outputs
        )
        
        # Compute statistics
        generated_stats = compute_terrain_statistics(generated_heightmap)
        
        # Compare to DEM
        metrics = compare_terrain_statistics(dem_stats, generated_stats)
        
        logger.info(f"  Overall error: {metrics['overall_error']:.4f}")
        
        # Track best parameters
        if metrics['overall_error'] < best_error:
            best_error = metrics['overall_error']
            best_params = NoiseParams(
                scale=current_params.scale,
                octaves=current_params.octaves,
                persistence=current_params.persistence,
                lacunarity=current_params.lacunarity,
                seed=current_params.seed,
                mountain_weight=current_params.mountain_weight,
                valley_weight=current_params.valley_weight,
                river_strength=current_params.river_strength,
                river_frequency=current_params.river_frequency
            )
        
        # Log iteration
        calibration_log['iterations'].append({
            'iteration': iteration,
            'params': {
                'scale': current_params.scale,
                'octaves': current_params.octaves,
                'persistence': current_params.persistence,
                'mountain_weight': current_params.mountain_weight
            },
            'metrics': metrics,
            'generated_stats': generated_stats
        })
        
        # Adjust parameters based on statistics comparison
        if iteration < calibration_config.iterations - 1:  # Don't adjust on last iteration
            current_params = _adjust_parameters(
                current_params,
                dem_stats,
                generated_stats,
                calibration_config
            )
    
    # Save debug outputs
    if debug_dir:
        _save_calibration_debug(
            dem_heightmap, dem_stats,
            best_params, calibration_log,
            debug_dir
        )
    
    logger.info(f"✓ Calibration complete: best_error={best_error:.4f}")
    logger.info(f"  Adjusted parameters:")
    logger.info(f"    scale: {initial_params.scale:.1f} → {best_params.scale:.1f}")
    logger.info(f"    octaves: {initial_params.octaves} → {best_params.octaves}")
    logger.info(f"    persistence: {initial_params.persistence:.2f} → {best_params.persistence:.2f}")
    
    return best_params, calibration_log


def _adjust_parameters(
    params: Any,
    dem_stats: TerrainStatistics,
    generated_stats: TerrainStatistics,
    config: CalibrationConfig
) -> Any:
    """Adjust parameters to reduce statistical differences.
    
    Uses simple gradient-free heuristics:
    - If slope too low → increase octaves or persistence
    - If slope too high → decrease persistence or increase scale
    - If roughness too low → increase persistence
    - If ridge spacing off → adjust scale
    """
    from pipeline.procedural_noise_utils import NoiseParams
    
    lr = config.learning_rate
    
    new_scale = params.scale
    new_octaves = params.octaves
    new_persistence = params.persistence
    new_mountain_weight = params.mountain_weight
    new_valley_weight = params.valley_weight
    
    # Adjust scale based on ridge spacing
    if config.adjust_scale:
        ridge_error = generated_stats.ridge_spacing - dem_stats.ridge_spacing
        if abs(ridge_error) > 5.0:  # Significant difference
            # If generated ridges too close → increase scale (spread out)
            # If generated ridges too far → decrease scale (pack together)
            scale_adjustment = ridge_error * lr * 10.0
            new_scale = np.clip(
                params.scale + scale_adjustment,
                config.scale_min,
                config.scale_max
            )
    
    # Adjust persistence based on roughness and slope
    if config.adjust_persistence:
        roughness_error = generated_stats.roughness - dem_stats.roughness
        slope_error = generated_stats.slope_mean - dem_stats.slope_mean
        
        # If too smooth → increase persistence (more high-freq detail)
        # If too rough → decrease persistence
        persistence_adjustment = (roughness_error + slope_error * 0.5) * lr * 0.5
        new_persistence = np.clip(
            params.persistence + persistence_adjustment,
            config.persistence_min,
            config.persistence_max
        )
    
    # Adjust octaves based on slope distribution
    if config.adjust_octaves:
        slope_std_error = generated_stats.slope_std - dem_stats.slope_std
        
        # If slope variance too low → add octaves (more detail variety)
        # If slope variance too high → reduce octaves
        if slope_std_error < -0.01 and params.octaves < config.octaves_max:
            new_octaves = params.octaves + 1
        elif slope_std_error > 0.01 and params.octaves > config.octaves_min:
            new_octaves = params.octaves - 1
    
    # Adjust terrain weights based on elevation distribution
    if config.adjust_weights:
        elev_mean_error = generated_stats.elevation_mean - dem_stats.elevation_mean
        
        # If generated too low → more mountains
        # If generated too high → more valleys
        if abs(elev_mean_error) > 0.05:
            weight_adjustment = elev_mean_error * lr * 0.3
            new_mountain_weight = np.clip(params.mountain_weight - weight_adjustment, 0.1, 0.9)
            new_valley_weight = np.clip(params.valley_weight + weight_adjustment, 0.1, 0.9)
    
    # Create new params
    adjusted = NoiseParams(
        scale=float(new_scale),
        octaves=int(new_octaves),
        persistence=float(new_persistence),
        lacunarity=params.lacunarity,
        seed=params.seed,
        mountain_weight=float(new_mountain_weight),
        valley_weight=float(new_valley_weight),
        river_strength=params.river_strength,
        river_frequency=params.river_frequency
    )
    
    return adjusted


# ============================================================================
# DEBUG VISUALIZATIONS
# ============================================================================
def _save_calibration_debug(
    dem_heightmap: np.ndarray,
    dem_stats: TerrainStatistics,
    calibrated_params: Any,
    calibration_log: Dict,
    debug_dir: str
) -> None:
    """Save calibration debug visualizations."""
    import matplotlib.pyplot as plt
    from pipeline.procedural_noise_utils import generate_procedural_heightmap
    
    os.makedirs(debug_dir, exist_ok=True)
    
    # Generate final calibrated terrain
    calibrated_heightmap = generate_procedural_heightmap(
        shape=dem_heightmap.shape,
        params=calibrated_params,
        debug_dir=None
    )
    calibrated_stats = compute_terrain_statistics(calibrated_heightmap)
    
    # 1. Side-by-side heightmap comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(dem_heightmap, cmap='terrain')
    axes[0].set_title('Reference DEM', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(calibrated_heightmap, cmap='terrain')
    axes[1].set_title('Calibrated Generated Terrain', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, 'dem_vs_generated.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Statistical comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DEM vs Generated Terrain Statistics', fontsize=16, fontweight='bold')
    
    # Elevation histogram
    axes[0, 0].stairs(dem_stats.elevation_histogram, dem_stats.elevation_bins, 
                     label='DEM', alpha=0.7, fill=True, color='blue')
    axes[0, 0].stairs(calibrated_stats.elevation_histogram, calibrated_stats.elevation_bins,
                     label='Generated', alpha=0.7, fill=True, color='orange')
    axes[0, 0].set_xlabel('Elevation')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Elevation Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Slope histogram
    axes[0, 1].stairs(dem_stats.slope_histogram, dem_stats.slope_bins,
                     label='DEM', alpha=0.7, fill=True, color='blue')
    axes[0, 1].stairs(calibrated_stats.slope_histogram, calibrated_stats.slope_bins,
                     label='Generated', alpha=0.7, fill=True, color='orange')
    axes[0, 1].set_xlabel('Slope')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Slope Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Metric comparison bar chart
    metric_names = ['Elev Mean', 'Elev Std', 'Slope Mean', 'Drainage', 'Roughness']
    dem_values = [
        dem_stats.elevation_mean,
        dem_stats.elevation_std,
        dem_stats.slope_mean,
        dem_stats.drainage_density,
        dem_stats.roughness
    ]
    gen_values = [
        calibrated_stats.elevation_mean,
        calibrated_stats.elevation_std,
        calibrated_stats.slope_mean,
        calibrated_stats.drainage_density,
        calibrated_stats.roughness
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, dem_values, width, label='DEM', alpha=0.7, color='blue')
    axes[1, 0].bar(x + width/2, gen_values, width, label='Generated', alpha=0.7, color='orange')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Metric Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Calibration convergence
    if len(calibration_log['iterations']) > 0:
        iterations = [it['iteration'] for it in calibration_log['iterations']]
        errors = [it['metrics']['overall_error'] for it in calibration_log['iterations']]
        
        axes[1, 1].plot(iterations, errors, marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Overall Error')
        axes[1, 1].set_title('Calibration Convergence')
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, 'calibration_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Calibration debug outputs saved to {debug_dir}")


# ============================================================================
# UTILITY: EXPORT DEM FROM GENERATED TERRAIN
# ============================================================================
def export_as_dem(
    heightmap: np.ndarray,
    output_path: str,
    format: str = 'png',
    metadata: Optional[Dict] = None
) -> None:
    """Export generated heightmap as DEM file.
    
    Args:
        heightmap: Terrain heightmap [0, 1]
        output_path: Path to save DEM
        format: Output format ('png', 'npy', 'npz', 'tif')
        metadata: Optional metadata to save with DEM
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'png':
        from PIL import Image
        # Convert to 16-bit grayscale for better precision
        heightmap_uint16 = (heightmap * 65535).astype(np.uint16)
        img = Image.fromarray(heightmap_uint16, mode='I;16')
        img.save(output_path)
        logger.info(f"Exported DEM as PNG: {output_path}")
    
    elif format == 'npy':
        np.save(output_path, heightmap)
        logger.info(f"Exported DEM as NPY: {output_path}")
    
    elif format == 'npz':
        save_dict = {'elevation': heightmap}
        if metadata:
            save_dict['metadata'] = metadata
        np.savez(output_path, **save_dict)
        logger.info(f"Exported DEM as NPZ: {output_path}")
    
    elif format == 'tif':
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            h, w = heightmap.shape
            # Default georeferencing (can be overridden with metadata)
            transform = from_bounds(0, 0, w, h, w, h)
            
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=h, width=w,
                count=1, dtype=heightmap.dtype,
                crs=metadata.get('crs', 'EPSG:4326') if metadata else 'EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(heightmap, 1)
            
            logger.info(f"Exported DEM as GeoTIFF: {output_path}")
        except ImportError:
            logger.error("rasterio not installed. Cannot export GeoTIFF.")
            raise
    
    else:
        raise ValueError(f"Unsupported export format: {format}")

"""Terrain analysis and biome mask generation.

PHASE 3: Compute terrain-derived attributes and smooth biome masks.

This module analyzes a heightmap to produce:
- Terrain attributes: elevation, slope, curvature, aspect, distance-to-water
- Smooth biome masks: snow, rock, scree, grassland, forest, wetlands

Biome masks are continuous [0,1] values that blend smoothly rather than
hard thresholds. This allows materials/textures to blend naturally.

Feature flag: ENABLE_BIOMES
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import os


# ============================================================================
# FEATURE FLAG
# ============================================================================
ENABLE_BIOMES = True


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class BiomeParams:
    """Parameters for biome classification.
    
    Elevation and slope ranges use soft transitions (sigmoid) for smooth blending.
    """
    # Snow biome: high elevation peaks
    snow_elevation_min: float = 0.7
    snow_elevation_range: float = 0.15  # transition width
    
    # Rock biome: steep slopes
    rock_slope_min: float = 0.4
    rock_slope_range: float = 0.2
    
    # Scree biome: moderately steep slopes at mid-elevation
    scree_slope_min: float = 0.25
    scree_slope_max: float = 0.5
    scree_elevation_min: float = 0.3
    scree_elevation_max: float = 0.8
    
    # Grassland: gentle slopes, mid elevation
    grassland_slope_max: float = 0.3
    grassland_elevation_min: float = 0.2
    grassland_elevation_max: float = 0.6
    
    # Forest: gentle slopes, mid-to-high elevation
    forest_slope_max: float = 0.25
    forest_elevation_min: float = 0.25
    forest_elevation_max: float = 0.7
    
    # Wetlands: low elevation, flat areas, near water
    wetland_elevation_max: float = 0.15
    wetland_slope_max: float = 0.1
    wetland_distance_max: float = 20.0  # pixels
    
    # Water detection threshold
    water_elevation_threshold: float = 0.05


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def _sigmoid(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """Smooth sigmoid transition from 0 to 1.
    
    Args:
        x: Input values
        center: Point where sigmoid = 0.5
        width: Controls transition steepness (smaller = sharper)
    
    Returns:
        Values in [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


def _inverse_sigmoid(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """Smooth sigmoid transition from 1 to 0.
    
    Args:
        x: Input values
        center: Point where sigmoid = 0.5
        width: Controls transition steepness (smaller = sharper)
    
    Returns:
        Values in [0, 1]
    """
    return 1.0 - _sigmoid(x, center, width)


def _soft_range(x: np.ndarray, min_val: float, max_val: float, width: float = 0.05) -> np.ndarray:
    """Smooth membership in a range [min_val, max_val].
    
    Args:
        x: Input values
        min_val: Lower bound
        max_val: Upper bound
        width: Transition width at boundaries
    
    Returns:
        Values in [0, 1], 1 inside range, 0 outside, smooth transitions
    """
    lower = _sigmoid(x, min_val, width)
    upper = _inverse_sigmoid(x, max_val, width)
    return lower * upper


# ============================================================================
# TERRAIN ATTRIBUTE COMPUTATION
# ============================================================================
def compute_slope(heightmap: np.ndarray) -> np.ndarray:
    """Compute terrain slope magnitude from heightmap gradients.
    
    Args:
        heightmap: 2D array of elevation values [0, 1]
    
    Returns:
        Slope magnitude (gradient norm), normalized to approximate [0, 1]
    """
    dy, dx = np.gradient(heightmap.astype(np.float64))
    slope = np.sqrt(dx**2 + dy**2)
    
    # Normalize: typical max slope ~0.1-0.2 for smooth terrain
    # Scale to make steep slopes approach 1.0
    slope_normalized = np.clip(slope * 5.0, 0.0, 1.0)
    
    return slope_normalized.astype(np.float32)


def compute_curvature(heightmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute terrain curvature (concave/convex).
    
    Args:
        heightmap: 2D array of elevation values
    
    Returns:
        (convexity, concavity): Both in [0, 1], higher = more curved
    """
    # Laplacian approximates curvature
    laplacian = ndimage.laplace(heightmap.astype(np.float64))
    
    # Positive laplacian = convex (peaks, ridges)
    # Negative laplacian = concave (valleys, basins)
    convexity = np.clip(laplacian * 50.0, 0.0, 1.0).astype(np.float32)
    concavity = np.clip(-laplacian * 50.0, 0.0, 1.0).astype(np.float32)
    
    return convexity, concavity


def compute_aspect(heightmap: np.ndarray) -> np.ndarray:
    """Compute terrain aspect (direction of slope).
    
    Args:
        heightmap: 2D array of elevation values
    
    Returns:
        Aspect in radians [0, 2π], flat areas = 0
    """
    dy, dx = np.gradient(heightmap.astype(np.float64))
    aspect = np.arctan2(dy, dx)
    
    # Shift to [0, 2π]
    aspect = (aspect + 2 * np.pi) % (2 * np.pi)
    
    return aspect.astype(np.float32)


def compute_distance_to_water(heightmap: np.ndarray, water_threshold: float = 0.05) -> np.ndarray:
    """Compute distance transform to nearest water pixels.
    
    Args:
        heightmap: 2D array of elevation values [0, 1]
        water_threshold: Elevation below which is considered water
    
    Returns:
        Distance in pixels to nearest water, normalized
    """
    water_mask = heightmap < water_threshold
    
    if not water_mask.any():
        # No water detected, return max distance everywhere
        return np.ones_like(heightmap, dtype=np.float32) * 100.0
    
    # Distance transform
    distance = ndimage.distance_transform_edt(~water_mask)
    
    return distance.astype(np.float32)


# ============================================================================
# BIOME MASK COMPUTATION
# ============================================================================
def compute_biome_masks(
    heightmap: np.ndarray,
    slope: np.ndarray,
    distance_to_water: np.ndarray,
    params: BiomeParams
) -> Dict[str, np.ndarray]:
    """Compute smooth biome masks from terrain attributes.
    
    Each biome mask is a continuous [0, 1] value indicating membership strength.
    Biomes can overlap; final assignment can take max or blend.
    
    Args:
        heightmap: Elevation [0, 1]
        slope: Slope magnitude [0, 1]
        distance_to_water: Distance in pixels to nearest water
        params: Biome classification parameters
    
    Returns:
        Dictionary of biome_name -> mask array
    """
    masks = {}
    
    # SNOW: High elevation
    masks['snow'] = _sigmoid(
        heightmap,
        params.snow_elevation_min,
        params.snow_elevation_range
    )
    
    # ROCK: Steep slopes (cliffs, escarpments)
    masks['rock'] = _sigmoid(
        slope,
        params.rock_slope_min,
        params.rock_slope_range
    )
    
    # SCREE: Moderate slopes at mid-elevation (loose rock debris)
    scree_slope = _soft_range(
        slope,
        params.scree_slope_min,
        params.scree_slope_max,
        width=0.05
    )
    scree_elevation = _soft_range(
        heightmap,
        params.scree_elevation_min,
        params.scree_elevation_max,
        width=0.1
    )
    masks['scree'] = scree_slope * scree_elevation
    
    # GRASSLAND: Gentle slopes, mid elevation
    grassland_slope = _inverse_sigmoid(
        slope,
        params.grassland_slope_max,
        0.05
    )
    grassland_elevation = _soft_range(
        heightmap,
        params.grassland_elevation_min,
        params.grassland_elevation_max,
        width=0.1
    )
    masks['grassland'] = grassland_slope * grassland_elevation
    
    # FOREST: Gentle slopes, mid-to-high elevation
    forest_slope = _inverse_sigmoid(
        slope,
        params.forest_slope_max,
        0.05
    )
    forest_elevation = _soft_range(
        heightmap,
        params.forest_elevation_min,
        params.forest_elevation_max,
        width=0.1
    )
    masks['forest'] = forest_slope * forest_elevation
    
    # WETLANDS: Low, flat, near water
    wetland_elevation = _inverse_sigmoid(
        heightmap,
        params.wetland_elevation_max,
        0.03
    )
    wetland_slope = _inverse_sigmoid(
        slope,
        params.wetland_slope_max,
        0.02
    )
    wetland_distance = _inverse_sigmoid(
        distance_to_water,
        params.wetland_distance_max,
        5.0
    )
    masks['wetlands'] = wetland_elevation * wetland_slope * wetland_distance
    
    # WATER: Very low elevation
    masks['water'] = _inverse_sigmoid(
        heightmap,
        params.water_elevation_threshold,
        0.01
    )
    
    return masks


def compute_dominant_biome(masks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dominant biome at each pixel.
    
    Args:
        masks: Dictionary of biome_name -> mask array
    
    Returns:
        (dominant_biome_index, dominant_biome_strength):
            - Index: integer index into sorted biome names
            - Strength: maximum mask value at each pixel
    """
    biome_names = sorted(masks.keys())
    stack = np.stack([masks[name] for name in biome_names], axis=0)
    
    dominant_idx = np.argmax(stack, axis=0).astype(np.uint8)
    dominant_strength = np.max(stack, axis=0).astype(np.float32)
    
    return dominant_idx, dominant_strength


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================
def analyze_terrain(
    heightmap: np.ndarray,
    params: Optional[BiomeParams] = None,
    debug_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """Analyze terrain and compute biome masks.
    
    Args:
        heightmap: 2D elevation array [0, 1]
        params: Biome classification parameters (uses defaults if None)
        debug_dir: Optional directory to save debug visualizations
    
    Returns:
        Dictionary with keys:
            - 'elevation': original heightmap
            - 'slope': slope magnitude [0, 1]
            - 'convexity', 'concavity': curvature [0, 1]
            - 'aspect': slope direction [0, 2π]
            - 'distance_to_water': distance in pixels
            - 'biome_masks': dict of biome_name -> mask
            - 'dominant_biome': (index, strength) tuple
    """
    if not ENABLE_BIOMES:
        return {
            'elevation': heightmap,
            'biome_masks': {},
            'dominant_biome': (np.zeros_like(heightmap, dtype=np.uint8), 
                              np.zeros_like(heightmap, dtype=np.float32))
        }
    
    if params is None:
        params = BiomeParams()
    
    # Compute terrain attributes
    slope = compute_slope(heightmap)
    convexity, concavity = compute_curvature(heightmap)
    aspect = compute_aspect(heightmap)
    distance_to_water = compute_distance_to_water(heightmap, params.water_elevation_threshold)
    
    # Compute biome masks
    biome_masks = compute_biome_masks(heightmap, slope, distance_to_water, params)
    dominant_biome_idx, dominant_biome_strength = compute_dominant_biome(biome_masks)
    
    # Debug visualizations
    if debug_dir is not None:
        _save_debug_visualizations(
            heightmap, slope, convexity, concavity, aspect, distance_to_water,
            biome_masks, dominant_biome_idx, debug_dir
        )
    
    return {
        'elevation': heightmap,
        'slope': slope,
        'convexity': convexity,
        'concavity': concavity,
        'aspect': aspect,
        'distance_to_water': distance_to_water,
        'biome_masks': biome_masks,
        'dominant_biome': (dominant_biome_idx, dominant_biome_strength)
    }


# ============================================================================
# DEBUG VISUALIZATIONS
# ============================================================================
def _save_debug_visualizations(
    heightmap: np.ndarray,
    slope: np.ndarray,
    convexity: np.ndarray,
    concavity: np.ndarray,
    aspect: np.ndarray,
    distance_to_water: np.ndarray,
    biome_masks: Dict[str, np.ndarray],
    dominant_biome_idx: np.ndarray,
    debug_dir: str
) -> None:
    """Save debug images for terrain attributes and biome masks."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    os.makedirs(debug_dir, exist_ok=True)
    
    # Terrain attributes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Terrain Attributes', fontsize=16)
    
    axes[0, 0].imshow(heightmap, cmap='terrain')
    axes[0, 0].set_title('Elevation')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(slope, cmap='hot')
    axes[0, 1].set_title('Slope')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(convexity, cmap='RdYlGn')
    axes[0, 2].set_title('Convexity (Ridges/Peaks)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(concavity, cmap='Blues')
    axes[1, 0].set_title('Concavity (Valleys/Basins)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(aspect, cmap='hsv')
    axes[1, 1].set_title('Aspect (Slope Direction)')
    axes[1, 1].axis('off')
    
    # Normalize distance for visualization
    dist_vis = np.clip(distance_to_water / 50.0, 0, 1)
    axes[1, 2].imshow(dist_vis, cmap='Blues_r')
    axes[1, 2].set_title('Distance to Water')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, 'terrain_attributes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Individual biome masks
    biome_names = sorted(biome_masks.keys())
    n_biomes = len(biome_names)
    cols = 3
    rows = (n_biomes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle('Biome Masks (Continuous [0,1])', fontsize=16)
    
    axes = axes.flatten() if n_biomes > 1 else [axes]
    
    for idx, biome_name in enumerate(biome_names):
        axes[idx].imshow(biome_masks[biome_name], cmap='viridis', vmin=0, vmax=1)
        axes[idx].set_title(biome_name.capitalize())
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_biomes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, 'biome_masks.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dominant biome composite
    # Create a colormap for biomes
    biome_colors = [
        [0.05, 0.05, 0.4],    # water - dark blue
        [0.3, 0.6, 0.3],      # forest - green
        [0.6, 0.7, 0.4],      # grassland - yellow-green
        [0.7, 0.3, 0.2],      # rock - brown-red
        [0.6, 0.6, 0.5],      # scree - gray-brown
        [0.9, 0.95, 1.0],     # snow - white
        [0.3, 0.5, 0.4],      # wetlands - teal
    ]
    
    # Ensure we have enough colors
    while len(biome_colors) < n_biomes:
        biome_colors.append([0.5, 0.5, 0.5])
    
    cmap = ListedColormap(biome_colors[:n_biomes])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(dominant_biome_idx, cmap=cmap, vmin=0, vmax=n_biomes-1)
    ax.set_title('Dominant Biome Classification', fontsize=14)
    ax.axis('off')
    
    # Add colorbar with biome names
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_biomes), fraction=0.046, pad=0.04)
    cbar.set_ticklabels([name.capitalize() for name in biome_names])
    
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, 'dominant_biome.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[TerrainAnalysis] Debug visualizations saved to {debug_dir}")

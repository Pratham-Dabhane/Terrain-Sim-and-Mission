"""
Cost Map Generator for Mission Planning
Converts heightmap and terrain parameters into traversal cost grid.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def cost_map(heightmap: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
    """
    Compute comprehensive cost map for pathfinding using terrain parameters.
    
    Incorporates multiple cost factors:
    - Slope cost: Steep terrain is expensive to traverse
    - Elevation cost: High altitude may be penalized
    - Obstacle cost: Water bodies and extreme terrain features
    
    Args:
        heightmap (np.ndarray): 2D heightmap array (normalized 0-1)
        parameters (Dict[str, Any]): Terrain parameters from prompt parsing
            Expected keys:
            - water_level: Threshold for water detection
            - elevation_scale: Height scaling factor
            - roughness: Terrain roughness (affects slope penalty)
            - biome_type: Terrain biome (affects cost weights)
        
    Returns:
        np.ndarray: Normalized cost map (0-1) where higher values are more expensive
    """
    # Ensure heightmap is properly normalized
    if heightmap.max() > 1.0:
        heightmap = heightmap / heightmap.max()
    
    logger.info(f"Computing cost map with parameters: biome={parameters.get('biome_type', 'unknown')}, "
                f"roughness={parameters.get('roughness', 0.5):.2f}")
    
    # Extract parameters with defaults
    water_level = parameters.get('water_level', 0.2)
    elevation_scale = parameters.get('elevation_scale', 1.0)
    roughness = parameters.get('roughness', 0.5)
    biome_type = parameters.get('biome_type', 'mountain')
    
    # 1. SLOPE COST: Compute terrain gradients
    gx, gy = np.gradient(heightmap)
    slope = np.sqrt(gx**2 + gy**2)
    
    # Scale slope cost by roughness parameter
    # More rough terrain = higher slope penalty
    slope_weight = 1.5 + roughness * 1.5  # Range: 1.5 - 3.0
    slope_cost = slope * slope_weight
    
    # 2. ELEVATION COST: Penalize extreme elevations
    # High elevations are harder to traverse (thin air, cold, etc.)
    # Scaled by elevation_scale parameter
    elevation_penalty = np.where(
        heightmap > 0.7,  # High elevation threshold
        (heightmap - 0.7) * elevation_scale * 0.8,  # Penalize high terrain
        0.0
    )
    
    # 3. OBSTACLE COST: Identify impassable or difficult terrain
    # Water bodies (below water_level)
    water_mask = heightmap < water_level
    water_cost = np.where(water_mask, 5.0, 0.0)  # High penalty for water
    
    # Extremely steep terrain (cliffs)
    cliff_threshold = 0.15  # Steep gradient threshold
    cliff_mask = slope > cliff_threshold
    cliff_cost = np.where(cliff_mask, 3.0, 0.0)  # Very high penalty for cliffs
    
    # 4. BIOME-SPECIFIC ADJUSTMENTS
    biome_modifiers = {
        'mountain': {'elevation_weight': 1.2, 'slope_weight': 1.3},
        'canyon': {'elevation_weight': 0.8, 'slope_weight': 1.5},
        'desert': {'elevation_weight': 0.6, 'slope_weight': 0.8},
        'valley': {'elevation_weight': 0.7, 'slope_weight': 0.9},
        'plateau': {'elevation_weight': 1.0, 'slope_weight': 0.7},
        'coastal': {'elevation_weight': 0.5, 'slope_weight': 1.0},
        'river': {'elevation_weight': 0.4, 'slope_weight': 1.1},
        'arctic': {'elevation_weight': 1.3, 'slope_weight': 1.4},
    }
    
    # Get biome-specific modifiers or use defaults
    biome_mod = biome_modifiers.get(biome_type, {'elevation_weight': 1.0, 'slope_weight': 1.0})
    
    logger.debug(f"Biome modifiers for '{biome_type}': {biome_mod}")
    
    # 5. COMBINE ALL COSTS
    # Weighted combination of all cost factors
    total_cost = (
        slope_cost * biome_mod['slope_weight'] * 0.4 +      # Slope: 40% base weight
        elevation_penalty * biome_mod['elevation_weight'] * 0.2 +  # Elevation: 20% base weight
        water_cost * 0.25 +                                  # Water: 25% weight
        cliff_cost * 0.15                                    # Cliffs: 15% weight
    )
    
    # 6. NORMALIZE to [0, 1] range
    if total_cost.max() > 0:
        cost_normalized = total_cost / total_cost.max()
    else:
        cost_normalized = total_cost
    
    # Add small base cost to all cells (minimum movement cost)
    cost_normalized = cost_normalized * 0.95 + 0.05  # Range: [0.05, 1.0]
    
    # Log statistics
    logger.info(f"Cost map computed - Mean: {cost_normalized.mean():.3f}, "
                f"Max: {cost_normalized.max():.3f}, "
                f"Water cells: {water_mask.sum()}, "
                f"Cliff cells: {cliff_mask.sum()}")
    
    return cost_normalized


def visualize_cost_map(cost_map: np.ndarray, output_path: str) -> None:
    """
    Create visualization of the cost map.
    
    Args:
        cost_map (np.ndarray): Cost map to visualize
        output_path (str): Output file path for visualization
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display cost map with hot colormap (red = expensive, blue = cheap)
    im = ax.imshow(cost_map, cmap='hot_r', origin='upper', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Traversal Cost')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low', 'Moderate', 'High', 'Very High', 'Extreme'])
    
    # Set title and labels
    ax.set_title("Mission Planning: Terrain Cost Map", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add text annotation
    ax.text(0.02, 0.98, "Blue = Easy terrain\nRed = Difficult terrain", 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Save figure 
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"✓ Cost map visualization saved to: {output_path}")


def analyze_cost_statistics(cost_map: np.ndarray, heightmap: np.ndarray, 
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze cost map and terrain statistics for mission planning.
    
    Args:
        cost_map (np.ndarray): Computed cost map
        heightmap (np.ndarray): Terrain heightmap
        parameters (Dict[str, Any]): Terrain parameters
        
    Returns:
        Dict[str, Any]: Statistical analysis of terrain difficulty
    """
    # Compute gradients for slope analysis
    gx, gy = np.gradient(heightmap)
    slope = np.sqrt(gx**2 + gy**2)
    
    # Cost thresholds for difficulty classification
    easy_threshold = 0.3
    moderate_threshold = 0.6
    hard_threshold = 0.8
    
    easy_area = np.sum(cost_map < easy_threshold) / cost_map.size * 100
    moderate_area = np.sum((cost_map >= easy_threshold) & (cost_map < moderate_threshold)) / cost_map.size * 100
    hard_area = np.sum((cost_map >= moderate_threshold) & (cost_map < hard_threshold)) / cost_map.size * 100
    extreme_area = np.sum(cost_map >= hard_threshold) / cost_map.size * 100
    
    water_level = parameters.get('water_level', 0.2)
    water_area = np.sum(heightmap < water_level) / heightmap.size * 100
    
    stats = {
        'elevation_range': (float(heightmap.min()), float(heightmap.max())),
        'mean_elevation': float(heightmap.mean()),
        'max_slope': float(slope.max()),
        'mean_slope': float(slope.mean()),
        'mean_cost': float(cost_map.mean()),
        'max_cost': float(cost_map.max()),
        'min_cost': float(cost_map.min()),
        'easy_terrain_percent': float(easy_area),
        'moderate_terrain_percent': float(moderate_area),
        'hard_terrain_percent': float(hard_area),
        'extreme_terrain_percent': float(extreme_area),
        'water_area_percent': float(water_area),
        'biome_type': parameters.get('biome_type', 'unknown'),
        'roughness': parameters.get('roughness', 0.5),
        'shape': heightmap.shape
    }
    
    logger.info(f"Terrain analysis: {easy_area:.1f}% easy, {moderate_area:.1f}% moderate, "
                f"{hard_area:.1f}% hard, {extreme_area:.1f}% extreme")
    
    return stats

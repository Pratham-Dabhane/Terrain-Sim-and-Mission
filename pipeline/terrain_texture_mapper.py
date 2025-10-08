"""
Terrain Texture Mapper
Provides slope-based color mapping for heightmaps with elevation-based terrain classification.
"""

import numpy as np
from PIL import Image


def colorize_by_elevation_and_slope(heightmap):
    """
    Colorize a heightmap based on elevation and slope for realistic terrain visualization.
    
    Args:
        heightmap (np.ndarray): 2D array representing terrain elevation (normalized 0-1)
        
    Returns:
        PIL.Image: RGB image with colored terrain
        
    Color scheme:
        - Water (< 0.2): Blue shades
        - Grass (0.2-0.45): Green shades  
        - Rock (0.45-0.75): Brown shades with slope darkening
        - Snow (>= 0.75): White shades
    """
    # Ensure heightmap is numpy array and normalized to 0-1
    if isinstance(heightmap, Image.Image):
        heightmap = np.array(heightmap.convert('L')) / 255.0
    elif isinstance(heightmap, np.ndarray):
        if heightmap.max() > 1.0:
            heightmap = heightmap / 255.0
        heightmap = np.clip(heightmap, 0, 1)
    
    height, width = heightmap.shape
    
    # Compute slope using gradient magnitude
    grad_y, grad_x = np.gradient(heightmap)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize slope for darkening factor (0-1)
    slope_normalized = np.clip(slope / slope.max() if slope.max() > 0 else slope, 0, 1)
    
    # Initialize RGB channels
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define color thresholds and base colors
    water_threshold = 0.2
    grass_threshold = 0.45
    rock_threshold = 0.75
    
    # Water areas (blue shades)
    water_mask = heightmap < water_threshold
    water_depth = (water_threshold - heightmap) / water_threshold  # Deeper = darker
    water_depth = np.clip(water_depth, 0, 1)
    
    rgb_image[water_mask, 0] = (30 + 50 * (1 - water_depth[water_mask])).astype(np.uint8)  # R: 30-80
    rgb_image[water_mask, 1] = (100 + 100 * (1 - water_depth[water_mask])).astype(np.uint8)  # G: 100-200
    rgb_image[water_mask, 2] = (200 + 55 * (1 - water_depth[water_mask])).astype(np.uint8)  # B: 200-255
    
    # Grass areas (green shades)
    grass_mask = (heightmap >= water_threshold) & (heightmap < grass_threshold)
    grass_height = (heightmap - water_threshold) / (grass_threshold - water_threshold)
    grass_height = np.clip(grass_height, 0, 1)
    
    # Apply slope darkening to grass
    slope_factor = 1 - 0.3 * slope_normalized  # Reduce brightness by up to 30% on steep slopes
    
    rgb_image[grass_mask, 0] = (50 + 80 * grass_height[grass_mask] * slope_factor[grass_mask]).astype(np.uint8)  # R: 50-130
    rgb_image[grass_mask, 1] = (120 + 100 * grass_height[grass_mask] * slope_factor[grass_mask]).astype(np.uint8)  # G: 120-220
    rgb_image[grass_mask, 2] = (30 + 50 * grass_height[grass_mask] * slope_factor[grass_mask]).astype(np.uint8)  # B: 30-80
    
    # Rock areas (brown shades with strong slope darkening)
    rock_mask = (heightmap >= grass_threshold) & (heightmap < rock_threshold)
    rock_height = (heightmap - grass_threshold) / (rock_threshold - grass_threshold)
    rock_height = np.clip(rock_height, 0, 1)
    
    # Strong slope darkening for rocks
    slope_factor = 1 - 0.5 * slope_normalized  # Reduce brightness by up to 50% on steep slopes
    
    rgb_image[rock_mask, 0] = (100 + 80 * rock_height[rock_mask] * slope_factor[rock_mask]).astype(np.uint8)  # R: 100-180
    rgb_image[rock_mask, 1] = (60 + 60 * rock_height[rock_mask] * slope_factor[rock_mask]).astype(np.uint8)  # G: 60-120
    rgb_image[rock_mask, 2] = (30 + 40 * rock_height[rock_mask] * slope_factor[rock_mask]).astype(np.uint8)  # B: 30-70
    
    # Snow areas (white shades)
    snow_mask = heightmap >= rock_threshold
    snow_height = (heightmap - rock_threshold) / (1 - rock_threshold)
    snow_height = np.clip(snow_height, 0, 1)
    
    # Light slope darkening for snow
    slope_factor = 1 - 0.2 * slope_normalized  # Reduce brightness by up to 20% on steep slopes
    
    base_snow = 200 + 55 * snow_height[snow_mask] * slope_factor[snow_mask]
    rgb_image[snow_mask, 0] = base_snow.astype(np.uint8)  # R: 200-255
    rgb_image[snow_mask, 1] = base_snow.astype(np.uint8)  # G: 200-255
    rgb_image[snow_mask, 2] = base_snow.astype(np.uint8)  # B: 200-255
    
    # Convert to PIL Image
    colored_terrain = Image.fromarray(rgb_image, mode='RGB')
    
    return colored_terrain


def get_terrain_stats(heightmap):
    """
    Get statistics about terrain distribution for debugging.
    
    Args:
        heightmap (np.ndarray): 2D heightmap array
        
    Returns:
        dict: Statistics about terrain type coverage
    """
    if isinstance(heightmap, Image.Image):
        heightmap = np.array(heightmap.convert('L')) / 255.0
    elif isinstance(heightmap, np.ndarray):
        if heightmap.max() > 1.0:
            heightmap = heightmap / 255.0
    
    total_pixels = heightmap.size
    
    water_pixels = np.sum(heightmap < 0.2)
    grass_pixels = np.sum((heightmap >= 0.2) & (heightmap < 0.45))
    rock_pixels = np.sum((heightmap >= 0.45) & (heightmap < 0.75))
    snow_pixels = np.sum(heightmap >= 0.75)
    
    return {
        'water_coverage': water_pixels / total_pixels * 100,
        'grass_coverage': grass_pixels / total_pixels * 100,
        'rock_coverage': rock_pixels / total_pixels * 100,
        'snow_coverage': snow_pixels / total_pixels * 100,
        'elevation_range': (heightmap.min(), heightmap.max()),
        'mean_elevation': heightmap.mean()
    }
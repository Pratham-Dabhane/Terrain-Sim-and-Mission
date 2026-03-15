"""
Terrain Texture Mapper
Provides slope-based color mapping for heightmaps with elevation-based terrain classification.
"""

import numpy as np
from PIL import Image


def _smoothstep(edge0, edge1, x):
    """Smooth interpolation in [0,1] between two edges."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gaussian_blur(arr, sigma=2.0):
    """Best-effort Gaussian blur with graceful fallback."""
    try:
        from scipy import ndimage
        return ndimage.gaussian_filter(arr, sigma=sigma)
    except Exception:
        # Fallback: simple box blur using rolling average.
        out = arr.astype(np.float32)
        for _ in range(3):
            out = (
                out
                + np.roll(out, 1, axis=0)
                + np.roll(out, -1, axis=0)
                + np.roll(out, 1, axis=1)
                + np.roll(out, -1, axis=1)
            ) / 5.0
        return out


def _compute_normals_and_slope(heightmap):
    """Compute per-pixel normals and normalized slope from heightmap gradients."""
    dy, dx = np.gradient(heightmap.astype(np.float32))
    slope = np.sqrt(dx * dx + dy * dy)
    slope_n = slope / (slope.max() + 1e-8)

    nx = -dx
    ny = -dy
    nz = np.ones_like(heightmap, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm
    return nx, ny, nz, slope_n


def _compute_relief_shading(heightmap, nx, ny, nz):
    """Compute hillshade and ambient-occlusion-like term from local relief."""
    # Sun direction (normalized) for readable relief.
    sun = np.array([0.45, -0.35, 0.82], dtype=np.float32)
    sun = sun / (np.linalg.norm(sun) + 1e-8)

    hillshade = np.clip(nx * sun[0] + ny * sun[1] + nz * sun[2], 0.0, 1.0)
    hillshade = 0.68 + 0.32 * hillshade

    local_mean = _gaussian_blur(heightmap.astype(np.float32), sigma=2.5)
    cavity = np.clip(local_mean - heightmap, 0.0, 1.0)
    ao = np.clip(1.0 - 0.7 * cavity, 0.62, 1.0)
    return hillshade, ao


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
    nx, ny, nz, slope_n = _compute_normals_and_slope(heightmap)

    # Curvature proxy from Laplacian-like second derivatives.
    dy, dx = np.gradient(heightmap.astype(np.float32))
    dyy = np.gradient(dy, axis=0)
    dxx = np.gradient(dx, axis=1)
    curvature = dxx + dyy
    convex = np.clip(curvature, 0.0, None)
    concave = np.clip(-curvature, 0.0, None)
    convex_n = convex / (convex.max() + 1e-8)
    concave_n = concave / (concave.max() + 1e-8)

    # Material weights from terrain attributes.
    water_w = 1.0 - _smoothstep(0.10, 0.20, heightmap)
    snow_w = _smoothstep(0.68, 0.88, heightmap) * (0.6 + 0.4 * nz)
    rock_w = np.clip(_smoothstep(0.20, 0.75, slope_n) + 0.35 * convex_n, 0.0, 1.0)
    grass_w = _smoothstep(0.16, 0.40, heightmap) * (1.0 - _smoothstep(0.40, 0.85, slope_n))
    soil_w = np.clip(0.45 * concave_n + _smoothstep(0.18, 0.55, heightmap) * (1.0 - grass_w), 0.0, 1.0)

    # Tri-planar-style proxy: steep faces bias to rock/soil, flat tops bias to grass/snow.
    steep = np.clip(1.0 - nz, 0.0, 1.0)
    rock_w = np.clip(rock_w + 0.6 * steep, 0.0, 1.0)
    soil_w = np.clip(soil_w + 0.25 * steep, 0.0, 1.0)
    grass_w = np.clip(grass_w * (1.0 - 0.5 * steep), 0.0, 1.0)
    snow_w = np.clip(snow_w * (0.75 + 0.25 * nz), 0.0, 1.0)

    # Normalize weights.
    stack = np.stack([water_w, soil_w, grass_w, rock_w, snow_w], axis=0)
    weight_sum = np.sum(stack, axis=0) + 1e-8
    weights = stack / weight_sum
    water_w, soil_w, grass_w, rock_w, snow_w = weights

    # Base material colors (R, G, B).
    water_color = np.array([70.0, 135.0, 205.0], dtype=np.float32)
    soil_color = np.array([132.0, 96.0, 62.0], dtype=np.float32)
    grass_color = np.array([88.0, 142.0, 76.0], dtype=np.float32)
    rock_color = np.array([138.0, 131.0, 123.0], dtype=np.float32)
    snow_color = np.array([236.0, 239.0, 244.0], dtype=np.float32)

    rgb = (
        water_w[..., None] * water_color
        + soil_w[..., None] * soil_color
        + grass_w[..., None] * grass_color
        + rock_w[..., None] * rock_color
        + snow_w[..., None] * snow_color
    )

    # Relief shading from normals + AO-like local shadowing.
    hillshade, ao = _compute_relief_shading(heightmap, nx, ny, nz)
    shade = hillshade * ao
    rgb = rgb * shade[..., None]

    # Slight elevation warmth in lowlands and coolness at peaks.
    elev = heightmap.astype(np.float32)
    warm_tint = np.array([1.04, 1.00, 0.96], dtype=np.float32)
    cool_tint = np.array([0.95, 0.98, 1.03], dtype=np.float32)
    tint = (1.0 - elev)[..., None] * warm_tint + elev[..., None] * cool_tint
    rgb = rgb * tint

    rgb_image = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
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
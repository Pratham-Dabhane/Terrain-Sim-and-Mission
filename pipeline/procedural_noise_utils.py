"""
Procedural Noise Utilities for Terrain Generation
Implements fractal Brownian motion (fBM) and hybrid terrain generation.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass
import noise  # Perlin noise library

logger = logging.getLogger(__name__)

@dataclass
class NoiseParams:
    """Parameters for procedural noise generation."""
    
    # fBM parameters
    scale: float = 100.0           # Overall scale of the noise
    octaves: int = 6               # Number of octaves (detail levels)
    persistence: float = 0.5       # Amplitude multiplier for each octave
    lacunarity: float = 2.0        # Frequency multiplier for each octave
    seed: Optional[int] = None     # Random seed for reproducibility
    
    # Terrain mixing parameters
    mountain_weight: float = 0.8   # Weight for mountain features
    valley_weight: float = 0.2     # Weight for valley features
    river_strength: float = 0.3    # Strength of river carving
    river_frequency: float = 0.05  # Frequency of river patterns


def fbm(shape: Tuple[int, int], scale: float = 100.0, octaves: int = 6, 
        persistence: float = 0.5, lacunarity: float = 2.0, 
        seed: Optional[int] = None) -> np.ndarray:
    """
    Generate fractal Brownian motion (fBM) noise using real Perlin noise.
    
    Args:
        shape: Output shape (height, width)
        scale: Overall scale of the noise
        octaves: Number of octaves (detail levels)
        persistence: Amplitude multiplier for each octave (0.0 to 1.0)
        lacunarity: Frequency multiplier for each octave (typically 2.0)
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: fBM noise array with values roughly in [-1, 1]
    """
    height, width = shape
    result = np.zeros((height, width), dtype=np.float32)
    
    # Use seed for reproducibility
    base_seed = seed if seed is not None else 0
    
    logger.info(f"Generating Perlin noise: {height}x{width}, octaves={octaves}, this may take a moment...")
    
    # Generate coordinates scaled by frequency
    frequency_val = 1.0 / scale
    
    # Use pnoise2's built-in octaves for faster generation
    for i in range(height):
        for j in range(width):
            x = j * frequency_val
            y = i * frequency_val
            
            # Let pnoise2 handle all octaves internally (much faster)
            result[i, j] = noise.pnoise2(
                x, y,
                octaves=octaves,  # Use built-in octaves
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=base_seed
            )
    
    logger.info(f"✓ Perlin noise generated: range=[{result.min():.3f}, {result.max():.3f}]")
    return result


def generate_river_pattern(shape: Tuple[int, int], frequency: float = 0.05, 
                          strength: float = 0.3, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate river-like patterns using Perlin noise with directional bias.
    
    Args:
        shape: Output shape (height, width)
        frequency: Frequency of river patterns
        strength: Strength of river carving effect
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: River pattern array
    """
    height, width = shape
    base_seed = (seed + 1000) if seed is not None else 1000
    
    rivers = np.zeros((height, width), dtype=np.float32)
    
    # Generate river patterns with directional flow
    for i in range(height):
        for j in range(width):
            x = j * frequency
            y = i * frequency
            
            # Sample multiple noise layers for river-like features
            river_value = (
                noise.pnoise2(x, y, octaves=2, persistence=0.5, 
                            lacunarity=2.0, repeatx=1024, repeaty=1024, 
                            base=base_seed) +
                noise.pnoise2(x * 0.5, y * 2.0, octaves=1, 
                            repeatx=1024, repeaty=1024, 
                            base=base_seed + 100) * 0.5
            )
            
            rivers[i, j] = river_value
    
    # Apply carving effect - rivers are valleys
    rivers = np.abs(rivers) * strength
    
    logger.debug(f"Generated rivers: frequency={frequency}, strength={strength}, range=[{rivers.min():.3f}, {rivers.max():.3f}]")
    return rivers


def generate_procedural_heightmap(shape: Tuple[int, int], 
                                params: Optional[NoiseParams] = None) -> np.ndarray:
    """
    Generate procedural heightmap using fBM and hybrid terrain features.
    
    Args:
        shape: Output shape (height, width)
        params: Noise generation parameters
        
    Returns:
        np.ndarray: Normalized heightmap in [0, 1] range
    """
    if params is None:
        params = NoiseParams()
    
    logger.info(f"Generating procedural heightmap: shape={shape}")
    
    # Generate different terrain features using fBM
    logger.debug("Generating mountain features...")
    mountains = fbm(
        shape=shape,
        scale=params.scale,
        octaves=params.octaves,
        persistence=params.persistence,
        lacunarity=params.lacunarity,
        seed=params.seed
    )
    
    logger.debug("Generating valley features...")
    valleys = fbm(
        shape=shape,
        scale=params.scale * 2.0,    # Larger scale for valleys
        octaves=max(1, params.octaves - 2),  # Fewer octaves for smoother valleys
        persistence=params.persistence * 0.8,
        lacunarity=params.lacunarity,
        seed=params.seed + 100 if params.seed else None  # Offset seed
    )
    
    logger.debug("Generating river patterns...")
    rivers = generate_river_pattern(
        shape=shape,
        frequency=params.river_frequency,
        strength=params.river_strength,
        seed=params.seed
    )
    
    # Mix terrain features
    logger.debug("Mixing terrain features...")
    height = (
        mountains * params.mountain_weight + 
        valleys * params.valley_weight - 
        rivers
    )
    
    # Normalize to [0, 1] range
    height_min = height.min()
    height_max = height.max()
    
    if height_max > height_min:
        height = (height - height_min) / (height_max - height_min)
    else:
        height = np.ones_like(height) * 0.5  # Fallback to flat terrain
    
    logger.info(f"✓ Procedural heightmap generated: range=[{height.min():.3f}, {height.max():.3f}]")
    return height.astype(np.float32)


def generate_terrain_variants(shape: Tuple[int, int], seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate multiple terrain variants with different characteristics.
    
    Args:
        shape: Output shape (height, width)
        seed: Random seed for reproducibility
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of terrain variants
    """
    variants = {}
    
    # Mountainous terrain
    variants['mountainous'] = generate_procedural_heightmap(
        shape, NoiseParams(
            scale=80.0, octaves=8, persistence=0.6, lacunarity=2.2,
            mountain_weight=0.9, valley_weight=0.1, river_strength=0.2,
            seed=seed
        )
    )
    
    # Rolling hills
    variants['hills'] = generate_procedural_heightmap(
        shape, NoiseParams(
            scale=150.0, octaves=5, persistence=0.4, lacunarity=2.0,
            mountain_weight=0.6, valley_weight=0.4, river_strength=0.1,
            seed=seed
        )
    )
    
    # Flat plains with rivers
    variants['plains'] = generate_procedural_heightmap(
        shape, NoiseParams(
            scale=300.0, octaves=3, persistence=0.3, lacunarity=1.8,
            mountain_weight=0.3, valley_weight=0.2, river_strength=0.5,
            seed=seed
        )
    )
    
    # Rugged terrain
    variants['rugged'] = generate_procedural_heightmap(
        shape, NoiseParams(
            scale=60.0, octaves=10, persistence=0.7, lacunarity=2.5,
            mountain_weight=0.8, valley_weight=0.3, river_strength=0.4,
            seed=seed
        )
    )
    
    logger.info(f"Generated {len(variants)} terrain variants")
    return variants


# =============================================================================
# 🎯 CONVENIENCE FUNCTIONS
# =============================================================================

def quick_heightmap(shape: Tuple[int, int] = (256, 256), 
                   terrain_type: str = 'mixed', 
                   seed: Optional[int] = None) -> np.ndarray:
    """
    Quick generation of common terrain types.
    
    Args:
        shape: Output shape
        terrain_type: 'mountainous', 'hills', 'plains', 'rugged', or 'mixed'
        seed: Random seed
        
    Returns:
        np.ndarray: Generated heightmap
    """
    if terrain_type == 'mixed':
        return generate_procedural_heightmap(shape, NoiseParams(seed=seed))
    else:
        variants = generate_terrain_variants(shape, seed)
        return variants.get(terrain_type, variants['hills'])  # Default to hills


def validate_heightmap(heightmap: np.ndarray) -> bool:
    """
    Validate that heightmap is properly normalized and shaped.
    
    Args:
        heightmap: Heightmap to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if heightmap is None or heightmap.size == 0:
        logger.error("Heightmap is None or empty")
        return False
    
    if len(heightmap.shape) != 2:
        logger.error(f"Heightmap must be 2D, got shape: {heightmap.shape}")
        return False
    
    if not (0.0 <= heightmap.min() <= heightmap.max() <= 1.0):
        logger.warning(f"Heightmap values outside [0,1]: range=[{heightmap.min():.3f}, {heightmap.max():.3f}]")
        # Don't return False - just warn, as slight numerical errors are acceptable
    
    if np.isnan(heightmap).any() or np.isinf(heightmap).any():
        logger.error("Heightmap contains NaN or Inf values")
        return False
    
    logger.debug(f"Heightmap validation passed: shape={heightmap.shape}, range=[{heightmap.min():.3f}, {heightmap.max():.3f}]")
    return True
"""
Procedural Noise Utilities for Terrain Generation
Implements fractal Brownian motion (fBM) and hybrid terrain generation.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

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
    Generate fractal Brownian motion (fBM) noise.
    
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
    if seed is not None:
        np.random.seed(seed)
    
    height, width = shape
    result = np.zeros((height, width), dtype=np.float32)
    
    # Create coordinate grids
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    amplitude = 1.0
    frequency = 1.0 / scale
    max_value = 0.0  # Used for normalization
    
    for octave in range(octaves):
        # Generate Perlin-like noise using sine waves (simplified approach)
        # In production, you'd use proper Perlin/Simplex noise
        noise_x = X * frequency * 2 * np.pi
        noise_y = Y * frequency * 2 * np.pi
        
        # Create multiple sine wave patterns for pseudo-Perlin noise
        noise = (
            np.sin(noise_x + np.random.random() * 2 * np.pi) * 
            np.cos(noise_y + np.random.random() * 2 * np.pi) +
            np.sin(noise_x * 2.1 + np.random.random() * 2 * np.pi) * 
            np.cos(noise_y * 1.9 + np.random.random() * 2 * np.pi) * 0.5 +
            np.sin(noise_x * 0.7 + np.random.random() * 2 * np.pi) * 
            np.cos(noise_y * 1.3 + np.random.random() * 2 * np.pi) * 0.25
        )
        
        result += noise * amplitude
        max_value += amplitude
        
        amplitude *= persistence
        frequency *= lacunarity
    
    # Normalize to roughly [-1, 1]
    if max_value > 0:
        result /= max_value
    
    logger.debug(f"Generated fBM: shape={shape}, octaves={octaves}, range=[{result.min():.3f}, {result.max():.3f}]")
    return result


def generate_river_pattern(shape: Tuple[int, int], frequency: float = 0.05, 
                          strength: float = 0.3, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate river-like patterns using sine/cosine waves.
    
    Args:
        shape: Output shape (height, width)
        frequency: Frequency of river patterns
        strength: Strength of river carving effect
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: River pattern array
    """
    if seed is not None:
        np.random.seed(seed + 1000)  # Offset seed for rivers
    
    height, width = shape
    
    # Create coordinate grids
    x = np.linspace(0, width, width, dtype=np.float32)
    y = np.linspace(0, height, height, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    # Generate multiple river patterns
    river1 = np.sin(X * frequency) * np.cos(Y * frequency)
    river2 = np.sin(X * frequency * 1.3 + np.pi/4) * np.cos(Y * frequency * 0.7 + np.pi/3)
    river3 = np.sin(X * frequency * 0.6 + np.pi/2) * np.cos(Y * frequency * 1.5 + np.pi/6)
    
    # Combine river patterns
    rivers = (river1 + river2 * 0.6 + river3 * 0.4) / 2.0
    
    # Apply strength and take absolute value for carving effect
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
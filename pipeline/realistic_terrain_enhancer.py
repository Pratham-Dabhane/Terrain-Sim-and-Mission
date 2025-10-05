"""
Realistic Terrain Enhancement System
Creates photorealistic terrain textures from heightmaps using CV2 and intelligent color mapping.
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class RealisticTerrainEnhancer:
    """
    Enhances grayscale heightmaps into realistic terrain images with proper coloring,
    textures, and environmental effects.
    """
    
    def __init__(self):
        # Define terrain color palettes for different biomes
        self.biome_colors = {
            'mountain': {
                'snow': np.array([240, 248, 255]),      # Snow white
                'rock': np.array([139, 137, 137]),      # Dark gray
                'stone': np.array([169, 169, 169]),     # Gray
                'vegetation': np.array([34, 139, 34]),   # Forest green
                'bare': np.array([160, 82, 45]),        # Saddle brown
            },
            'desert': {
                'sand': np.array([238, 203, 173]),      # Navajo white
                'dune': np.array([210, 180, 140]),      # Tan
                'rock': np.array([139, 69, 19]),        # Saddle brown
                'oasis': np.array([0, 100, 0]),         # Dark green
            },
            'forest': {
                'tree_dense': np.array([0, 100, 0]),    # Dark green
                'tree_light': np.array([34, 139, 34]),  # Forest green  
                'grass': np.array([124, 252, 0]),       # Lawn green
                'soil': np.array([160, 82, 45]),        # Saddle brown
            },
            'arctic': {
                'ice': np.array([240, 248, 255]),       # Alice blue
                'snow': np.array([255, 250, 250]),      # Snow
                'rock': np.array([112, 128, 144]),      # Slate gray
                'water': np.array([70, 130, 180]),      # Steel blue
            }
        }
        
    def detect_biome_from_prompt(self, prompt: str) -> str:
        """Detect the terrain biome from the text prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['snow', 'mountain', 'peak', 'alpine', 'cold']):
            return 'mountain'
        elif any(word in prompt_lower for word in ['desert', 'sand', 'dune', 'arid', 'hot']):
            return 'desert'
        elif any(word in prompt_lower for word in ['forest', 'tree', 'jungle', 'green', 'woods']):
            return 'forest'
        elif any(word in prompt_lower for word in ['arctic', 'ice', 'frozen', 'tundra', 'glacier']):
            return 'arctic'
        else:
            return 'mountain'  # Default
            
    def create_elevation_mask(self, heightmap: np.ndarray, threshold_low: float = 0.3, 
                            threshold_mid: float = 0.6, threshold_high: float = 0.8) -> dict:
        """Create elevation-based masks for different terrain features."""
        masks = {}
        
        # Normalize heightmap
        normalized = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        
        masks['low'] = normalized < threshold_low       # Valleys, water
        masks['mid_low'] = (normalized >= threshold_low) & (normalized < threshold_mid)  # Plains
        masks['mid_high'] = (normalized >= threshold_mid) & (normalized < threshold_high) # Hills
        masks['high'] = normalized >= threshold_high    # Peaks, snow
        
        return masks
        
    def apply_gradient_mapping(self, heightmap: np.ndarray) -> np.ndarray:
        """Apply slope-based gradient mapping for more realistic terrain features."""
        # Calculate gradients (slopes)
        grad_x = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / \
                           (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)
        
        return gradient_magnitude
        
    def add_texture_noise(self, image: np.ndarray, texture_strength: float = 0.1) -> np.ndarray:
        """Add subtle texture noise for more realistic appearance."""
        noise = np.random.normal(0, texture_strength * 255, image.shape).astype(np.int16)
        textured = image.astype(np.int16) + noise
        return np.clip(textured, 0, 255).astype(np.uint8)
        
    def enhance_heightmap(self, heightmap: np.ndarray, prompt: str = "", 
                         apply_lighting: bool = True, texture_strength: float = 0.05) -> np.ndarray:
        """
        Transform a grayscale heightmap into a realistic colored terrain image.
        
        Args:
            heightmap: 2D numpy array with height values
            prompt: Text prompt used to determine biome/style
            apply_lighting: Whether to apply lighting effects
            texture_strength: Strength of texture noise (0-1)
            
        Returns:
            RGB terrain image as numpy array
        """
        # Detect biome
        biome = self.detect_biome_from_prompt(prompt)
        colors = self.biome_colors[biome]
        
        logger.info(f"Enhancing terrain with {biome} biome")
        
        # Ensure heightmap is in correct format
        if len(heightmap.shape) == 3:
            heightmap = cv2.cvtColor(heightmap, cv2.COLOR_RGB2GRAY)
        
        # Normalize heightmap to 0-1
        heightmap_norm = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
        
        # Create elevation masks
        masks = self.create_elevation_mask(heightmap_norm)
        
        # Calculate gradients for slope-based coloring
        gradient = self.apply_gradient_mapping(heightmap_norm)
        
        # Initialize RGB image
        h, w = heightmap.shape
        enhanced_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply biome-specific coloring based on elevation and slope
        if biome == 'mountain':
            # Snow on high elevations
            snow_mask = masks['high']
            enhanced_image[snow_mask] = colors['snow']
            
            # Rock on steep slopes  
            steep_slopes = gradient > 0.4
            rock_mask = steep_slopes & ~snow_mask
            enhanced_image[rock_mask] = colors['rock']
            
            # Vegetation on moderate slopes and mid elevations
            vegetation_mask = masks['mid_low'] & ~steep_slopes
            enhanced_image[vegetation_mask] = colors['vegetation']
            
            # Stone/bare rock for remaining areas
            remaining_mask = ~(snow_mask | rock_mask | vegetation_mask)
            enhanced_image[remaining_mask] = colors['stone']
            
        elif biome == 'desert':
            # Sand in low areas
            enhanced_image[masks['low']] = colors['sand']
            enhanced_image[masks['mid_low']] = colors['dune']
            enhanced_image[masks['mid_high']] = colors['dune']
            enhanced_image[masks['high']] = colors['rock']
            
        elif biome == 'forest':
            # Dense trees on slopes
            steep_areas = gradient > 0.3
            enhanced_image[steep_areas] = colors['tree_dense']
            enhanced_image[~steep_areas & masks['mid_high']] = colors['tree_light']
            enhanced_image[masks['mid_low']] = colors['grass']
            enhanced_image[masks['low']] = colors['soil']
            
        elif biome == 'arctic':
            # Ice and snow dominate
            enhanced_image[masks['high']] = colors['snow']
            enhanced_image[masks['mid_high']] = colors['ice']
            enhanced_image[masks['mid_low']] = colors['ice']
            enhanced_image[masks['low']] = colors['water']
            
        # Apply lighting effects
        if apply_lighting:
            enhanced_image = self._apply_lighting(enhanced_image, heightmap_norm, gradient)
            
        # Add texture noise
        if texture_strength > 0:
            enhanced_image = self.add_texture_noise(enhanced_image, texture_strength)
            
        return enhanced_image
        
    def _apply_lighting(self, image: np.ndarray, heightmap: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Apply realistic lighting effects based on elevation and slope."""
        # Simulate sunlight from top-left
        light_direction = np.array([-1, -1])  # Top-left
        
        # Calculate normal vectors (simplified)
        grad_x = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate lighting intensity
        lighting = np.abs(grad_x * light_direction[0] + grad_y * light_direction[1])
        lighting = (lighting - lighting.min()) / (lighting.max() - lighting.min() + 1e-8)
        
        # Apply lighting (brighten sunny slopes, darken shadowed areas)
        lighting_factor = 0.7 + 0.6 * lighting  # Range from 0.7 to 1.3
        
        # Apply to each color channel
        lit_image = image.astype(np.float32)
        for i in range(3):
            lit_image[:, :, i] *= lighting_factor
            
        return np.clip(lit_image, 0, 255).astype(np.uint8)
        
    def create_comparison_visualization(self, original: np.ndarray, enhanced: np.ndarray, 
                                     prompt: str, save_path: str = None) -> np.ndarray:
        """Create a side-by-side comparison visualization."""
        # Ensure original is grayscale converted to RGB
        if len(original.shape) == 2:
            original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original
            
        # Create side-by-side comparison
        comparison = np.hstack([original_rgb, enhanced])
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            logger.info(f"Comparison saved to: {save_path}")
            
        return comparison

# Example usage and testing
if __name__ == "__main__":
    enhancer = RealisticTerrainEnhancer()
    
    # Test with sample data
    test_heightmap = np.random.rand(256, 256)
    test_prompt = "snow-covered mountain peaks with deep valleys"
    
    enhanced = enhancer.enhance_heightmap(test_heightmap, test_prompt)
    print(f"Enhanced terrain shape: {enhanced.shape}")
    print(f"Enhanced terrain range: [{enhanced.min()}, {enhanced.max()}]")
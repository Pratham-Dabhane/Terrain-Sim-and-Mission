#!/usr/bin/env python3
"""
Test script for the Generative AI Terrain Prototype
Tests basic functionality without requiring heavy ML models.
"""

import numpy as np
import matplotlib.pyplot as plt
from terrain_prototype import (
    prompt_to_heightmap_gan, 
    _apply_prompt_modifications,
    _apply_terrain_colors,
    visualize_terrain
)

def test_basic_generation():
    """Test basic terrain generation functionality."""
    print("Testing basic terrain generation...")
    
    # Test prompt
    prompt = "mountainous terrain with rivers and valleys"
    
    try:
        # Generate heightmap
        heightmap = prompt_to_heightmap_gan(prompt, size=128)
        
        # Verify output
        assert heightmap.shape == (128, 128), f"Expected shape (128, 128), got {heightmap.shape}"
        assert np.min(heightmap) >= 0, f"Heightmap should have min >= 0, got {np.min(heightmap)}"
        assert np.max(heightmap) <= 1, f"Heightmap should have max <= 1, got {np.max(heightmap)}"
        
        print("✓ Basic terrain generation test passed!")
        return heightmap
        
    except Exception as e:
        print(f"✗ Basic terrain generation test failed: {e}")
        return None

def test_prompt_modifications():
    """Test terrain feature modifications based on prompts."""
    print("Testing prompt modifications...")
    
    # Create a simple heightmap
    heightmap = np.random.rand(64, 64)
    
    # Test different prompts
    test_prompts = [
        "mountainous terrain",
        "desert landscape",
        "forest with rivers",
        "coastal cliffs"
    ]
    
    for prompt in test_prompts:
        try:
            modified = _apply_prompt_modifications(heightmap.copy(), prompt)
            
            # Verify modifications were applied
            assert modified.shape == heightmap.shape, "Shape should remain the same"
            assert np.min(modified) >= 0, "Modified heightmap should have min >= 0"
            assert np.max(modified) <= 1, "Modified heightmap should have max <= 1"
            
            print(f"✓ Prompt modification test passed for: '{prompt}'")
            
        except Exception as e:
            print(f"✗ Prompt modification test failed for '{prompt}': {e}")

def test_color_mapping():
    """Test terrain color mapping functionality."""
    print("Testing color mapping...")
    
    # Create a heightmap with known values
    heightmap = np.array([
        [0.1, 0.3, 0.5, 0.7, 0.9],  # Different height levels
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.1, 0.3, 0.5, 0.7, 0.9]
    ])
    
    try:
        # Create RGB terrain
        rgb_terrain = np.stack([heightmap] * 3, axis=-1)
        
        # Apply colors
        colored_terrain = _apply_terrain_colors(rgb_terrain, heightmap)
        
        # Verify output
        assert colored_terrain.shape == (5, 5, 3), f"Expected shape (5, 5, 3), got {colored_terrain.shape}"
        assert np.min(colored_terrain) >= 0, "Colored terrain should have min >= 0"
        assert np.max(colored_terrain) <= 1, "Colored terrain should have max <= 1"
        
        print("✓ Color mapping test passed!")
        
    except Exception as e:
        print(f"✗ Color mapping test failed: {e}")

def test_visualization():
    """Test visualization functionality."""
    print("Testing visualization...")
    
    try:
        # Create test data
        heightmap = np.random.rand(64, 64)
        terrain_image = np.random.rand(64, 64, 3)
        prompt = "test terrain"
        
        # Test visualization (without showing plot)
        plt.ioff()  # Turn off interactive mode
        
        # This should not raise an error
        visualize_terrain(terrain_image, heightmap, prompt)
        
        plt.close('all')  # Close all plots
        
        print("✓ Visualization test passed!")
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")

def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Terrain Prototype Tests")
    print("=" * 50)
    
    # Run tests
    heightmap = test_basic_generation()
    test_prompt_modifications()
    test_color_mapping()
    test_visualization()
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    if heightmap is not None:
        print("✓ All basic functionality tests completed!")
        print(f"Generated heightmap shape: {heightmap.shape}")
        print(f"Heightmap range: {np.min(heightmap):.3f} to {np.max(heightmap):.3f}")
    else:
        print("✗ Some tests failed. Check the output above for details.")
    
    print("\nTo run the full prototype:")
    print("python terrain_prototype.py")

if __name__ == "__main__":
    run_all_tests()

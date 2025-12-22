"""
Unit tests for prompt_parser.py and procedural_noise_utils.py
Tests prompt parsing and terrain generation logic.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.prompt_parser import parse_prompt_to_params
from pipeline.procedural_noise_utils import generate_procedural_heightmap, NoiseParams


class TestPromptParameters(unittest.TestCase):
    """Test prompt parsing functionality"""
    
    def test_mountain_detection(self):
        """Test that mountain keywords are detected"""
        params = parse_prompt_to_params("snowy mountain peaks")
        self.assertIn('mountain_octaves', params)
        self.assertGreater(params.get('mountain_octaves', 0), 5)
    
    def test_desert_detection(self):
        """Test that desert keywords are detected"""
        params = parse_prompt_to_params("sandy desert dunes")
        self.assertIn('mountain_octaves', params)  # Even deserts have octaves
    
    def test_empty_prompt(self):
        """Test behavior with empty prompt"""
        params = parse_prompt_to_params("")
        # Should return default parameters
        self.assertIsInstance(params, dict)
        self.assertIn('mountain_octaves', params)
    
    def test_complex_prompt(self):
        """Test parsing of complex multi-keyword prompt"""
        params = parse_prompt_to_params("rugged mountain peaks with steep valleys")
        self.assertIn('mountain_octaves', params)
        # Rugged should increase octaves
        self.assertGreaterEqual(params.get('mountain_octaves', 0), 8)
    
    def test_scale_parameter(self):
        """Test that scale parameter is extracted"""
        params = parse_prompt_to_params("mountain")
        self.assertIn('scale', params)
        self.assertGreater(params['scale'], 0)
    
    def test_persistence_range(self):
        """Test persistence is in valid range"""
        params = parse_prompt_to_params("smooth hills")
        self.assertIn('persistence', params)
        self.assertGreaterEqual(params['persistence'], 0.0)
        self.assertLessEqual(params['persistence'], 1.0)
    
    def test_octaves_variation(self):
        """Test different terrains have different octaves"""
        smooth = parse_prompt_to_params("smooth plains")
        rugged = parse_prompt_to_params("rugged mountains")
        
        self.assertLess(smooth['mountain_octaves'], rugged['mountain_octaves'])
    
    def test_seed_generation(self):
        """Test seed parameter handling"""
        params1 = parse_prompt_to_params("mountain")
        params2 = parse_prompt_to_params("mountain")
        
        # Both should have default parameters
        self.assertIsInstance(params1, dict)
        self.assertIsInstance(params2, dict)


class TestTerrainGeneration(unittest.TestCase):
    """Test heightmap generation"""
    
    def test_heightmap_shape(self):
        """Test that generated heightmap has correct shape"""
        shape = (64, 64)
        params = NoiseParams(octaves=4, scale=5.0)
        heightmap = generate_procedural_heightmap(shape, params)
        
        self.assertEqual(heightmap.shape, shape)
    
    def test_heightmap_range(self):
        """Test heightmap values are normalized to [0, 1]"""
        params = NoiseParams(octaves=6, scale=5.0)
        heightmap = generate_procedural_heightmap((128, 128), params)
        
        self.assertGreaterEqual(heightmap.min(), 0.0)
        self.assertLessEqual(heightmap.max(), 1.0)
    
    def test_heightmap_variation(self):
        """Test that heightmap has reasonable variation"""
        params = NoiseParams(octaves=6, scale=5.0)
        heightmap = generate_procedural_heightmap((128, 128), params)
        
        # Check standard deviation > 0 (not all same value)
        std_dev = np.std(heightmap)
        self.assertGreater(std_dev, 0.01)
    
    def test_seed_reproducibility(self):
        """Test same seed produces same terrain"""
        seed = 42
        params = NoiseParams(octaves=4, scale=5.0, seed=seed)
        
        heightmap1 = generate_procedural_heightmap((64, 64), params)
        heightmap2 = generate_procedural_heightmap((64, 64), params)
        
        np.testing.assert_array_almost_equal(heightmap1, heightmap2)


if __name__ == '__main__':
    unittest.main()

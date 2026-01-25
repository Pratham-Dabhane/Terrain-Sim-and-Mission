"""Unit tests for terrain analysis and biome mask generation.

Focus:
- Terrain attributes are computed correctly and stay in valid ranges.
- Biome masks are in [0, 1] and behave as expected.
- ENABLE_BIOMES flag correctly controls analysis.
- Edge cases (flat terrain, extreme slopes, etc.) are handled.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import terrain_analysis


class TestTerrainAttributes(unittest.TestCase):
    """Tests for terrain attribute computation."""

    def setUp(self) -> None:
        # Create a simple test heightmap with known features
        self.height = np.zeros((64, 64), dtype=np.float32)
        
        # Add a peak in the center
        center = 32
        y, x = np.ogrid[:64, :64]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        self.height = 0.5 * np.exp(-dist**2 / 200.0)
        
        # Ensure normalized
        self.height = np.clip(self.height, 0.0, 1.0).astype(np.float32)

    def test_slope_in_range(self) -> None:
        slope = terrain_analysis.compute_slope(self.height)
        
        self.assertGreaterEqual(slope.min(), 0.0)
        self.assertLessEqual(slope.max(), 1.0)
        self.assertEqual(slope.shape, self.height.shape)

    def test_slope_flat_terrain(self) -> None:
        flat = np.ones((32, 32), dtype=np.float32) * 0.5
        slope = terrain_analysis.compute_slope(flat)
        
        # Flat terrain should have near-zero slope
        self.assertLess(slope.max(), 0.01)

    def test_curvature_in_range(self) -> None:
        convexity, concavity = terrain_analysis.compute_curvature(self.height)
        
        self.assertGreaterEqual(convexity.min(), 0.0)
        self.assertLessEqual(convexity.max(), 1.0)
        self.assertGreaterEqual(concavity.min(), 0.0)
        self.assertLessEqual(concavity.max(), 1.0)

    def test_aspect_in_range(self) -> None:
        aspect = terrain_analysis.compute_aspect(self.height)
        
        self.assertGreaterEqual(aspect.min(), 0.0)
        self.assertLessEqual(aspect.max(), 2 * np.pi)

    def test_distance_to_water_no_water(self) -> None:
        # Heightmap with no water (all above threshold)
        high_terrain = np.ones((32, 32), dtype=np.float32) * 0.5
        distance = terrain_analysis.compute_distance_to_water(high_terrain, water_threshold=0.05)
        
        # Should return large distances everywhere
        self.assertGreater(distance.min(), 50.0)

    def test_distance_to_water_with_water(self) -> None:
        # Heightmap with water region
        mixed = np.ones((32, 32), dtype=np.float32) * 0.5
        mixed[:10, :10] = 0.02  # Water in corner
        
        distance = terrain_analysis.compute_distance_to_water(mixed, water_threshold=0.05)
        
        # Water pixels should have distance ~0
        self.assertLess(distance[5, 5], 1.0)
        
        # Far corner should have larger distance
        self.assertGreater(distance[30, 30], 20.0)


class TestBiomeMasks(unittest.TestCase):
    """Tests for biome mask generation."""

    def setUp(self) -> None:
        # Synthetic heightmap with diverse features
        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 1, 128)
        xv, yv = np.meshgrid(x, y)
        
        # Gradient from low (left) to high (right) with some noise
        self.height = xv * 0.8 + 0.1 * np.sin(8 * np.pi * yv)
        self.height = np.clip(self.height, 0.0, 1.0).astype(np.float32)
        
        self.params = terrain_analysis.BiomeParams()

    def test_biome_masks_in_range(self) -> None:
        slope = terrain_analysis.compute_slope(self.height)
        distance = terrain_analysis.compute_distance_to_water(self.height, self.params.water_elevation_threshold)
        
        masks = terrain_analysis.compute_biome_masks(self.height, slope, distance, self.params)
        
        for biome_name, mask in masks.items():
            with self.subTest(biome=biome_name):
                self.assertGreaterEqual(mask.min(), 0.0, f"{biome_name} mask has negative values")
                self.assertLessEqual(mask.max(), 1.0, f"{biome_name} mask exceeds 1.0")
                self.assertFalse(np.isnan(mask).any(), f"{biome_name} mask contains NaN")
                self.assertFalse(np.isinf(mask).any(), f"{biome_name} mask contains Inf")

    def test_snow_biome_high_elevation(self) -> None:
        # High elevation should favor snow
        high_terrain = np.ones((32, 32), dtype=np.float32) * 0.9
        slope = terrain_analysis.compute_slope(high_terrain)
        distance = terrain_analysis.compute_distance_to_water(high_terrain, self.params.water_elevation_threshold)
        
        masks = terrain_analysis.compute_biome_masks(high_terrain, slope, distance, self.params)
        
        # Snow mask should be strong at high elevation
        self.assertGreater(masks['snow'].mean(), 0.5)

    def test_water_biome_low_elevation(self) -> None:
        # Very low elevation should be water
        low_terrain = np.ones((32, 32), dtype=np.float32) * 0.02
        slope = terrain_analysis.compute_slope(low_terrain)
        distance = terrain_analysis.compute_distance_to_water(low_terrain, self.params.water_elevation_threshold)
        
        masks = terrain_analysis.compute_biome_masks(low_terrain, slope, distance, self.params)
        
        # Water mask should be strong at very low elevation
        self.assertGreater(masks['water'].mean(), 0.8)

    def test_rock_biome_steep_slopes(self) -> None:
        # Create steep terrain
        x = np.linspace(0, 1, 32)
        steep = np.outer(x**2, np.ones(32))  # Steep gradient
        steep = steep.astype(np.float32)
        
        slope = terrain_analysis.compute_slope(steep)
        distance = terrain_analysis.compute_distance_to_water(steep, self.params.water_elevation_threshold)
        
        masks = terrain_analysis.compute_biome_masks(steep, slope, distance, self.params)
        
        # Rock mask should be stronger in steep regions
        self.assertGreater(masks['rock'].max(), 0.3)

    def test_dominant_biome_computation(self) -> None:
        slope = terrain_analysis.compute_slope(self.height)
        distance = terrain_analysis.compute_distance_to_water(self.height, self.params.water_elevation_threshold)
        
        masks = terrain_analysis.compute_biome_masks(self.height, slope, distance, self.params)
        dominant_idx, dominant_strength = terrain_analysis.compute_dominant_biome(masks)
        
        # Check shapes
        self.assertEqual(dominant_idx.shape, self.height.shape)
        self.assertEqual(dominant_strength.shape, self.height.shape)
        
        # Check ranges
        n_biomes = len(masks)
        self.assertGreaterEqual(dominant_idx.min(), 0)
        self.assertLess(dominant_idx.max(), n_biomes)
        self.assertGreaterEqual(dominant_strength.min(), 0.0)
        self.assertLessEqual(dominant_strength.max(), 1.0)


class TestAnalyzeTerrainIntegration(unittest.TestCase):
    """Tests for the main analyze_terrain function."""

    def setUp(self) -> None:
        # Simple test terrain
        x = np.linspace(0, 1, 64)
        y = np.linspace(0, 1, 64)
        xv, yv = np.meshgrid(x, y)
        self.height = 0.5 + 0.3 * np.sin(2 * np.pi * xv) * np.cos(2 * np.pi * yv)
        self.height = np.clip(self.height, 0.0, 1.0).astype(np.float32)

    def test_analyze_terrain_returns_expected_keys(self) -> None:
        result = terrain_analysis.analyze_terrain(self.height)
        
        expected_keys = {
            'elevation', 'slope', 'convexity', 'concavity', 
            'aspect', 'distance_to_water', 'biome_masks', 'dominant_biome'
        }
        
        self.assertTrue(expected_keys.issubset(result.keys()))

    def test_analyze_terrain_biomes_disabled(self) -> None:
        # Temporarily disable biomes
        original_flag = terrain_analysis.ENABLE_BIOMES
        terrain_analysis.ENABLE_BIOMES = False
        
        try:
            result = terrain_analysis.analyze_terrain(self.height)
            
            # Should return minimal data when disabled
            self.assertIn('elevation', result)
            self.assertIn('biome_masks', result)
            self.assertEqual(len(result['biome_masks']), 0)
            
        finally:
            terrain_analysis.ENABLE_BIOMES = original_flag

    def test_analyze_terrain_biomes_enabled(self) -> None:
        # Ensure biomes are enabled
        original_flag = terrain_analysis.ENABLE_BIOMES
        terrain_analysis.ENABLE_BIOMES = True
        
        try:
            result = terrain_analysis.analyze_terrain(self.height)
            
            # Should compute all biomes
            self.assertGreater(len(result['biome_masks']), 0)
            self.assertIn('snow', result['biome_masks'])
            self.assertIn('water', result['biome_masks'])
            
        finally:
            terrain_analysis.ENABLE_BIOMES = original_flag

    def test_analyze_terrain_with_custom_params(self) -> None:
        custom_params = terrain_analysis.BiomeParams(
            snow_elevation_min=0.8,
            water_elevation_threshold=0.1
        )
        
        result = terrain_analysis.analyze_terrain(self.height, params=custom_params)
        
        # Should use custom parameters
        self.assertIsNotNone(result)
        self.assertIn('biome_masks', result)


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def test_sigmoid_behavior(self) -> None:
        x = np.linspace(0, 1, 100)
        sig = terrain_analysis._sigmoid(x, center=0.5, width=0.1)
        
        # Should be in [0, 1]
        self.assertGreaterEqual(sig.min(), 0.0)
        self.assertLessEqual(sig.max(), 1.0)
        
        # Should be ~0.5 at center
        self.assertAlmostEqual(sig[50], 0.5, places=1)
        
        # Should be monotonic increasing
        self.assertTrue(np.all(np.diff(sig) >= 0))

    def test_inverse_sigmoid_behavior(self) -> None:
        x = np.linspace(0, 1, 100)
        inv_sig = terrain_analysis._inverse_sigmoid(x, center=0.5, width=0.1)
        
        # Should be in [0, 1]
        self.assertGreaterEqual(inv_sig.min(), 0.0)
        self.assertLessEqual(inv_sig.max(), 1.0)
        
        # Should be ~0.5 at center
        self.assertAlmostEqual(inv_sig[50], 0.5, places=1)
        
        # Should be monotonic decreasing
        self.assertTrue(np.all(np.diff(inv_sig) <= 0))

    def test_soft_range_behavior(self) -> None:
        x = np.linspace(0, 1, 100)
        soft = terrain_analysis._soft_range(x, min_val=0.3, max_val=0.7, width=0.05)
        
        # Should be in [0, 1]
        self.assertGreaterEqual(soft.min(), 0.0)
        self.assertLessEqual(soft.max(), 1.0)
        
        # Should be ~1 in the middle of range
        self.assertGreater(soft[50], 0.8)
        
        # Should be ~0 outside range
        self.assertLess(soft[10], 0.1)
        self.assertLess(soft[90], 0.1)


if __name__ == "__main__":
    unittest.main()

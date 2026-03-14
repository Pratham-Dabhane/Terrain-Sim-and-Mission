"""Unit tests for DEM analysis and calibration.

Focus:
- DEM loading from various formats
- Terrain statistics computation is stable and reasonable
- Calibration preserves determinism
- Non-destructive behavior (generation works without DEM)
- Parameter adjustments are bounded and sensible
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import dem_analysis
from pipeline.procedural_noise_utils import NoiseParams


class TestDEMLoading(unittest.TestCase):
    """Tests for DEM file loading."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic DEM
        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 1, 128)
        xv, yv = np.meshgrid(x, y)
        self.test_dem = 0.5 + 0.3 * np.sin(2 * np.pi * xv) * np.cos(2 * np.pi * yv)
        self.test_dem = np.clip(self.test_dem, 0.0, 1.0).astype(np.float32)

    def test_load_npy_format(self) -> None:
        # Save as NPY
        npy_path = os.path.join(self.temp_dir, 'test_dem.npy')
        np.save(npy_path, self.test_dem)
        
        # Load
        heightmap, metadata = dem_analysis.load_dem(npy_path)
        
        self.assertEqual(heightmap.shape, self.test_dem.shape)
        self.assertGreaterEqual(heightmap.min(), 0.0)
        self.assertLessEqual(heightmap.max(), 1.0)
        self.assertIn('source', metadata)

    def test_load_png_format(self) -> None:
        from PIL import Image
        
        # Save as PNG
        png_path = os.path.join(self.temp_dir, 'test_dem.png')
        img_data = (self.test_dem * 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='L')
        img.save(png_path)
        
        # Load
        heightmap, metadata = dem_analysis.load_dem(png_path)
        
        self.assertEqual(heightmap.shape, self.test_dem.shape)
        self.assertGreaterEqual(heightmap.min(), 0.0)
        self.assertLessEqual(heightmap.max(), 1.0)

    def test_load_nonexistent_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            dem_analysis.load_dem('nonexistent_dem.npy')


class TestTerrainStatistics(unittest.TestCase):
    """Tests for terrain statistics computation."""

    def setUp(self) -> None:
        # Create diverse synthetic terrain
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        xv, yv = np.meshgrid(x, y)
        
        # Multiple features: gradient + sine waves
        self.terrain = (
            xv * 0.4 +  # Gradient
            0.3 * np.sin(8 * np.pi * xv) * np.cos(8 * np.pi * yv) +  # High-freq
            0.2 * np.sin(2 * np.pi * xv)  # Low-freq
        )
        self.terrain = np.clip(self.terrain, 0.0, 1.0).astype(np.float32)

    def test_statistics_computation(self) -> None:
        stats = dem_analysis.compute_terrain_statistics(self.terrain)
        
        # Check all fields exist
        self.assertIsNotNone(stats.elevation_mean)
        self.assertIsNotNone(stats.elevation_std)
        self.assertIsNotNone(stats.slope_mean)
        self.assertIsNotNone(stats.slope_std)
        self.assertIsNotNone(stats.drainage_density)
        self.assertIsNotNone(stats.ridge_spacing)
        self.assertIsNotNone(stats.roughness)

    def test_elevation_stats_in_range(self) -> None:
        stats = dem_analysis.compute_terrain_statistics(self.terrain)
        
        # Mean should be in [0, 1]
        self.assertGreaterEqual(stats.elevation_mean, 0.0)
        self.assertLessEqual(stats.elevation_mean, 1.0)
        
        # Std should be reasonable
        self.assertGreaterEqual(stats.elevation_std, 0.0)
        self.assertLessEqual(stats.elevation_std, 0.5)

    def test_slope_stats_reasonable(self) -> None:
        stats = dem_analysis.compute_terrain_statistics(self.terrain)
        
        # Slope should be positive
        self.assertGreater(stats.slope_mean, 0.0)
        self.assertGreater(stats.slope_std, 0.0)

    def test_drainage_density_in_range(self) -> None:
        stats = dem_analysis.compute_terrain_statistics(self.terrain)
        
        # Drainage density is a fraction
        self.assertGreaterEqual(stats.drainage_density, 0.0)
        self.assertLessEqual(stats.drainage_density, 1.0)

    def test_ridge_spacing_positive(self) -> None:
        stats = dem_analysis.compute_terrain_statistics(self.terrain)
        
        self.assertGreater(stats.ridge_spacing, 0.0)

    def test_flat_terrain_statistics(self) -> None:
        flat = np.ones((64, 64), dtype=np.float32) * 0.5
        stats = dem_analysis.compute_terrain_statistics(flat)
        
        # Flat terrain should have near-zero slope and std
        self.assertLess(stats.slope_mean, 0.01)
        self.assertLess(stats.elevation_std, 0.01)


class TestStatisticalComparison(unittest.TestCase):
    """Tests for statistical comparison functions."""

    def test_compare_identical_terrains(self) -> None:
        # Create terrain
        terrain = np.random.rand(128, 128).astype(np.float32)
        
        # Compute stats for same terrain
        stats1 = dem_analysis.compute_terrain_statistics(terrain)
        stats2 = dem_analysis.compute_terrain_statistics(terrain)
        
        # Compare
        metrics = dem_analysis.compare_terrain_statistics(stats1, stats2)
        
        # Errors should be near zero
        self.assertLess(metrics['elevation_mean_error'], 1e-6)
        self.assertLess(metrics['slope_mean_error'], 1e-6)
        self.assertLess(metrics['overall_error'], 1e-3)

    def test_compare_different_terrains(self) -> None:
        terrain1 = np.random.rand(128, 128).astype(np.float32) * 0.3  # Low elevation
        terrain2 = np.random.rand(128, 128).astype(np.float32) * 0.9 + 0.1  # High elevation
        
        stats1 = dem_analysis.compute_terrain_statistics(terrain1)
        stats2 = dem_analysis.compute_terrain_statistics(terrain2)
        
        metrics = dem_analysis.compare_terrain_statistics(stats1, stats2)
        
        # Should detect elevation difference
        self.assertGreater(metrics['elevation_mean_error'], 0.1)
        self.assertGreater(metrics['overall_error'], 0.0)


class TestCalibration(unittest.TestCase):
    """Tests for DEM calibration."""

    def test_calibration_disabled_returns_original(self) -> None:
        # Ensure calibration is disabled
        original_flag = dem_analysis.ENABLE_DEM_CALIBRATION
        dem_analysis.ENABLE_DEM_CALIBRATION = False
        
        try:
            params = NoiseParams(scale=100.0, octaves=6)
            
            # Create dummy DEM file
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                dem_path = f.name
                np.save(f, np.random.rand(128, 128).astype(np.float32))
            
            try:
                result_params, log = dem_analysis.calibrate_to_dem(dem_path, params)
                
                # Should return original params when disabled
                self.assertEqual(result_params.scale, params.scale)
                self.assertEqual(result_params.octaves, params.octaves)
                self.assertEqual(log['calibration'], 'disabled')
            finally:
                os.unlink(dem_path)
        
        finally:
            dem_analysis.ENABLE_DEM_CALIBRATION = original_flag

    def test_calibration_preserves_seed(self) -> None:
        # Calibration should not change seed (preserves determinism)
        original_flag = dem_analysis.ENABLE_DEM_CALIBRATION
        dem_analysis.ENABLE_DEM_CALIBRATION = True
        
        try:
            params = NoiseParams(scale=100.0, octaves=6, seed=42)
            
            # Create synthetic DEM
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                dem_path = f.name
                dem = np.random.rand(128, 128).astype(np.float32)
                np.save(f, dem)
            
            try:
                config = dem_analysis.CalibrationConfig(iterations=2)
                result_params, log = dem_analysis.calibrate_to_dem(
                    dem_path, params, calibration_config=config
                )
                
                # Seed must be preserved
                self.assertEqual(result_params.seed, params.seed)
            finally:
                os.unlink(dem_path)
        
        finally:
            dem_analysis.ENABLE_DEM_CALIBRATION = original_flag

    def test_calibrated_parameters_in_bounds(self) -> None:
        original_flag = dem_analysis.ENABLE_DEM_CALIBRATION
        dem_analysis.ENABLE_DEM_CALIBRATION = True
        
        try:
            params = NoiseParams(scale=100.0, octaves=6, persistence=0.5)
            
            # Create synthetic DEM
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                dem_path = f.name
                dem = np.random.rand(64, 64).astype(np.float32)  # Small for speed
                np.save(f, dem)
            
            try:
                config = dem_analysis.CalibrationConfig(iterations=2)
                result_params, log = dem_analysis.calibrate_to_dem(
                    dem_path, params, calibration_config=config
                )
                
                # Parameters should be within bounds
                self.assertGreaterEqual(result_params.scale, config.scale_min)
                self.assertLessEqual(result_params.scale, config.scale_max)
                self.assertGreaterEqual(result_params.octaves, config.octaves_min)
                self.assertLessEqual(result_params.octaves, config.octaves_max)
                self.assertGreaterEqual(result_params.persistence, config.persistence_min)
                self.assertLessEqual(result_params.persistence, config.persistence_max)
            finally:
                os.unlink(dem_path)
        
        finally:
            dem_analysis.ENABLE_DEM_CALIBRATION = original_flag


class TestDEMExport(unittest.TestCase):
    """Tests for DEM export functionality."""

    def test_export_as_png(self) -> None:
        terrain = np.random.rand(64, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'exported.png')
            dem_analysis.export_as_dem(terrain, output_path, format='png')
            
            # Verify file exists
            self.assertTrue(os.path.exists(output_path))
            
            # Load and verify (close immediately to avoid Windows file lock)
            from PIL import Image
            with Image.open(output_path) as img:
                self.assertEqual(img.size, (64, 64))

    def test_export_as_npy(self) -> None:
        terrain = np.random.rand(64, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'exported.npy')
            dem_analysis.export_as_dem(terrain, output_path, format='npy')
            
            # Load and verify
            loaded = np.load(output_path)
            np.testing.assert_allclose(loaded, terrain)


if __name__ == "__main__":
    unittest.main()

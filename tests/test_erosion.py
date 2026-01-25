"""Unit tests for the erosion post-processing stage.

Focus:
- Erosion does not introduce NaNs or infinities.
- Height range remains within [0, 1] after erosion.
- Erosion is disabled cleanly when feature flags are off.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import erosion


class TestErosionStability(unittest.TestCase):
    """Stability and range checks for erosion."""

    def setUp(self) -> None:
        # Simple synthetic heightfield with variation
        x = np.linspace(0, 1, 64, dtype=np.float32)
        y = np.linspace(0, 1, 64, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        self.base = 0.5 + 0.25 * np.sin(2 * np.pi * xv) * np.cos(2 * np.pi * yv)
        self.base = np.clip(self.base, 0.0, 1.0).astype(np.float32)

    def test_no_nan_or_inf(self) -> None:
        params = erosion.ErosionParams(hydraulic_iterations=10, thermal_iterations=5)

        # Ensure erosion flags are enabled for this test
        erosion.ENABLE_HYDRAULIC_EROSION = True
        erosion.ENABLE_THERMAL_EROSION = True

        eroded = erosion.apply_erosion(self.base, params=params)

        self.assertFalse(np.isnan(eroded).any(), "Erosion produced NaN values")
        self.assertFalse(np.isinf(eroded).any(), "Erosion produced Inf values")

    def test_height_range_clamped(self) -> None:
        params = erosion.ErosionParams(hydraulic_iterations=15, thermal_iterations=10)

        erosion.ENABLE_HYDRAULIC_EROSION = True
        erosion.ENABLE_THERMAL_EROSION = True

        eroded = erosion.apply_erosion(self.base, params=params)

        self.assertGreaterEqual(eroded.min(), 0.0)
        self.assertLessEqual(eroded.max(), 1.0)


class TestErosionFlags(unittest.TestCase):
    """Tests for erosion feature flag behaviour."""

    def setUp(self) -> None:
        self.height = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)

    def test_erosion_disabled_returns_same_height(self) -> None:
        # Turn off both erosion stages
        erosion.ENABLE_HYDRAULIC_EROSION = False
        erosion.ENABLE_THERMAL_EROSION = False

        params = erosion.ErosionParams(hydraulic_iterations=20, thermal_iterations=20)

        result = erosion.apply_erosion(self.height, params=params)

        np.testing.assert_allclose(result, self.height.astype(np.float32))

    def test_erosion_enabled_changes_height(self) -> None:
        # Enable at least one erosion stage
        erosion.ENABLE_HYDRAULIC_EROSION = True
        erosion.ENABLE_THERMAL_EROSION = False

        params = erosion.ErosionParams(hydraulic_iterations=10, thermal_iterations=0)

        result = erosion.apply_erosion(self.height, params=params)

        # Expect some change when erosion is enabled on a non-uniform field
        self.assertGreater(np.abs(result - self.height).sum(), 0.0)


if __name__ == "__main__":
    unittest.main()

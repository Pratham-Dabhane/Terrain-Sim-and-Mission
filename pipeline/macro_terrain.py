"""Macro-scale terrain generation utilities.

This module adds a "MacroHeightField" stage that creates large-scale
continental structure and broad elevation zones before fine noise is applied.

Design goals:
- Use only low-frequency noise and simple ridge shaping (no erosion here).
- Provide domain-warp fields (dx, dy) that are applied before high-frequency
  noise sampling in the existing pipeline.
- Stay compatible with the existing heightmap pipeline and avoid new
  dependencies.
"""

from typing import Tuple, Optional
import os
import logging

import numpy as np
import noise  # Perlin noise library

logger = logging.getLogger(__name__)


def _normalize(height: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1].

    This helper is used for both the macro heightfield and debug visualizations.
    """
    h_min = float(height.min())
    h_max = float(height.max())
    if h_max > h_min:
        return (height - h_min) / (h_max - h_min)
    return np.ones_like(height, dtype=np.float32) * 0.5


def generate_macro_terrain(
    shape: Tuple[int, int],
    params: Optional[object] = None,
    debug_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a low-frequency macro heightfield and domain-warp fields.

    Assumptions (documented for clarity):
    - We model "continental" structure using very low-frequency Perlin noise.
    - Ridge shaping (1 - |noise|) is used to create broad mountain ranges.
    - A secondary low-frequency noise field is used to hint at basins.
    - Domain warping is expressed as (dx, dy) offsets in index space; the
      high-frequency noise functions convert these into noise-space coords.

    Args:
        shape: (height, width) of the terrain grid.
        params: Noise parameter object; only a subset is used here:
            - scale: used as a reference scale for low-frequency noise.
            - seed: controls reproducibility of macro patterns.
        debug_dir: Optional directory where macro debug visualizations
            (macro heightfield and warp magnitude) will be written.

    Returns:
        macro_height: Normalized macro heightfield in [0, 1].
        warp_dx: X-offsets for domain warping (same shape as macro_height).
        warp_dy: Y-offsets for domain warping.
    """
    height, width = shape

    # Pull a few knobs from params if present; fall back to conservative defaults
    base_scale = getattr(params, "scale", 100.0)
    base_seed = getattr(params, "seed", None)
    base_seed = base_seed if base_seed is not None else 0

    logger.debug(
        "Generating macro terrain: shape=%s, base_scale=%.1f, seed=%d",
        shape,
        base_scale,
        base_seed,
    )

    macro = np.zeros((height, width), dtype=np.float32)
    basins = np.zeros_like(macro)

    # Very low-frequency noise for continental plates.
    # We intentionally use a much larger scale than the micro noise.
    plate_scale = base_scale * 6.0  # assumption: macro > micro by a factor ~6
    ridge_scale = base_scale * 4.0

    plate_freq = 1.0 / max(plate_scale, 1.0)
    ridge_freq = 1.0 / max(ridge_scale, 1.0)

    for i in range(height):
        for j in range(width):
            x_plate = j * plate_freq
            y_plate = i * plate_freq

            # Broad elevation zones ("continents" vs. oceans).
            plate_val = noise.pnoise2(
                x_plate,
                y_plate,
                octaves=2,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=2048,
                repeaty=2048,
                base=base_seed,
            )

            x_ridge = j * ridge_freq
            y_ridge = i * ridge_freq

            # Ridge shaping to emphasize long mountain chains.
            ridge_raw = noise.pnoise2(
                x_ridge,
                y_ridge,
                octaves=3,
                persistence=0.6,
                lacunarity=2.0,
                repeatx=2048,
                repeaty=2048,
                base=base_seed + 37,
            )
            ridge = 1.0 - abs(ridge_raw)  # convert valleys in noise to peaks
            ridge = ridge ** 2  # sharpen ranges

            macro[i, j] = plate_val
            basins[i, j] = ridge

    # Combine plates and ridges into a macro elevation field.
    macro_combined = 0.55 * macro + 0.45 * basins
    macro_height = _normalize(macro_combined)

    # Domain warp fields: low-frequency offsets that will be applied before
    # high-frequency noise sampling. We keep the magnitude modest so that
    # features are bent and curved but not completely folded over.
    warp_dx = np.zeros_like(macro_height, dtype=np.float32)
    warp_dy = np.zeros_like(macro_height, dtype=np.float32)

    warp_scale = base_scale * 3.0
    warp_freq = 1.0 / max(warp_scale, 1.0)
    warp_seed = base_seed + 1234

    # Maximum offset in index space; this is an empirically chosen value
    # that bends features without creating extreme distortions.
    max_offset = min(height, width) * 0.08

    for i in range(height):
        for j in range(width):
            x_warp = j * warp_freq
            y_warp = i * warp_freq

            wx = noise.pnoise2(
                x_warp,
                y_warp,
                octaves=2,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=2048,
                repeaty=2048,
                base=warp_seed,
            )
            wy = noise.pnoise2(
                x_warp + 100.0,
                y_warp + 100.0,
                octaves=2,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=2048,
                repeaty=2048,
                base=warp_seed + 17,
            )

            # Map noise in [-1, 1] to symmetric offsets in [-max_offset, max_offset].
            warp_dx[i, j] = float(wx) * max_offset
            warp_dy[i, j] = float(wy) * max_offset

    # Optional debug visualizations: macro heightfield and warp magnitude.
    if debug_dir is not None:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            import matplotlib.pyplot as plt  # local import to avoid hard dependency at import time

            macro_img = macro_height
            warp_mag = np.sqrt(warp_dx ** 2 + warp_dy ** 2)
            warp_img = _normalize(warp_mag)

            macro_path = os.path.join(debug_dir, "macro_heightfield.png")
            warp_path = os.path.join(debug_dir, "domain_warp_magnitude.png")

            plt.imsave(macro_path, macro_img, cmap="terrain")
            plt.imsave(warp_path, warp_img, cmap="magma")

            logger.info(
                "Saved macro debug images to %s (macro_heightfield.png, domain_warp_magnitude.png)",
                debug_dir,
            )
        except Exception as exc:  # pragma: no cover - best-effort debug output
            logger.warning("Failed to write macro debug images: %s", exc)

    logger.debug(
        "Macro terrain generated: range=[%.3f, %.3f]",
        float(macro_height.min()),
        float(macro_height.max()),
    )
    return macro_height.astype(np.float32), warp_dx, warp_dy

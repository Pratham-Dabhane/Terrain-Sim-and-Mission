"""Macro terrain comparison script.

Generates two heightmaps for visual comparison:
- Legacy pipeline (macro terrain disabled)
- Macro-to-micro pipeline (macro terrain enabled)

Heightmaps are saved as PNGs in Output/macro_debug/.

This is intentionally a simple script rather than a unit test so that it can
be run manually when tuning macro terrain parameters.
"""

import os

import numpy as np

from pipeline import procedural_noise_utils as pnu
from pipeline.procedural_noise_utils import NoiseParams


def main() -> None:
    output_dir = os.path.join("Output", "macro_debug")
    os.makedirs(output_dir, exist_ok=True)

    shape = (256, 256)
    params = NoiseParams(seed=42)

    # Legacy pipeline (macro terrain off)
    pnu.ENABLE_MACRO_TERRAIN = False
    height_legacy = pnu.generate_procedural_heightmap(shape, params)

    # Macro-to-micro pipeline (macro terrain on)
    pnu.ENABLE_MACRO_TERRAIN = True
    height_macro = pnu.generate_procedural_heightmap(
        shape,
        params,
        debug_dir=output_dir,
    )

    # Save both heightmaps as images for side-by-side inspection.
    try:
        import matplotlib.pyplot as plt

        legacy_path = os.path.join(output_dir, "heightmap_legacy.png")
        macro_path = os.path.join(output_dir, "heightmap_macro.png")

        plt.imsave(legacy_path, height_legacy, cmap="terrain")
        plt.imsave(macro_path, height_macro, cmap="terrain")

        print(f"Saved comparison images to: {output_dir}")
        print(" -", legacy_path)
        print(" -", macro_path)
    except Exception as exc:  # pragma: no cover
        # Fallback: at least persist the raw arrays.
        np.save(os.path.join(output_dir, "heightmap_legacy.npy"), height_legacy)
        np.save(os.path.join(output_dir, "heightmap_macro.npy"), height_macro)
        print("Failed to write PNGs, saved .npy arrays instead:", exc)


if __name__ == "__main__":
    main()

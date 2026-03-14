"""Erosion post-processing stage for terrain heightfields.

Phase 2: add physically-motivated erosion on top of the existing
macro-to-micro noise pipeline.

Design constraints:
- Implemented as a separate post-process: apply_erosion(heightfield, params).
- Guarded by feature flags so it can be disabled cheaply.
- Multiple iterations over the heightfield (both hydraulic and thermal).
- Deterministic given the same input heightfield and parameters.
- No new external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import logging

import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Feature flags (can be toggled by callers)
# -----------------------------------------------------------------------------

ENABLE_HYDRAULIC_EROSION: bool = True
ENABLE_THERMAL_EROSION: bool = True


# -----------------------------------------------------------------------------
# Parameterisation
# -----------------------------------------------------------------------------

@dataclass
class ErosionParams:
    """Configurable parameters for erosion.

    Iteration counts are parameters (not hard-coded in the algorithm) so that
    callers can adjust quality/performance trade-offs.
    """

    # Number of global hydraulic / thermal iterations
    hydraulic_iterations: int = 35
    thermal_iterations: int = 20

    # Hydraulic erosion parameters
    rainfall: float = 0.01          # Water added per cell per hydraulic step
    evaporation: float = 0.02       # Fraction of water removed each step
    sediment_capacity: float = 0.08 # Controls how much sediment water can carry
    erosion_rate: float = 0.3       # Max bedrock erosion per step
    deposition_rate: float = 0.3    # Max sediment deposition per step
    flow_rate: float = 0.5          # Fraction of water allowed to move per step

    # Thermal (talus) erosion parameters
    talus_angle: float = 0.02       # Slope threshold for material creep
    thermal_rate: float = 0.3       # Fraction of excess slope moved per step

    # Optional seed to keep room for future stochastic extensions; currently
    # unused so the algorithm is entirely deterministic given the inputs.
    seed: Optional[int] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] for debug visualisation.

    This does not change the erosion result; it is only used when writing
    debug PNGs for inspection.
    """
    a_min = float(arr.min())
    a_max = float(arr.max())
    if a_max > a_min:
        return (arr - a_min) / (a_max - a_min)
    return np.zeros_like(arr, dtype=np.float32)


def _clip_height(height: np.ndarray) -> None:
    """Clip heightfield in-place to a conservative range [0, 1].

    The base pipeline already normalises heightmaps to [0, 1]. Erosion should
    keep them within sensible bounds and avoid numeric explosions.
    """
    np.clip(height, 0.0, 1.0, out=height)


# -----------------------------------------------------------------------------
# Hydraulic erosion
# -----------------------------------------------------------------------------

def _run_hydraulic_erosion(
    height: np.ndarray,
    params: ErosionParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a simplified hydraulic erosion model.

    This implementation is intentionally conservative and grid-based:
    - Uniform rainfall
    - 4-neighbour water flow routing
    - Sediment capacity based on local slope and water amount
    - Erosion / deposition limited per step to avoid instability

    Returns:
        (new_height, water_field, sediment_field)
    """
    h = height.astype(np.float32).copy()
    water = np.zeros_like(h, dtype=np.float32)
    sediment = np.zeros_like(h, dtype=np.float32)

    eps = 1e-6

    for it in range(params.hydraulic_iterations):
        # 1) Rainfall: add a small, uniform amount of water everywhere.
        water += params.rainfall

        # 2) Compute water surface (height + water depth).
        surface = h + water

        # 3) Approximate flow to 4 neighbours (N, S, E, W).
        sN = np.roll(surface, -1, axis=0)
        sS = np.roll(surface, 1, axis=0)
        sE = np.roll(surface, -1, axis=1)
        sW = np.roll(surface, 1, axis=1)

        dN = surface - sN
        dS = surface - sS
        dE = surface - sE
        dW = surface - sW

        # Positive slopes only: water flows downhill.
        pN = np.maximum(dN, 0.0)
        pS = np.maximum(dS, 0.0)
        pE = np.maximum(dE, 0.0)
        pW = np.maximum(dW, 0.0)

        sum_p = pN + pS + pE + pW + eps

        # Fraction of water that can leave the cell this step.
        max_out = water * params.flow_rate

        outN = max_out * (pN / sum_p)
        outS = max_out * (pS / sum_p)
        outE = max_out * (pE / sum_p)
        outW = max_out * (pW / sum_p)

        # Prevent wrap-around artefacts on the domain boundaries by
        # cancelling flow that would cross the outer edges.
        outN[-1, :] = 0.0
        outS[0, :] = 0.0
        outE[:, -1] = 0.0
        outW[:, 0] = 0.0

        # Inflow is just neighbours' outflow in the opposite direction.
        inN = np.roll(outS, -1, axis=0)
        inS = np.roll(outN, 1, axis=0)
        inE = np.roll(outW, -1, axis=1)
        inW = np.roll(outE, 1, axis=1)

        total_out = outN + outS + outE + outW
        total_in = inN + inS + inE + inW

        water = water + total_in - total_out
        water = np.maximum(water, 0.0)

        # 4) Sediment capacity based on slope magnitude and water amount.
        slope_mag = (
            np.abs(dN) + np.abs(dS) + np.abs(dE) + np.abs(dW)
        ) * 0.25

        capacity = params.sediment_capacity * slope_mag * (water + eps)

        # 5) Erode or deposit bed material depending on current load.
        # Erosion: if we can carry more sediment than we currently have.
        erosion_potential = capacity - sediment
        erode = np.clip(erosion_potential, 0.0, params.erosion_rate)

        h -= erode
        sediment += erode

        # Deposition: if we are above capacity.
        deposition_potential = sediment - capacity
        deposit = np.clip(deposition_potential, 0.0, params.deposition_rate)

        h += deposit
        sediment -= deposit

        # Simple evaporation to avoid unbounded water accumulation.
        water *= max(0.0, 1.0 - params.evaporation)

        _clip_height(h)

    return h, water, sediment


# -----------------------------------------------------------------------------
# Thermal erosion
# -----------------------------------------------------------------------------

def _run_thermal_erosion(height: np.ndarray, params: ErosionParams) -> np.ndarray:
    """Apply slope-based (talus) erosion.

    Material moves from cells whose slope to a neighbour exceeds the talus
    angle down to that neighbour, over multiple iterations.
    """
    h = height.astype(np.float32).copy()

    for _ in range(params.thermal_iterations):
        # Neighbour heights
        hN = np.roll(h, -1, axis=0)
        hS = np.roll(h, 1, axis=0)
        hE = np.roll(h, -1, axis=1)
        hW = np.roll(h, 1, axis=1)

        dN = h - hN
        dS = h - hS
        dE = h - hE
        dW = h - hW

        # Amount above talus angle to move towards each neighbour.
        moveN = np.maximum(dN - params.talus_angle, 0.0)
        moveS = np.maximum(dS - params.talus_angle, 0.0)
        moveE = np.maximum(dE - params.talus_angle, 0.0)
        moveW = np.maximum(dW - params.talus_angle, 0.0)

        # Scale by thermal rate so we do not over-flatten in a single step.
        moveN *= params.thermal_rate
        moveS *= params.thermal_rate
        moveE *= params.thermal_rate
        moveW *= params.thermal_rate

        # Cancel movement across domain boundaries.
        moveN[-1, :] = 0.0
        moveS[0, :] = 0.0
        moveE[:, -1] = 0.0
        moveW[:, 0] = 0.0

        total_out = moveN + moveS + moveE + moveW

        # Incoming material from neighbours.
        inN = np.roll(moveS, -1, axis=0)
        inS = np.roll(moveN, 1, axis=0)
        inE = np.roll(moveW, -1, axis=1)
        inW = np.roll(moveE, 1, axis=1)

        total_in = inN + inS + inE + inW

        h = h - total_out + total_in

        _clip_height(h)

    return h


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def apply_erosion(
    heightfield: np.ndarray,
    params: Optional[ErosionParams] = None,
    debug_dir: Optional[str] = None,
) -> np.ndarray:
    """Apply hydraulic and/or thermal erosion to a heightfield.

    Erosion is fully optional and controlled via feature flags. When both
    ENABLE_HYDRAULIC_EROSION and ENABLE_THERMAL_EROSION are False, the input
    heightfield is returned unchanged.

    Args:
        heightfield: Base terrain in [0, 1] produced by the noise pipeline.
        params: Erosion parameters. If None, a default ErosionParams is used.
        debug_dir: Optional directory for debug maps (water, sediment, delta).

    Returns:
        Eroded heightfield (same shape as input), clipped to [0, 1].
    """
    if params is None:
        params = ErosionParams()

    # Short-circuit when erosion is disabled via flags.
    if not (ENABLE_HYDRAULIC_EROSION or ENABLE_THERMAL_EROSION):
        logger.debug("Erosion disabled by feature flags; returning input heightfield unchanged.")
        return heightfield.astype(np.float32).copy()

    # Work on a copy so callers can still inspect the original field.
    original = heightfield.astype(np.float32).copy()
    h = original.copy()

    water = np.zeros_like(h, dtype=np.float32)
    sediment = np.zeros_like(h, dtype=np.float32)

    if ENABLE_HYDRAULIC_EROSION and params.hydraulic_iterations > 0:
        logger.info(
            "Running hydraulic erosion: iterations=%d, rainfall=%.4f",
            params.hydraulic_iterations,
            params.rainfall,
        )
        h, water, sediment = _run_hydraulic_erosion(h, params)

    if ENABLE_THERMAL_EROSION and params.thermal_iterations > 0:
        logger.info(
            "Running thermal erosion: iterations=%d, talus_angle=%.4f",
            params.thermal_iterations,
            params.talus_angle,
        )
        h = _run_thermal_erosion(h, params)

    _clip_height(h)

    # Optional debug outputs for visibility into the erosion process.
    if debug_dir is not None:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            import matplotlib.pyplot as plt  # Local import to avoid hard import-time dependency

            # Water flow map: final water depth after hydraulic erosion.
            if ENABLE_HYDRAULIC_EROSION:
                water_img = _normalize(water)
                plt.imsave(os.path.join(debug_dir, "erosion_water.png"), water_img, cmap="Blues")

                sed_img = _normalize(sediment)
                plt.imsave(os.path.join(debug_dir, "erosion_sediment.png"), sed_img, cmap="cividis")

            # Erosion delta map: where the surface moved most.
            delta = original - h
            delta_img = _normalize(delta)
            plt.imsave(os.path.join(debug_dir, "erosion_delta.png"), delta_img, cmap="PuOr")

            logger.info("Saved erosion debug maps to %s", debug_dir)
        except Exception as exc:  # pragma: no cover - best-effort debug visualisation
            logger.warning("Failed to write erosion debug maps: %s", exc)

    # Safety: ensure no NaNs or infinities are returned.
    if not np.isfinite(h).all():
        logger.error("Erosion produced non-finite values; reverting to original heightfield.")
        return original

    return h.astype(np.float32)

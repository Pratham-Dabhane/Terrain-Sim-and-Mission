#!/usr/bin/env python3
"""
Generative AI Terrain Prototype (Refactored & Annotated)

Pipeline:
  Text Prompt → GAN Heightmap → Feature Modifications → Refinement
  → Colored Terrain → Visualization (2D + 3D)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Optional deps
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers.utils import load_image
    from transformers import logging
    import torch
    DIFFUSION_AVAILABLE = True
    logging.set_verbosity_error()
except ImportError:
    DIFFUSION_AVAILABLE = False


# -----------------------------
# 1. GAN-like heightmap generator
# -----------------------------
class SimpleStyleGAN2Generator(nn.Module):
    """Tiny fully connected GAN-like generator producing heightmaps."""
    def __init__(self, latent_dim=512, output_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_size * output_size),
            nn.Sigmoid()  # ensures values ∈ [0,1]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, self.output_size, self.output_size).cpu().detach().numpy()


# -----------------------------
# 2. Terrain generation
# -----------------------------
def prompt_to_heightmap_gan(prompt, size=128):
    """Generate a base heightmap from a prompt using GAN or Perlin fallback."""
    if TORCH_AVAILABLE:
        gen = SimpleStyleGAN2Generator(output_size=size)
        z = torch.randn(1, gen.latent_dim)
        heightmap = gen(z)[0]
    else:
        # Fallback: random smooth noise
        rng = np.random.default_rng()
        heightmap = rng.random((size, size))
        try:
            from scipy.ndimage import gaussian_filter
            heightmap = gaussian_filter(heightmap, sigma=3)
        except ImportError:
            # Simple smoothing
            heightmap = (heightmap[:-2, :-2] + heightmap[2:, 2:]) / 2

    # Apply prompt-driven modifications
    return _apply_prompt_modifications(heightmap, prompt)


# -----------------------------
# 3. Prompt-driven modifications
# -----------------------------
def _apply_prompt_modifications(heightmap, prompt):
    """Add terrain features depending on keywords in prompt."""
    p = prompt.lower()
    if "mountain" in p or "peak" in p:
        heightmap = _add_mountains(heightmap)
    if "valley" in p or "canyon" in p:
        heightmap = _add_valleys(heightmap)
    if "river" in p or "water" in p:
        heightmap = _add_rivers(heightmap)
    if "forest" in p or "tree" in p:
        heightmap = _add_forest_roughness(heightmap)
    if "desert" in p or "sand" in p:
        heightmap = _add_desert_features(heightmap)

    return np.clip(heightmap, 0, 1)


def _add_mountains(hm):
    """Raise several random bumps."""
    size = hm.shape[0]
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    for _ in range(3):
        cx, cy = np.random.randint(0, size, 2)
        r = np.random.randint(size // 8, size // 4)
        mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
        bump = np.maximum(0, 1 - ((X - cx) ** 2 + (Y - cy) ** 2) / (r ** 2))
        hm += 0.4 * bump * mask
    return hm


def _add_valleys(hm):
    """Carve linear depressions."""
    size = hm.shape[0]
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    for _ in range(2):
        x0, y0 = np.random.randint(0, size, 2)
        x1, y1 = np.random.randint(0, size, 2)
        # Distance to line
        d = np.abs((y1 - y0) * X - (x1 - x0) * Y + x1 * y0 - y1 * x0) / (
            np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2) + 1e-8
        )
        mask = d < size // 8
        hm[mask] -= 0.3 * (1 - d[mask] / (size // 8))
    return hm


def _add_rivers(hm):
    """Add thin meandering rivers."""
    size = hm.shape[0]
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    for _ in range(2):
        x0, y0 = np.random.randint(0, size, 2)
        x1, y1 = np.random.randint(0, size, 2)
        d = np.abs((y1 - y0) * X - (x1 - x0) * Y + x1 * y0 - y1 * x0) / (
            np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2) + 1e-8
        )
        mask = d < size // 16
        hm[mask] -= 0.5 * (1 - d[mask] / (size // 16))
    return hm


def _add_forest_roughness(hm):
    """Add small-scale noise for forest roughness."""
    hm += 0.05 * np.random.rand(*hm.shape)
    return hm


def _add_desert_features(hm):
    """Add low dunes."""
    size = hm.shape[0]
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    for _ in range(5):
        cx, cy = np.random.randint(0, size, 2)
        r = np.random.randint(size // 10, size // 6)
        mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
        dune = np.maximum(0, 1 - ((X - cx) ** 2 + (Y - cy) ** 2) / (r ** 2))
        hm += 0.2 * dune * mask
    return hm


# -----------------------------
# 4. Refinement & coloring
# -----------------------------
def _basic_enhancement(hm, prompt):
    """Enhance heightmap with contrast + noise + colors."""
    hm = np.clip(hm ** 0.8 + 0.05 * np.random.rand(*hm.shape), 0, 1)
    return _apply_terrain_colors(hm)


def _apply_terrain_colors(hm):
    """Vectorized mapping of heightmap → RGB terrain image."""
    h, w = hm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Apply masks for each band
    rgb[hm < 0.2] = [51, 102, 204]     # water
    rgb[(hm >= 0.2) & (hm < 0.35)] = [230, 204, 153]  # sand
    rgb[(hm >= 0.35) & (hm < 0.5)] = [77, 153, 51]    # grass
    rgb[(hm >= 0.5) & (hm < 0.7)] = [51, 102, 26]     # forest
    rgb[(hm >= 0.7) & (hm < 0.85)] = [128, 128, 128]  # rock
    rgb[hm >= 0.85] = [230, 230, 230]  # snow

    return rgb


# -----------------------------
# 5. Visualization
# -----------------------------
def visualize_terrain(terrain, hm, prompt, save_path=None):
    """Show heightmap, final terrain, and 3D surface."""
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131)
    ax1.imshow(hm, cmap="terrain")
    ax1.set_title("Heightmap")

    ax2 = fig.add_subplot(132)
    ax2.imshow(terrain)
    ax2.set_title("Final Terrain")

    ax3 = fig.add_subplot(133, projection="3d")
    y, x = np.mgrid[0 : hm.shape[0], 0 : hm.shape[1]]
    ax3.plot_surface(x, y, hm, cmap="terrain")
    ax3.set_title("3D View")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# 6. Main
# -----------------------------
if __name__ == "__main__":
    prompt = input("Enter terrain description: ") or "mountainous terrain with rivers"
    hm = prompt_to_heightmap_gan(prompt, size=128)
    terrain = _basic_enhancement(hm, prompt)
    visualize_terrain(terrain, hm, prompt, save_path="terrain_output.png")

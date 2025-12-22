# Terrain Simulation and Mission Planning

End-to-end terrain generation and mission planning system that turns natural language prompts into 3D landscapes and computes cost-aware navigation paths over them.

The project combines procedural terrain synthesis, GPU-accelerated 3D rendering, and an A* planner operating on learned cost maps derived from elevation and slope.

---

## Features

- Text-to-terrain generation from natural language prompts using procedural noise and prompt-derived parameters
- GPU-accelerated 3D terrain rendering with PyVista/VTK (interactive and offline)
- Cost-aware mission planning via A* on terrain cost maps (elevation, slope, water)
- Automatic cost-map and path statistics (difficulty, elevation gain/loss, total cost)
- Image and mesh exports for further analysis (`.png`, `.vtk`)
- Comprehensive automated tests for parsing, terrain generation, cost maps, and pathfinding

---

## Tech Stack

- Python 3.11
- Deep learning / generation: PyTorch, TensorFlow, Hugging Face Transformers, Diffusers
- Geometry & visualization: NumPy, PyVista, VTK, Matplotlib
- Utilities & tooling: OpenCV, SciPy, Pillow, pytest/pytest-cov

---

## Getting Started

### Installation

```bash
# Clone the repository

cd "Terrain Sim and Mission"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements_stable.txt
```

### Basic Usage

```bash
# Generate terrain with interactive 3D viewer
python pipeline_demo.py --prompt "snowy mountain peaks" --interactive-3d

# Generate terrain and compute an optimal mission path
python pipeline_demo.py --prompt "desert dunes" --simulate-path

# Faster run without image remastering
python pipeline_demo.py --prompt "rolling hills" --no-remaster
```

---

## Architecture Overview

High-level flow from text prompt to mission plan:

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│ Text Prompt  │ --> │ Terrain Generator│ --> │ Cost Map      │
│ "mountains"  │     │ (Perlin Noise)   │     │ (Elevation +  │
└──────────────┘     └──────────────────┘     │  Slope)       │
                              │               └───────────────┘
                              ↓                        │
                     ┌──────────────────┐              ↓
                     │ 3D Mesh          │     ┌───────────────┐
                     │ (PyVista)        │     │ A* Pathfinder │
                     └──────────────────┘     │ (Optimal Path)│
                              │               └───────────────┘
                              ↓                        │
                     ┌──────────────────┐              ↓
                     │ Interactive      │     ┌───────────────┐
                     │ Viewer (GPU)     │     │ Path Overlay  │
                     └──────────────────┘     └───────────────┘
```

Component breakdown:

- Text Prompt
  - Free-form user description of the terrain (for example, "snowy mountain peaks" or "sandy desert dunes").
  - Parsed by the prompt parser in [pipeline/prompt_parser.py](pipeline/prompt_parser.py) into noise and style parameters.

- Terrain Generator (Perlin Noise)
  - Uses multi-octave Perlin-based fractional Brownian motion to synthesize a normalized heightmap.
  - Implemented in [pipeline/procedural_noise_utils.py](pipeline/procedural_noise_utils.py).

- Cost Map (Elevation + Slope)
  - Converts the heightmap into a traversal cost grid by combining elevation, slope, and water thresholds.
  - Implemented in [pipeline/cost_map.py](pipeline/cost_map.py).

- 3D Mesh (PyVista)
  - Lifts the heightmap into 3D, applies realistic terrain colouring, and builds a renderable mesh.
  - Implemented in [pipeline/advanced_terrain_renderer.py](pipeline/advanced_terrain_renderer.py) and [pipeline/mesh_visualize.py](pipeline/mesh_visualize.py).

- A* Pathfinder (Optimal Path)
  - Runs A* search over the cost map to find an efficient route between start and goal positions.
  - Computes path statistics such as length, elevation gain/loss, and total cost.
  - Implemented in [pipeline/planner.py](pipeline/planner.py).

- Interactive Viewer (GPU) and Path Overlay
  - Renders the 3D mesh with optional mission path overlay using a GPU-accelerated PyVista window.
  - Provides real-time interaction (orbit, zoom, pan) and static image exports.
  - Implemented in [pipeline/advanced_terrain_renderer.py](pipeline/advanced_terrain_renderer.py) with a Matplotlib fallback in [pipeline/matplotlib_viewer.py](pipeline/matplotlib_viewer.py).

---

## Output Artefacts

Each run writes to a timestamped directory under `Output/`:

```
Output/
└── session_<timestamp>/
    ├── heightmap.png                 # 2D elevation map
    ├── enhanced_terrain.png          # Colorized terrain (if remastered)
    ├── mesh.vtk                      # 3D mesh (VTK format)
    ├── visualization_3d.png          # Static 3D render
    ├── visualization_3d_PHOTOREALISTIC.png
    ├── cost_map.png                  # Traversal cost visualization
    ├── mission_path_overlay.png      # Optimal path on terrain
    └── metadata.json                 # Generation parameters and settings
```

---

## Testing

Run the automated test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific modules
python -m pytest tests/test_prompt_parameters.py -v
python -m pytest tests/test_cost_map.py -v
python -m pytest tests/test_planner.py -v

# With coverage
python -m pytest tests/ --cov=pipeline --cov-report=html
```

Tests cover prompt parsing, terrain generation, cost-map behaviour, and pathfinding edge cases.

---

## Acknowledgements

This project builds on the open-source ecosystems around PyVista, VTK, NumPy, Matplotlib, and the `noise` library for procedural generation.

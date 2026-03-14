# Terrain Simulation and Mission Planning

End-to-end terrain generation and mission planning system that turns natural language prompts into 3D landscapes and computes cost-aware navigation paths over them.

The project combines procedural terrain synthesis, GPU-accelerated 3D rendering, and an A* planner operating on learned cost maps derived from elevation and slope.

---

## Features

- **Multi-stage terrain pipeline**:
  - **Phase 1**: Macro-to-micro terrain generation with continental-scale structure and domain warping
  - **Phase 2**: Physically-motivated erosion (hydraulic + thermal) for realistic weathering
  - **Phase 3**: Terrain analysis with automatic biome classification (snow, rock, scree, forest, grassland, wetlands, water)
  - **Phase 4**: DEM-based calibration to ground procedural parameters in real-world terrain statistics
- Text-to-terrain generation from natural language prompts using procedural noise and prompt-derived parameters
- GPU-accelerated 3D terrain rendering with PyVista/VTK (interactive and offline)
- Cost-aware mission planning via A* on terrain cost maps (elevation, slope, water, biomes)
- Automatic cost-map and path statistics (difficulty, elevation gain/loss, total cost)
- Image and mesh exports for further analysis (`.png`, `.vtk`)
- Comprehensive automated tests (80 tests) for all pipeline stages

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

# Visualize biome classification
python biome_visualization_demo.py

# Calibrate to real-world DEM statistics
python dem_calibration_demo.py --dem path/to/real_dem.tif

# Faster run without image remastering
python pipeline_demo.py --prompt "rolling hills" --no-remaster
```

---

## Architecture Overview

High-level flow from text prompt to mission plan:
Pipeline │ --> │ Biome Analysis│
│ "mountains"  │     │ Macro + Erosion  │     │ 7 Biome Masks │
└──────────────┘     └──────────────────┘     └───────────────┘
                              │                        │
                              ↓                        ↓
                     ┌──────────────────┐     ┌───────────────┐
                     │ 3D Mesh          │     │ Cost Map      │
                     │ (PyVista)        │     │ (Multi-factor)│
                     └──────────────────┘     └───────────────┘
                              │                        │
                              ↓                        ↓
                     ┌──────────────────┐     ┌───────────────┐
                     │ Interactive      │     │ A* Pathfinder │
                     │ Viewer (GPU)     │     │ (Optimal Path)
                     ┌──────────────────┐              ↓
                     │ Interactive      │     ┌───────────────┐
                     │ Viewer (GPU)     │     │ Path Overlay  │
                     └──────────────────┘     └───────────────┘
```

Component breakdown:

- Text Prompt
  - Free-form user description of the terrain (for example, "snowy mountain peaks" or "sandy desert dunes").
  - Parsed by the prompt parser in [pipeline/prompt_parser.py](pipeline/prompt_parser.py) into noise and style parameters.

- Terrain Pipeline (Macro + Erosion)
  - **Phase 1**: Macro-scale heightfield with continental plates and domain warping ([pipeline/macro_terrain.py](pipeline/macro_terrain.py))
  - **Phase 2**: Hydraulic and thermal erosion for realistic weathering ([pipeline/erosion.py](pipeline/erosion.py))
  - **Core**: Multi-octave Perlin-based fractional Brownian motion ([pipeline/procedural_noise_utils.py](pipeline/procedural_noise_utils.py))
  
- Biome Analysis
  - **Phase 3**: Computes terrain attributes (slope, curvature, aspect, distance-to-water)
  - Generates smooth biome masks for snow, rock, scree, grassland, forest, wetlands, and water
  - Implemented in [pipeline/terrain_analysis.py](pipeline/terrain_analysis.py)

- DEM Calibration
  - **Phase 4**: Loads real-world DEMs (GeoTIFF, PNG, NPY) and computes terrain statistics
  - Calibrates procedural parameters to match DEM elevation, slope, drainage, and ridge spacing
  - Non-destructive and optional (procedural generation works independently)
  - Implemented in [pipeline/dem_analysis.py](pipeline/dem_analysis.py)

- Cost Map (Multi-factor)
  - Converts the heightmap into a traversal cost grid by combining elevation, slope, water, and biome-specific modifiers
  - Implemented in [pipeline/cost_map.py](pipeline/cost_map.py)

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
     # 2D elevation map
    ├── enhanced_terrain.png               # Colorized terrain (if remastered)
    ├── mesh.vtk                           # 3D mesh (VTK format)
    ├── visualization_3d.png               # Static 3D render
    ├── visualization_3d_PHOTOREALISTIC.png
    ├── cost_map.png                       # Traversal cost visualization
    ├── mission_path_overlay.png           # Optimal path on terrain
    ├── metadata.json                      # Generation parameters and settings
    └── debug/                             # Debug outputs (when enabled)
        ├── macro_heightfield.png          # Phase 1: Macro terrain structure
        ├── domain_warp_magnitude.png      # Phase 1: Domain warping
        ├── erosion_water.png              # Phase 2: Water accumulation
        ├── erosion_sediment.png           # Phase 2: Sediment distribution
        ├── erosion_delta.png              # Phase 2: Erosion/deposition
        ├── terrain_attributes.png         # Phase 3: Slope, curvature, etc.
        ├── biome_masks.png                # Phase 3: Individual biome masks
        └── dominant_biome.png             # Phase 3: Biome classification
# Run all tests (80 tests total)
python -m pytest tests/ -v

# Run specific modules
python -m pytest tests/test_prompt_parameters.py -v
python -m pytest tests/test_cost_map.py -v
python -m pytest tests/test_planner.py -v
python -m pytest tests/test_erosion.py -v
python -m pytest tests/test_terrain_analysis.py -v
python -m pytest tests/test_dem_analysis.py -v

# With coverage
python -m pytest tests/ --cov=pipeline --cov-report=html
```

Tests cover all pipeline stages: prompt parsing, macro terrain, erosion stability, biome classification, DEM calibration, cost-map behaviour, and pathfinding.

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

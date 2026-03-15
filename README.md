# Terrain Simulation and Mission Planning System

## 1. Project Title
Terrain Simulation and Mission Planning System

## 2. Short Description
This project is a research-oriented pipeline for generating synthetic terrain, calibrating it against real-world DEM statistics, and running mission planning analysis on the resulting landscapes.  
It combines procedural terrain synthesis, physically motivated erosion, terrain attribute and biome analysis, optional diffusion-based texture remastering, and A*-based route planning.  
The system is designed for reproducible experimentation, algorithm benchmarking, and scenario generation where realistic topography and traversal feasibility are both required.  
It supports both script-based workflows and service-style execution, with exportable artifacts for visualization and downstream analysis.

## 3. Key Features
- Procedural macro-to-micro terrain generation using fBM, domain warping, and terrain mixing controls
- Physical post-processing with hydraulic and thermal erosion
- Terrain attribute extraction and smooth biome mask generation
- DEM-based calibration against real-world terrain statistics
- Cost-map construction for mobility/traversability analysis
- A* mission path planning with path statistics and visual overlays
- High-quality 3D rendering and interactive terrain viewing
- Optional Stable Diffusion + ControlNet remastering for top-surface texturing
- FastAPI service layer for end-to-end generation workflows
- Automated tests for core analysis, calibration, erosion, cost-map, and planner behavior

## 4. Project Architecture
### High-Level Overview
The system is implemented as a modular terrain pipeline with optional branches for calibration, texture remastering, and service deployment.

```text
Text Prompt / Seed
      |
      v
Prompt Parsing + Parameterization
      |
      v
Procedural Terrain (Macro + Micro Noise)
      |
      v
Erosion (Hydraulic + Thermal)
      |
      +--------------------------+
      |                          |
      v                          v
Terrain Analysis + Biomes    DEM Statistics Calibration (optional)
      |                          |
      v                          |
Texture Mapping / Remaster  <----+
      |
      v
3D Rendering + Exports
      |
      v
Cost Map + A* Mission Planning
```

### Main Modules and Components
- Core generation
  - `pipeline/procedural_noise_utils.py`: orchestrates terrain generation
  - `pipeline/macro_terrain.py`: macro structure and domain warp field generation
  - `pipeline/erosion.py`: hydraulic and thermal erosion
- Prompt and parameter layer
  - `pipeline/prompt_parser.py`: keyword-to-parameter parsing
  - `pipeline/prompt_parameters.py`: deterministic prompt-to-parameter conversion
- Analysis and calibration
  - `pipeline/terrain_analysis.py`: slope/curvature/aspect/water-distance + biome masks
  - `pipeline/dem_analysis.py`: DEM loading, statistics, comparison, and iterative calibration
- Planning and simulation
  - `pipeline/cost_map.py`: terrain-to-traversal cost mapping
  - `pipeline/planner.py`: A* pathfinding, overlays, and path statistics
  - `pipeline/mission_simulator.py`: interactive mission simulation helpers
- Rendering and visualization
  - `pipeline/advanced_terrain_renderer.py`: static and interactive 3D rendering
  - `pipeline/terrain_texture_mapper.py`: physically inspired color/material mapping
  - `pipeline/mesh_visualize.py`, `pipeline/matplotlib_viewer.py`: additional visualization utilities
- Optional AI components
  - `pipeline/remaster_sd_controlnet.py`: SD + ControlNet remaster pipeline
  - `pipeline/ai_heightmap_generator.py`, `pipeline/models_awcgan.py`, `pipeline/clip_encoder.py`: AI-assisted generation paths
- Service layer
  - `serve.py`: FastAPI endpoints for asynchronous generation workflows

### Data Flow
1. Prompt and seed are parsed into deterministic terrain parameters.
2. Procedural generation creates a normalized heightmap (macro shape + micro detail).
3. Erosion modifies the terrain to improve physical plausibility.
4. Terrain analysis computes attributes and biome masks.
5. Optional DEM calibration adjusts procedural parameters to match target DEM statistics.
6. Texture generation (and optional remaster) creates render-ready surface appearance.
7. Renderer produces static/interactive 3D outputs and mesh artifacts.
8. Cost map and A* planner produce mission routes and traversal metrics.

### Interaction Between Generation, Calibration, and Mission Planning
- Generation provides the base heightfield.
- Calibration refines generation parameters to better match real DEM characteristics while preserving procedural diversity.
- Mission planning consumes the generated/calibrated terrain via cost maps to evaluate route feasibility and path efficiency.

## 5. Repository Structure
```text
.
|-- pipeline/                     # Core modules (generation, analysis, planning, rendering)
|   |-- procedural_noise_utils.py
|   |-- macro_terrain.py
|   |-- erosion.py
|   |-- terrain_analysis.py
|   |-- dem_analysis.py
|   |-- cost_map.py
|   |-- planner.py
|   |-- mission_simulator.py
|   |-- advanced_terrain_renderer.py
|   |-- terrain_texture_mapper.py
|   `-- ...
|-- tests/                        # Unit tests for core pipeline modules
|-- data/                         # Input DEM files and synthetic references
|-- Output/                       # Generated terrain, renders, calibration, and debug artifacts
|-- pipeline_demo.py              # Main end-to-end demo workflow
|-- generate_calibrated_terrain.py# DEM-calibrated terrain generation script
|-- dem_calibration_demo.py       # Standalone calibration demonstration
|-- biome_visualization_demo.py   # Terrain-analysis and biome visualization demo
|-- interactive_terrain_viewer.py # Interactive viewer for generated sessions
|-- serve.py                      # FastAPI service interface
|-- requirements.txt              # Full dependency set
`-- requirements_stable.txt       # Pinned/stable dependency set
```

## 6. Technologies Used
- Language
  - Python 3.10+
- Numerical and scientific computing
  - NumPy, SciPy, pandas
- Visualization and rendering
  - Matplotlib, PyVista, VTK, Pillow, OpenCV
- Terrain and image processing
  - noise (Perlin/fBM), scikit-image, imageio
- AI/ML stack (optional paths)
  - PyTorch, torchvision, Transformers, Diffusers, Accelerate, TensorFlow
- Service and API
  - FastAPI, Uvicorn, Pydantic
- Testing and quality
  - pytest, pytest-cov

## 7. How It Works
### 1) Terrain generation
- Prompt parsing maps textual cues to procedural parameters.
- Macro terrain stage creates continental-scale structure and warp fields.
- Micro detail is added with fBM and feature mixing (mountain/valley/river terms).

### 2) DEM loading
- DEM files are loaded from TIFF/PNG/JPG/NPY/NPZ formats.
- Input data is normalized and metadata is retained when available.

### 3) Terrain calibration
- Terrain statistics are computed for both generated terrain and reference DEM.
- A bounded iterative heuristic adjusts selected procedural parameters.
- Calibration improves distribution-level similarity without copying DEM geometry.

### 4) Statistical comparison
- Elevation/slope distributions and scalar terrain metrics are compared.
- Metrics include roughness, drainage density, ridge spacing, and aggregate error.

### 5) Mission planning simulation
- A traversal cost grid is computed from terrain characteristics.
- A* pathfinding finds route candidates between start and goal.
- Output includes overlays and mission-relevant statistics (cost, elevation gain/loss, path efficiency).

## 8. Installation
### Prerequisites
- Python 3.10 or newer
- Windows/Linux/macOS
- Optional CUDA-capable GPU for AI/remaster acceleration

### Setup
```bash
# 1) Clone and enter repository
cd "Terrain Sim and Mission"

# 2) Create virtual environment
python -m venv .venv

# 3) Activate environment
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

# 4) Install dependencies (recommended stable set)
pip install -r requirements_stable.txt

# Optional: full dependency profile
# pip install -r requirements.txt
```

### Verify installation
```bash
python -m pytest tests -v
```

## 9. Usage
### End-to-end pipeline demo
```bash
python pipeline_demo.py --prompt "rugged alpine terrain with valleys" --interactive-3d
```

### Mission planning mode
```bash
python pipeline_demo.py --prompt "semi-arid canyon" --simulate-path
```

### DEM calibration demo
```bash
# Synthetic reference DEM
python dem_calibration_demo.py --iterations 5

# Real DEM file
python dem_calibration_demo.py --dem data/dem_files/your_dem.tif --iterations 8
```

### Generate terrain with calibrated parameters
```bash
python generate_calibrated_terrain.py --seed 42 --size 512 --interactive-3d
```

### Optional top-texture remaster
```bash
python generate_calibrated_terrain.py --seed 42 --size 320 --interactive-3d --remaster-top
```

### Run API server
```bash
python serve.py
```

## 10. Example Outputs
Typical outputs are written under `Output/` and may include:
- Heightmaps (`heightmap.npy`, `heightmap.png`)
- 3D renders (`terrain_3d_render.png`, interactive session artifacts)
- Mesh files (`mesh.vtk` and other export formats)
- Calibration plots and reports (before/after comparisons, metric charts)
- Biome and terrain-attribute visualizations
- Cost maps and mission path overlays
- Metadata files (`metadata.json`) describing parameters and run settings

## 11. Contribution Guidelines
Contributions are welcome for research and engineering improvements.

1. Fork the repository and create a feature branch.
2. Keep changes focused and include tests for behavior changes.
3. Follow existing coding style and module boundaries.
4. Run the test suite before opening a pull request.
5. Document new flags, modules, and outputs in this README.

Suggested workflow:
```bash
git checkout -b feature/your-feature-name
python -m pytest tests -v
```

## 12. License
License information to be added.

## 13. Author(s)
- Project Author: [Pratham Dabhane](https://github.com/Pratham-Dabhane), [Sanskruti Sugandhi](https://github.com/sanskruti048), [Vedang Chikane](https://github.com/vedangchikane04), [Sahil Saste](https://github.com/Sahil-Saste)
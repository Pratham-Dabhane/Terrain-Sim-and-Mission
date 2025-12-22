# Terrain Simulation and Mission Planning

**AI-powered terrain generation with intelligent mission pathfinding**

A complete pipeline for generating photorealistic 3D terrain from text prompts and planning optimal navigation paths through cost-aware A* pathfinding.

---

## 🎯 Features

- **🗺️ Text-to-Terrain Generation**: Generate realistic heightmaps from natural language prompts
- **🎨 Photorealistic 3D Rendering**: GPU-accelerated interactive visualization with physically-based materials  
- **🧭 Intelligent Pathfinding**: Cost-aware A* algorithm for optimal route planning
- **📊 Terrain Analysis**: Automatic cost map generation based on elevation and slope
- **🎮 Interactive Visualization**: Real-time 3D terrain exploration with mouse controls
- **💾 Export Support**: VTK mesh export, PNG visualizations, mission overlays

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
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

# Generate with mission path planning
python pipeline_demo.py --prompt "desert dunes" --simulate-path

# Generate without remastering (faster)
python pipeline_demo.py --prompt "rolling hills" --no-remaster
```

---

## 📖 Pipeline Overview

### Prompt → Terrain → Planner

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│ Text Prompt  │ --> │ Terrain Generator│ --> │ Cost Map      │
│ "mountains"  │     │ (Perlin Noise)   │     │ (Elevation +  │
└──────────────┘     └──────────────────┘     │  Slope)       │
                              │                └───────────────┘
                              ↓                        │
                     ┌──────────────────┐              ↓
                     │ 3D Mesh          │     ┌───────────────┐
                     │ (PyVista)        │     │ A* Pathfinder │
                     └──────────────────┘     │ (Optimal Path)│
                              │                └───────────────┘
                              ↓                        │
                     ┌──────────────────┐              ↓
                     │ Interactive      │     ┌───────────────┐
                     │ Viewer (GPU)     │     │ Path Overlay  │
                     └──────────────────┘     └───────────────┘
```

### Phase Breakdown

#### Phase 1: Prompt Parsing
- **Input**: Natural language text (e.g., "snowy mountain peaks")
- **Process**: Extract keywords, determine terrain type, set generation parameters
- **Output**: Terrain parameters (octaves, scale, persistence, biome)
- **Module**: `pipeline/prompt_parser.py`

#### Phase 2: Heightmap Generation
- **Input**: Terrain parameters from Phase 1
- **Process**: Generate 2D heightmap using multi-octave Perlin noise
- **Output**: Normalized heightmap array (256×256, values 0-1)
- **Module**: `pipeline/procedural_noise_utils.py`
- **Performance**: ~10-12 seconds for 256×256 terrain

#### Phase 3: Texture Mapping
- **Input**: Heightmap from Phase 2
- **Process**: Apply realistic colors based on elevation and slope
- **Output**: RGB texture array (256×256×3)
- **Module**: `pipeline/advanced_terrain_renderer.py`
- **Color Scheme**: Blues (water) → Greens (lowlands) → Browns (hills) → Grays/White (peaks)

#### Phase 4: 3D Mesh Generation
- **Input**: Heightmap + texture
- **Process**: Create structured 3D mesh with elevation-based coloring
- **Output**: PyVista mesh object (65,536 points, 130,050 cells)
- **Module**: `pipeline/mesh_visualize.py`
- **Export**: VTK format for external tools

#### Phase 5: Cost Map Computation
- **Input**: Heightmap
- **Process**: Calculate traversal cost based on:
  - Elevation penalty (×0.5 weight)
  - Slope penalty (×2.0 weight - steep slopes are expensive)
  - Water detection (elevation < 0.2 = impassable)
- **Output**: Normalized cost map (0=easy, 1=extreme)
- **Module**: `pipeline/cost_map.py`

#### Phase 6: Path Planning
- **Input**: Cost map + start/goal coordinates
- **Process**: A* pathfinding with diagonal movement
- **Output**: Optimal path waypoints + statistics
- **Module**: `pipeline/planner.py`
- **Features**: 
  - Cost-aware routing
  - Obstacle avoidance
  - Elevation change tracking

#### Phase 7: Visualization
- **Input**: Mesh + (optional) path
- **Process**: GPU-accelerated rendering with physically-based materials
- **Output**: Interactive 3D window OR static PNG
- **Module**: `pipeline/advanced_terrain_renderer.py`
- **Renderer**: PyVista + VTK (OpenGL backend)
- **GPU**: NVIDIA GPU support with automatic detection

---

## 💻 Usage Examples

### Example 1: Mountain Terrain with 3D Viewer
```bash
python pipeline_demo.py --prompt "rocky mountains with snow peaks" --interactive-3d --no-remaster
```
**Output**:
- `heightmap.png` - 2D elevation map
- `mesh.vtk` - 3D mesh export
- `visualization_3d_PHOTOREALISTIC.png` - Static render
- Interactive 3D window with mouse controls

### Example 2: Desert with Mission Planning
```bash
python pipeline_demo.py --prompt "sandy desert dunes" --simulate-path --no-remaster
```
**Output**:
- All terrain files from Example 1
- `cost_map.png` - Traversal cost visualization
- `mission_path_overlay.png` - Optimal path on terrain
- Path statistics (waypoints, cost, elevation changes)

### Example 3: Custom Seed for Reproducibility
```bash
python pipeline_demo.py --prompt "volcanic crater" --seed 12345 --no-remaster
```
**Output**: Same terrain every time with seed 12345

---

## 🎮 Interactive Controls

When using `--interactive-3d`, the 3D viewer supports:

- **Mouse Drag**: Rotate view
- **Scroll Wheel**: Zoom in/out
- **Right-Click + Drag**: Pan camera
- **'R' Key**: Reset camera
- **'Q' or ESC**: Quit viewer

---

## 📁 Output Structure

```
Output/
└── session_<timestamp>/
    ├── heightmap.png                    # 2D elevation map
    ├── enhanced_terrain.png             # Colored terrain (if remastered)
    ├── mesh.vtk                         # 3D mesh (VTK format)
    ├── visualization_3d.png             # Static 3D render
    ├── visualization_3d_PHOTOREALISTIC.png  # High-quality render
    ├── cost_map.png                     # Traversal cost visualization
    ├── mission_path_overlay.png         # Path on terrain
    └── metadata.json                    # Generation parameters
```

---

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_prompt_parameters.py -v
python -m pytest tests/test_cost_map.py -v
python -m pytest tests/test_planner.py -v

# Run with coverage
python -m pytest tests/ --cov=pipeline --cov-report=html
```

### Test Coverage

- **test_prompt_parameters.py**: Prompt parsing, parameter extraction, terrain generation
- **test_cost_map.py**: Cost calculation, biome modifiers, water/cliff detection
- **test_planner.py**: A* pathfinding, obstacle avoidance, path statistics

---

## ⚙️ Configuration

### Terrain Parameters

Controlled by prompt keywords:

| Terrain Type | Octaves | Scale | Persistence | Features |
|-------------|---------|-------|-------------|----------|
| Mountains   | 8       | 120   | 0.5         | High detail, sharp peaks |
| Desert      | 3       | 100   | 0.5         | Smooth dunes, low detail |
| Hills       | 4       | 100   | 0.4         | Rolling terrain |
| Canyon      | 6       | 110   | 0.6         | Layered features |
| Forest      | 6       | 100   | 0.5         | Mixed elevation |

### Material Properties

Terrain-specific physically-based rendering:

- **Snow/Ice**: High ambient (0.4), subtle specular (0.05), matte finish
- **Desert**: Warm tones, minimal specular (0.02), granular appearance
- **Rocky**: Mixed reflectivity (0.03), earthy colors
- **Forest**: Very matte (0.0 specular), organic texture

---

## 📊 Phase Checklist

Development phases and completion status:

- [x] **Phase 1**: Prompt parsing and parameter extraction
- [x] **Phase 2**: Perlin noise terrain generation
- [x] **Phase 3**: Realistic texture mapping (slope-based)
- [x] **Phase 4**: 3D mesh generation and VTK export
- [x] **Phase 5**: Cost map computation with biome modifiers
- [x] **Phase 6**: A* pathfinding with diagonal movement
- [x] **Phase 7**: GPU-accelerated interactive visualization
- [x] **Phase 8**: Mission planning with path overlay
- [x] **Phase 9**: Unit testing and documentation
- [x] **Phase 10**: Performance optimization (Perlin noise)

---

## 🛠️ Troubleshooting

### Common Issues

**Black Screen in Interactive Viewer**:
- Solution: Fixed - GPU driver issue resolved with proper event loop handling
- Ensure NVIDIA GPU drivers are up to date

**Slow Terrain Generation**:
- Use `--no-remaster` flag to skip SD ControlNet remastering
- Reduce octaves for faster generation (lower detail)

**Import Errors**:
- Ensure virtual environment is activated
- Reinstall: `pip install -r requirements_stable.txt`

---

## 📚 Additional Documentation

- **PIPELINE_README.md**: Detailed pipeline architecture
- **QUICKSTART.md**: Step-by-step beginner guide
- **API Documentation**: See inline docstrings in modules

---

## 🙏 Acknowledgments

- **PyVista**: 3D visualization library
- **noise**: Perlin/Simplex noise generation
- **NumPy**: Numerical computing
- **Matplotlib**: 2D plotting and visualization

---

**Made with ❤️ for terrain enthusiasts and mission planners**

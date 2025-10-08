# Quick Start Guide - AI Terrain Generation & Mission Planning

## 🚀 Get Started in 3 Steps  

### Step 1: Install Dependencies
```bash
# Clone the repository  
git clone https://github.com/Pratham-Dabhane/Terrain-Sim-and-Mission.git
cd "Terrain Sim and Mission"

# Install all requirements
pip install -r requirements.txt

# Or use stable requirements for guaranteed compatibility
pip install -r requirements_stable.txt
```

### Step 2: Test the Installation
```bash
# Quick test with default settings
python pipeline_demo.py --prompt "mountain terrain"

# Verify all components are working
python -c "from pipeline.config import get_default_config; print('✅ All imports successful')"
```

### Step 3: Generate Your First Terrain
```bash
# Basic terrain generation
python pipeline_demo.py --prompt "alpine valley with crystal clear lake"

# With mission planning
python pipeline_demo.py --prompt "mountainous terrain with valleys" --simulate-path

# Interactive 3D exploration
python pipeline_demo.py --prompt "volcanic landscape" --interactive-3d
```

## 🎯 What You'll Get

### 🌍 Terrain Generation Modes
- **AI Mode**: CLIP + aWCGAN neural network generation (default)
- **Procedural Mode**: Fractal Brownian Motion (fBM) noise generation
- **Diffusion Mode**: Optional AI diffusion model for enhanced realism

### 🎨 Visualization Features
- **Photorealistic 3D**: Cinema-quality rendering with advanced lighting and SSAO
- **Slope-based Coloring**: Realistic terrain colors based on elevation and gradients
- **High-Resolution Output**: Ultra-HD (1920x1080) static terrain visualization
- **Interactive 3D Viewer**: Real-time exploration with rotate/zoom/pan controls

### 🚁 Mission Planning System
- **A* Pathfinding**: Intelligent route planning with terrain-aware cost analysis
- **Interactive Point Selection**: Click to select start (red) and goal (blue) points
- **Mission Visualization**: Path overlay with cost analysis and waypoint tracking
- **Cost Analysis**: Elevation and slope penalties for realistic navigation planning

## 💡 Smart Prompt Examples

The system intelligently parses terrain features from natural language:

### ⛰️ Mountain Terrains
- `"snow-capped mountain peaks with alpine valleys"` → Mountain + Snow + Valley features
- `"rugged mountain range with steep cliffs"` → Mountain + Rocky + Steep features
- `"gentle rolling hills with green meadows"` → Hills + Green + Gentle features

### 🏝️ Water Features  
- `"tropical island with pristine sandy beaches"` → Island + Beach + Tropical features
- `"mountain lake surrounded by pine forests"` → Lake + Mountain + Forest features
- `"river valley with meandering streams"` → River + Valley + Water features

### 🏜️ Specialized Landscapes
- `"volcanic landscape with lava craters and rocky outcrops"` → Volcanic + Crater + Rocky features
- `"arctic tundra with frozen lakes and snow cover"` → Arctic + Flat + Lake + Snow features
- `"desert oasis with palm trees and sand dunes"` → Desert + Oasis + Dunes features

## 🔧 Advanced Usage

### Configuration System
```bash
# Enable specific features
python pipeline_demo.py --prompt "terrain" --config-flags procedural,mission_sim,photorealistic

# Use different generation modes
python pipeline_demo.py --prompt "landscape" --mode procedural
python pipeline_demo.py --prompt "landscape" --mode ai
```

### Interactive Viewer
```bash
# View latest generated terrain
python interactive_terrain_viewer.py --latest

# View specific session
python interactive_terrain_viewer.py --session session_1759684013

# List all sessions
python interactive_terrain_viewer.py --list
```

### Mission Planning Workflow
1. **Generate Terrain**: Run with `--simulate-path` flag
2. **Select Points**: Left-click for start (red), right-click for goal (blue)
3. **Close Plot**: Window closure triggers A* pathfinding computation
4. **View Results**: Check `mission_path_overlay.png` for optimal route

## 🔧 Troubleshooting

### Installation Issues
```bash
# Update pip and try again
pip install --upgrade pip
pip install -r requirements.txt

# For CUDA issues
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Performance Issues
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Monitor GPU usage
nvidia-smi
```

### Import Errors
```bash
# Verify all pipeline modules
python -c "import sys; sys.path.append('pipeline'); from config import get_default_config; print('✅ Pipeline imports working')"
```

**For advanced features (diffusion models):**
```bash
pip install diffusers transformers accelerate
```

**Windows users:** Double-click `run_terrain.bat` or run `run_terrain.ps1` in PowerShell

## 📁 Project Structure

```
Terrain Sim and Mission/
├── terrain_prototype.py    # Main prototype
├── demo.py                 # Demo examples
├── test_terrain.py         # Test functionality
├── install.py              # Installation helper
├── requirements.txt        # Dependencies
├── README.md               # Full documentation
└── QUICKSTART.md           # This file
```

## 🎮 Ready to Generate Terrain?

Run the prototype and start creating amazing landscapes from text!

```bash
python terrain_prototype.py
```

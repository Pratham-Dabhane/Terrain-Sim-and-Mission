# 🌍 AI-Powered Terrain Generation & Mission Planning Pipeline

A comprehensive AI system that generates photorealistic 3D terrains from natural language descriptions and provides intelligent mission planning capabilities. Built with advanced machine learning technologies including CLIP, Wasserstein GANs, procedural generation, and A* pathfinding.

## ✨ Key Features

### 🎯 Terrain Generation
- **Text-to-Terrain AI**: Convert natural language into realistic 3D landscapes
- **Multiple Generation Modes**: AI-powered (aWCGAN), Procedural (fBM noise), and AI diffusion
- **Smart Prompt Parsing**: Intelligent keyword extraction for terrain types and features
- **Slope-based Color Mapping**: Realistic terrain coloring based on elevation and slope

### 🎨 Advanced Visualization  
- **Photorealistic Rendering**: Cinema-quality 3D visualization with SSAO and advanced lighting
- **Multi-light Setup**: Golden hour lighting with 4-point illumination system
- **High-Resolution Output**: Ultra-HD (1920x1080) static terrain visualization
- **Interactive 3D Viewer**: Real-time terrain exploration with rotate/zoom/pan controls

### 🚁 Mission Planning
- **A* Pathfinding**: Intelligent route planning with terrain-aware cost analysis
- **Interactive Point Selection**: Click-to-select start and goal points on terrain
- **Mission Visualization**: Path overlay with cost analysis and waypoint tracking
- **Terrain Cost Analysis**: Elevation and slope-based navigation difficulty assessment

### 🔧 Technical Features
- **Multiple Output Formats**: 2D images, 3D meshes (VTK/PLY/OBJ), mission overlays
- **Enhanced Metadata**: Comprehensive terrain statistics and generation analytics
- **Configuration System**: Feature flags for enabling/disabling pipeline components
- **Web API Support**: REST API for web applications and integrations
- **⚡ GPU Accelerated**: CUDA support for fast generation on modern GPUs

## 🏗️ System Architecture

```
Text Input → Prompt Parser → Multi-Mode Generation → Enhanced Rendering → Mission Planning
     ↓             ↓              ↓                    ↓                ↓
"alpine valley" → Keywords → [AI/Procedural/Diffusion] → Photorealistic → A* Pathfinding
```

**Complete Pipeline:**
1. **Intelligent Prompt Parsing**: Extracts terrain keywords and features from natural language
2. **Multi-Mode Generation**: 
   - **AI Mode**: CLIP + aWCGAN neural network generation
   - **Procedural Mode**: Fractal Brownian Motion (fBM) noise generation  
   - **Diffusion Mode**: Optional AI diffusion model for heightmap creation
3. **Slope-based Color Mapping**: Realistic terrain coloring based on elevation and gradients
4. **Advanced 3D Rendering**: Cinema-quality visualization with SSAO, multi-light setup
5. **Mission Planning**: A* pathfinding with interactive point selection and cost analysis
6. **Enhanced Metadata**: Comprehensive terrain statistics and generation analytics

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8+ required
- **GPU**: CUDA-compatible GPU recommended (tested on GTX 1650)
- **RAM**: 8GB+ recommended for large terrain generation

### Installation

```bash
# Clone the repository
git clone https://github.com/Pratham-Dabhane/Terrain-Sim-and-Mission.git
cd "Terrain Sim and Mission"

# Install dependencies
pip install -r requirements.txt

# Optional: Install stable requirements for guaranteed compatibility
pip install -r requirements_stable.txt
```

## 🎮 Usage Examples

### 🚀 Terrain Generation with Mission Planning

```bash
# Generate terrain with mission simulation
python pipeline_demo.py --prompt "alpine valley with river" --simulate-path

# Generate with interactive 3D viewer
python pipeline_demo.py --prompt "majestic mountain lake with forested shores" --interactive-3d

# Use procedural generation mode
python pipeline_demo.py --prompt "rolling hills terrain" --mode procedural

# Force AI generation mode
python pipeline_demo.py --prompt "volcanic landscape" --mode ai
```

**Example Prompts (with intelligent parsing):**
- `"snow-capped mountain peaks with alpine valleys"` → Mountain + Snow + Valley features
- `"tropical island with pristine beaches"` → Island + Beach + Tropical features  
- `"volcanic landscape with lava craters"` → Volcanic + Crater + Rocky features
- `"rolling hills with dense forest coverage"` → Hills + Forest + Green features
- `"arctic tundra with frozen lakes"` → Arctic + Flat + Lake features

### 🎯 Mission Planning & Pathfinding

The A* pathfinding system provides intelligent route planning with terrain-aware cost analysis:

```bash
# Generate terrain and plan mission automatically
python pipeline_demo.py --prompt "mountainous terrain" --simulate-path

# Interactive point selection:
# 1. Left-click to select start point (red)
# 2. Right-click to select goal point (blue)  
# 3. Close the plot window to compute optimal path
# 4. View results in mission_path_overlay.png
```

**Cost Analysis Features:**
- **Elevation Penalty**: Higher elevations increase travel cost
- **Slope Penalty**: Steep terrain is harder to traverse
- **Optimal Routing**: A* algorithm finds lowest-cost path
- **Visual Feedback**: Path displayed with cost information

### � Interactive Viewer (Standalone)

```bash
# View latest generated terrain interactively
python interactive_terrain_viewer.py --latest

# View specific terrain session  
python interactive_terrain_viewer.py --session session_1759684013

# List all available sessions
python interactive_terrain_viewer.py --list
```

### �🌐 Web API Server

```bash
# Start REST API server
python serve.py

# Generate terrain via API
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "mountain landscape", "interactive": true}'
```

### 🎨 Demo & Showcase

```bash
# Run realistic terrain demo
python demo_realistic_terrain.py
```

## 📁 Output Files

Each terrain generation creates a complete session folder:

```
Output/session_XXXXXXXXX/
├── heightmap.png                      # Grayscale elevation data
├── enhanced_terrain.png               # Realistic colored terrain  
├── mesh.vtk                          # 3D mesh (for Blender/Unity)
├── visualization_3d.png              # Standard 3D render
├── visualization_3d_PHOTOREALISTIC.png # Cinema-quality render
└── metadata.json                     # Generation parameters
```

**Supported Export Formats:**
- **Images**: PNG (2D heightmaps, enhanced terrain, 3D renders)
- **3D Meshes**: VTK, PLY, OBJ, STL (for external 3D software)
- **Data**: JSON metadata with all generation parameters

## ⚡ Performance & System Requirements

### 🔧 Hardware Specifications
- **CPU**: Modern multi-core processor
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible (GTX 1650 or better) for optimal performance
- **Storage**: 5GB+ free space for models and output

### ⏱️ Generation Times (GTX 1650)
- **Text → Heightmap**: ~1-2 seconds
- **3D Mesh Generation**: ~1-2 seconds  
- **Photorealistic Rendering**: ~2-3 seconds
- **Interactive Launch**: Instant
- **Total Pipeline**: ~5-8 seconds per terrain

### 🚀 Performance Optimizations
- **GPU Acceleration**: CUDA support for all AI components
- **Memory Efficient**: Optimized for 4GB VRAM systems
- **Batch Processing**: Generate multiple terrains efficiently
- **Caching**: CLIP model loaded once, reused for all generations

### Diffusion Integration

- **Model**: Stable Diffusion 1.5
- **ControlNet**: Depth map conditioning
- **Input Size**: 512x512 pixels
- **Inference Steps**: 20 (configurable)

### Terrain Features

## 🔧 Technical Architecture

### 🧠 AI Components
- **CLIP Text Encoder**: OpenAI's vision-language model for semantic understanding
- **aWCGAN Generator**: 3.3M parameter Wasserstein GAN with gradient penalty
- **Realistic Enhancer**: Biome-aware terrain coloring with 4 specialized color schemes
- **Advanced Renderer**: 3-point lighting system for cinema-quality visualization

### 🎨 3D Pipeline
- **Mesh Generation**: PyVista StructuredGrid with 65,536 vertices
- **Material System**: Physically-based rendering for different biomes
- **Interactive Engine**: Real-time manipulation with mouse/keyboard controls
- **Export System**: Multiple format support (VTK, PLY, OBJ, STL)

### 🌐 Integration Capabilities
- **Web API**: FastAPI-based REST endpoints with async processing
- **File System**: Organized session management with metadata tracking
- **External Tools**: Direct export to Blender, Unity, Unreal Engine
- **Batch Processing**: Multiple terrain generation with consistent quality

## 🎯 Applications & Use Cases

- **🎮 Game Development**: Rapid terrain prototyping for open-world games
- **🎬 Film & Animation**: Landscape creation for visual effects
- **🏗️ Architecture**: Site planning and environmental visualization  
- **📚 Education**: Geological and geographical concept demonstration
- **🔬 Research**: AI-driven terrain analysis and generation studies
- **🎨 Digital Art**: Landscape reference and inspiration generation

## 🚀 Development Roadmap

### ✅ Completed Features
- [x] **Interactive 3D Visualization** - Complete with PyVista
- [x] **Photorealistic Rendering** - Advanced lighting with SSAO
- [x] **Mission Planning System** - A* pathfinding with cost analysis  
- [x] **Multi-Mode Generation** - AI, Procedural, and Diffusion modes
- [x] **Intelligent Prompt Parsing** - Keyword extraction and feature mapping
- [x] **Slope-based Color Mapping** - Realistic terrain visualization
- [x] **Enhanced Metadata System** - Comprehensive terrain analytics
- [x] **Configuration Management** - Feature flags and modular design
- [x] **Web API Integration** - REST API for web applications

### 🔄 Future Enhancements  
- [ ] **Real-time Terrain Editing** - Interactive terrain modification
- [ ] **Multi-scale Generation** - Hierarchical detail levels
- [ ] **Weather System Integration** - Dynamic environmental effects
- [ ] **Multi-agent Mission Planning** - Collaborative pathfinding
- [ ] **Mobile App Support** - iOS/Android terrain viewer

## 📄 License

This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **OpenAI CLIP**: For revolutionary vision-language understanding
- **PyVista Team**: For excellent 3D visualization capabilities  
- **PyTorch Community**: For deep learning framework excellence
- **FastAPI**: For high-performance web API development

---

<div align="center">

**🌍 Transform Text into Worlds with AI 🚀**

[Documentation](PIPELINE_README.md) • [Quick Start](QUICKSTART.md) • [Examples](#-usage-examples) • [API Reference](serve.py)

</div>

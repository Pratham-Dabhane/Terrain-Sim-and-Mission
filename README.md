# 🌍 AI-Powered Terrain Generation Pipeline

A production-ready AI system that generates photorealistic 3D terrains from natural language descriptions using advanced machine learning technologies including CLIP, Wasserstein GANs, and interactive 3D visualization.

## ✨ Key Features

- **🎯 Text-to-Terrain AI**: Convert natural language into realistic 3D landscapes
- **🧠 CLIP + aWCGAN Pipeline**: OpenAI CLIP text encoding with Wasserstein Conditional GAN generation
- **🎨 Photorealistic Rendering**: Cinema-quality 3D visualization with advanced lighting
- **🎮 Interactive 3D Viewer**: Real-time terrain exploration with rotate/zoom/pan controls
- **🌲 Intelligent Biome Detection**: Automatic terrain coloring (forest, desert, mountain, arctic)
- **🔧 Multiple Output Formats**: 2D images, 3D meshes (VTK/PLY/OBJ), interactive views
- **🚀 Web API Support**: REST API for web applications and integrations
- **⚡ GPU Accelerated**: CUDA support for fast generation on modern GPUs

## 🏗️ System Architecture

```
Text Input → CLIP Encoder → aWCGAN Generator → Heightmap → 3D Mesh → Interactive Viewer
     ↓              ↓              ↓            ↓           ↓            ↓
"mountain lake" → [512D vector] → Neural Net → Elevation → Photorealistic → User Controls
```

**Complete Pipeline:**
1. **CLIP Text Encoding**: Converts natural language to semantic embeddings
2. **aWCGAN Generation**: Creates terrain heightmaps from text embeddings  
3. **Biome Enhancement**: Applies realistic colors based on terrain type
4. **3D Mesh Creation**: Generates interactive 3D models with 65K+ vertices
5. **Photorealistic Rendering**: Cinema-quality visualization with 3-point lighting
6. **Interactive Exploration**: Real-time 3D interaction like matplotlib figures

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
```

## 🎮 Usage Examples

### 🚀 Interactive Terrain Generation (Recommended)

```bash
# Generate terrain with interactive 3D viewer
python pipeline_demo.py --prompt "majestic mountain lake with forested shores" --interactive-3d

# Quick generation without interaction
python pipeline_demo.py --prompt "vast desert dunes with rocky outcrops"
```

**Example Prompts:**
- `"snow-capped mountain peaks with alpine valleys"`
- `"tropical island with pristine beaches"`
- `"volcanic landscape with lava craters"`
- `"rolling hills with dense forest coverage"`
- `"arctic tundra with frozen lakes"`

### 🎯 Interactive Viewer (Standalone)

```bash
# View latest generated terrain interactively
python interactive_terrain_viewer.py --latest

# View specific terrain session
python interactive_terrain_viewer.py --session session_1759684013

# List all available sessions
python interactive_terrain_viewer.py --list
```

### 🌐 Web API Server

```bash
# Start REST API server
python serve.py

# Generate via API
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

## 🚀 Future Roadmap

- [x] **Interactive 3D Visualization** - Complete ✅
- [x] **Photorealistic Rendering** - Complete ✅  
- [x] **Web API Integration** - Complete ✅
- [x] **Multi-format Export** - Complete ✅
- [ ] **Real-time Terrain Editing** - Planned
- [ ] **Multi-scale Generation** - Planned
- [ ] **Custom Model Training** - Available
- [ ] **Mobile App Support** - Future

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

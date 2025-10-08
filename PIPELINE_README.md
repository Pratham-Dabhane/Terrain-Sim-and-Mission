# AI-Powered Terrain Generation & Mission Planning Pipeline

A comprehensive end-to-end system for generating realistic 3D terrains from text descriptions and providing intelligent mission planning capabilities. Built with state-of-the-art AI techniques including CLIP-conditioned Wasserstein GANs, procedural generation, Stable Diffusion, and A* pathfinding algorithms.

## 🚀 Core Features

### 🎯 Multi-Mode Terrain Generation
- **AI-Powered Generation**: CLIP-conditioned aWCGAN with advanced text understanding
- **Procedural Generation**: Fractal Brownian Motion (fBM) noise with terrain variants  
- **AI Diffusion**: Optional Stable Diffusion + ControlNet heightmap enhancement
- **Intelligent Prompt Parsing**: Automatic keyword extraction and feature mapping

### 🎨 Advanced Visualization
- **Photorealistic Rendering**: Cinema-quality 3D visualization with SSAO and advanced lighting
- **Slope-based Color Mapping**: Realistic terrain coloring based on elevation and gradients
- **Multi-light Setup**: Golden hour lighting with 4-point illumination system
- **High-Resolution Output**: Ultra-HD (1920x1080) static terrain visualization

### 🚁 Mission Planning System  
- **A* Pathfinding**: Terrain-aware route optimization with cost analysis
- **Interactive Point Selection**: Click-to-select start and goal points
- **Cost Analysis**: Elevation and slope-based navigation difficulty assessment
- **Mission Visualization**: Path overlay with waypoint tracking and statistics

### 🔧 Technical Capabilities
- **Memory Optimization**: Designed for low VRAM systems (GTX 1650, 4GB VRAM)
- **Configuration System**: Feature flags for modular component control
- **Enhanced Metadata**: Comprehensive terrain analytics and generation statistics
- **FastAPI Server**: REST API for web deployment and integration
- **Multiple Export Formats**: PLY, OBJ, STL, VTK mesh export capabilities

## 🏗️ System Architecture

```
Text Input → Prompt Parser → Generation Router → Terrain Enhancement → Mission Planning
     ↓             ↓              ↓                    ↓                ↓ 
"mountain lake" → Keywords → [AI/Procedural/Diffusion] → Color Mapping → A* Pathfinding
     ↓             ↓              ↓                    ↓                ↓
   Config → Feature Flags → Multi-mode Pipeline → 3D Rendering → Interactive Viewer
```

### Core Pipeline Components

#### 🧠 AI & Text Processing
1. **CLIP Text Encoder** (`clip_encoder.py`): Advanced text-to-embedding conversion with prompt enhancement
2. **Prompt Parser** (`prompt_parser.py`): Intelligent keyword extraction with 20+ terrain features
3. **aWCGAN Models** (`models_awcgan.py`): Generator and Critic with FiLM/AdaIN conditioning
4. **AI Heightmap Generator** (`ai_heightmap_generator.py`): Optional diffusion-based terrain generation

#### 🌍 Terrain Generation & Enhancement  
5. **Procedural Noise Utils** (`procedural_noise_utils.py`): fBM fractal terrain generation with variants
6. **Terrain Texture Mapper** (`terrain_texture_mapper.py`): Slope-based realistic color mapping
7. **Realistic Terrain Enhancer** (`realistic_terrain_enhancer.py`): SD + ControlNet enhancement
8. **Advanced Terrain Renderer** (`advanced_terrain_renderer.py`): Cinema-quality 3D visualization

#### 🚁 Mission Planning & Visualization
9. **Mission Simulator** (`mission_simulator.py`): A* pathfinding with interactive point selection
10. **Mesh Visualizer** (`mesh_visualize.py`): PyVista-based 3D mesh generation and rendering
11. **Configuration System** (`config.py`): Feature flags and modular component control

#### 🔧 Supporting Systems
12. **Training System** (`train_awcgan.py`): Memory-efficient training loop for GTX 1650
13. **Data Pipeline** (`data.py`): DES heightmap dataset loader with preprocessing
14. **API Server** (`serve.py`): FastAPI endpoint for web deployment and integration

## 🔧 Installation

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended: GTX 1650 or better)
- 8GB+ RAM
- 10GB+ disk space

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd terrain-generation-pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Optional: Install memory optimizations**
```bash
# For better performance on low VRAM
pip install xformers
```

## 🚀 Quick Start

### Option 1: Terrain Generation with Mission Planning
```bash
# Complete pipeline with pathfinding
python pipeline_demo.py --prompt "alpine valley with lakes" --simulate-path

# AI generation mode (default)
python pipeline_demo.py --prompt "mountainous terrain with rivers" --mode ai

# Procedural generation mode  
python pipeline_demo.py --prompt "rolling hills landscape" --mode procedural
```

### Option 2: Interactive 3D Exploration
```bash
# Generate with interactive viewer
python pipeline_demo.py --prompt "volcanic landscape" --interactive-3d

# View existing terrain sessions
python interactive_terrain_viewer.py --latest
```

### Option 3: API Server & Integration
```bash
# Start the REST API server
python serve.py

# Generate terrain via API
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "snow-capped mountain peaks", "mode": "ai"}'
```

### Option 4: Configuration-based Generation
```python
from pipeline.config import PipelineConfig, get_default_config

# Customize pipeline features
config = get_default_config()
config.mission_sim_enabled = True
config.photorealistic_render_enabled = True
config.ai_heightmap_enabled = False

# Run with custom config
pipeline = TerrainPipelineDemo(config=config)
result = pipeline.generate_terrain("desert oasis with palm trees")
```

## 📚 Usage Examples

### Basic Terrain Generation
```python
from pipeline.clip_encoder import CLIPTextEncoderWithProcessor
from pipeline.models_awcgan import CLIPConditionedGenerator
from pipeline.mesh_visualize import TerrainMeshGenerator

# Initialize components
clip_encoder = CLIPTextEncoderWithProcessor()
generator = CLIPConditionedGenerator().cuda()
mesh_generator = TerrainMeshGenerator()

# Generate terrain
prompt = "snow-capped mountain peaks with deep valleys"
clip_embedding = clip_encoder.encode_text(prompt)
noise = torch.randn(1, 128).cuda()
heightmap = generator(noise, clip_embedding)

# Create 3D mesh
mesh = mesh_generator.generate_mesh(heightmap)
mesh.save("terrain.ply")
```

### Complete Pipeline with Enhancement
```python
from pipeline_demo import TerrainPipelineDemo

pipeline = TerrainPipelineDemo()
result = pipeline.run_complete_pipeline(
    prompt="coastal cliffs with beaches and ocean",
    enable_remastering=True,
    output_size=(1024, 1024)
)

print(f"Files saved to: {result['file_paths']}")
```

## 🎯 Training Your Own Models

### 1. Prepare Dataset
```python
from pipeline.data import create_synthetic_data, DESDataModule

# Create synthetic training data
create_synthetic_data("data/heightmaps", num_samples=1000)

# Or use your own DEM data in data/heightmaps/
```

### 2. Train the aWCGAN
```python
python pipeline/train_awcgan.py
```

Training configuration is optimized for GTX 1650:
- Batch size: 1-2
- Mixed precision (FP16)
- Gradient checkpointing
- CPU offloading for memory efficiency

### 3. Monitor Training
```python
# View generated samples during training
# Samples saved to: samples/samples_epoch_*.png
# Checkpoints saved to: checkpoints/checkpoint_epoch_*.pth
```

## 🌐 API Reference

### Generate Terrain
**POST** `/generate`

```json
{
  "prompt": "mountainous terrain with rivers",
  "negative_prompt": "blurry, low quality",
  "seed": 42,
  "width": 512,
  "height": 512,
  "num_inference_steps": 20,
  "guidance_scale": 7.5,
  "controlnet_scale": 0.8,
  "enhance_realism": true,
  "export_formats": ["ply", "obj"]
}
```

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "pending",
  "message": "Terrain generation started"
}
```

### Check Status
**GET** `/status/{task_id}`

### Download Results
**GET** `/download/{task_id}/{file_type}`

File types: `heightmap`, `remastered`, `mesh_ply`, `mesh_obj`, `visualization`, `all`

## 🖼️ Example Results

| Prompt | Heightmap | Remastered | 3D Mesh |
|--------|-----------|------------|---------|
| "Mountain ranges with snow peaks" | ![Heightmap](examples/mountain_heightmap.png) | ![Remastered](examples/mountain_remastered.png) | ![3D](examples/mountain_3d.png) |
| "Desert with sand dunes" | ![Heightmap](examples/desert_heightmap.png) | ![Remastered](examples/desert_remastered.png) | ![3D](examples/desert_3d.png) |

## ⚡ Performance Optimization

### For Low VRAM (GTX 1650, 4GB)
- Uses FP16 mixed precision
- CPU offloading for large models
- Batch size 1-2
- Gradient checkpointing
- xFormers attention optimization

### Memory Usage
- **Training**: ~3.5GB VRAM
- **Inference**: ~2.8GB VRAM
- **Full Pipeline**: ~3.8GB VRAM peak

## 🛠️ Configuration

### Training Config (`train_awcgan.py`)
```python
config = {
    'batch_size': 1,          # Small for GTX 1650
    'lr_g': 0.0001,          # Generator learning rate
    'lr_c': 0.0002,          # Critic learning rate
    'lambda_gp': 10.0,       # Gradient penalty weight
    'n_critic': 5,           # Critic updates per generator
    'mixed_precision': True,  # FP16 optimization
}
```

### SD Enhancement Config
```python
remaster_config = {
    'use_fp16': True,
    'enable_cpu_offload': True,
    'num_inference_steps': 20,  # Reduced for speed
    'controlnet_conditioning_scale': 0.8,  # Preserve geometry
}
```

## 📊 Model Architecture

### aWCGAN Generator
- **Input**: Noise (128D) + CLIP embedding (512D)
- **Architecture**: Progressive upsampling with FiLM conditioning
- **Output**: 256×256 heightmap
- **Parameters**: ~2.1M

### aWCGAN Critic
- **Input**: Heightmap + CLIP embedding
- **Architecture**: Convolutional discriminator with spectral normalization
- **Loss**: Wasserstein distance + gradient penalty
- **Parameters**: ~1.8M

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config
config['batch_size'] = 1

# Enable CPU offloading
pipeline = TerrainRemaster(enable_cpu_offload=True)
```

**Import Errors**
```bash
# Install missing dependencies
pip install transformers diffusers pyvista

# For visualization issues
pip install vtk
```

**Slow Generation**
```bash
# Install xformers for memory efficiency
pip install xformers

# Reduce inference steps
num_inference_steps = 10
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stable Diffusion**: Runway ML and Stability AI
- **CLIP**: OpenAI
- **PyVista**: 3D visualization framework
- **Diffusers**: Hugging Face diffusion library
- **ControlNet**: Lvmin Zhang and Maneesh Agrawala

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

## 🔮 Roadmap

- [ ] Support for larger resolutions (1024×1024+)
- [ ] Multiple ControlNet conditioning types
- [ ] Real DEM dataset integration
- [ ] Texture synthesis for enhanced realism
- [ ] WebGL viewer for 3D meshes
- [ ] Batch processing optimization
- [ ] Advanced terrain physics simulation

---

**Built with ❤️ using PyTorch, Stable Diffusion, and PyVista**
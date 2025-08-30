# Generative AI Terrain Prototype

A Python prototype that generates 2.5D terrain from text prompts using a combination of GAN (Generative Adversarial Network) and Diffusion models.

## Features

- **Text-to-Terrain Generation**: Convert natural language descriptions into realistic terrain
- **GAN Component**: Uses a simplified StyleGAN2 architecture to generate base heightmaps
- **Diffusion Component**: Integrates with Stable Diffusion 1.5 + ControlNet for realistic terrain refinement
- **Smart Terrain Features**: Automatically adds mountains, valleys, rivers, forests, and desert features based on text prompts
- **Multiple Visualization Options**: 2D heightmap, 2.5D colored terrain, and 3D surface plots
- **Fallback Support**: Works even without heavy ML models installed

## Architecture

```
Text Prompt → GAN Generator → Heightmap → Diffusion Refinement → 2.5D Terrain
```

1. **Text Analysis**: Parses the input prompt for terrain features
2. **GAN Generation**: Creates a base heightmap using neural network generation
3. **Feature Addition**: Applies terrain-specific modifications (mountains, rivers, etc.)
4. **Diffusion Refinement**: Uses Stable Diffusion with ControlNet for realistic enhancement
5. **Visualization**: Displays results in multiple formats

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster diffusion model inference)

### Quick Install

```bash
# Clone or download the project
cd "Terrain Sim and Mission"

# Install dependencies
pip install -r requirements.txt
```

### Optional: Install Diffusion Models

For full functionality with Stable Diffusion:

```bash
pip install diffusers transformers accelerate
```

## Usage

### Basic Usage

```bash
python terrain_prototype.py
```

The program will prompt you to enter a terrain description. Examples:

- "mountainous terrain with rivers and valleys"
- "desert landscape with sand dunes"
- "forest terrain with rolling hills"
- "coastal cliffs with rocky outcrops"
- "alpine landscape with snow-capped peaks"

### Programmatic Usage

```python
from terrain_prototype import prompt_to_heightmap_gan, refine_with_diffusion, visualize_terrain

# Generate terrain from prompt
prompt = "mountainous terrain with rivers and valleys"
heightmap = prompt_to_heightmap_gan(prompt, size=256)

# Refine with diffusion (optional)
refined_terrain = refine_with_diffusion(heightmap, prompt)

# Visualize results
visualize_terrain(refined_terrain, heightmap, prompt, save_path="my_terrain.png")
```

## Output

The prototype generates:

1. **Heightmap**: Grayscale elevation data (0-1 range)
2. **2.5D Terrain**: Color-coded terrain with realistic textures
3. **3D Visualization**: Interactive 3D surface plot
4. **Saved Image**: High-resolution output saved as PNG

## Customization

### Adding New Terrain Features

Extend the `_apply_prompt_modifications()` function:

```python
def _add_custom_feature(heightmap):
    # Your custom terrain generation logic
    return modified_heightmap

# Add to the prompt modifications
if 'custom' in prompt_lower:
    heightmap = _add_custom_feature(heightmap)
```

### Modifying Color Schemes

Edit the `_apply_terrain_colors()` function to change terrain appearance:

```python
colors = {
    'water': [0.2, 0.4, 0.8],      # Blue
    'sand': [0.9, 0.8, 0.6],       # Sand
    'grass': [0.3, 0.6, 0.2],      # Green
    # Add your custom colors
}
```

## Performance Notes

- **Without GPU**: Basic terrain generation works on CPU
- **With GPU**: Diffusion refinement is significantly faster
- **Memory Usage**: ~2-4GB RAM for basic operation, 8GB+ for full diffusion models
- **Generation Time**: 5-30 seconds for basic terrain, 1-5 minutes with diffusion

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing packages with `pip install package_name`
2. **CUDA Errors**: Set `CUDA_VISIBLE_DEVICES=""` to force CPU usage
3. **Memory Issues**: Reduce terrain size (e.g., 128 instead of 256)
4. **Model Download Issues**: Check internet connection and firewall settings

### Fallback Modes

The prototype automatically falls back to simpler methods if:
- PyTorch is not available (uses NumPy-based generation)
- Diffusion models are not available (uses basic enhancement)
- GPU is not available (runs on CPU)

## Technical Details

### GAN Architecture

- **Latent Dimension**: 512
- **Network**: 4-layer MLP with LeakyReLU activations
- **Output**: Sigmoid-activated heightmap (0-1 range)

### Diffusion Integration

- **Model**: Stable Diffusion 1.5
- **ControlNet**: Depth map conditioning
- **Input Size**: 512x512 pixels
- **Inference Steps**: 20 (configurable)

### Terrain Features

- **Mountains**: Gaussian-based peak generation
- **Valleys**: Linear depression paths
- **Rivers**: Meandering water courses
- **Forests**: Noise-based roughness
- **Deserts**: Dune-like elevation patterns

## Future Enhancements

- [ ] Real-time terrain generation
- [ ] Multi-scale terrain generation
- [ ] Integration with game engines
- [ ] Support for more terrain types
- [ ] Batch processing capabilities
- [ ] API endpoint for web integration

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the prototype.

## Acknowledgments

- StyleGAN2 architecture inspiration
- Stable Diffusion and ControlNet models
- PyTorch and Hugging Face communities

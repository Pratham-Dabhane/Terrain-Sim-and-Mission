# Quick Start Guide - Generative AI Terrain Prototype

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
# Windows (PowerShell)
python install.py

# Or manually install core packages
pip install numpy matplotlib pillow scipy
```

### Step 2: Test the Installation
```bash
python test_terrain.py
```

### Step 3: Run the Prototype
```bash
# Interactive mode
python terrain_prototype.py

# Or run demos
python demo.py
```

## 🎯 What You'll Get

- **Text-to-Terrain Generation**: Type descriptions like "mountainous terrain with rivers"
- **Smart Feature Detection**: Automatically adds mountains, valleys, rivers based on your text
- **Multiple Views**: 2D heightmap, 2.5D colored terrain, and 3D visualization
- **Export Options**: Save high-resolution terrain images as PNG files

## 💡 Example Prompts

Try these terrain descriptions:
- "mountainous terrain with rivers and valleys"
- "desert landscape with sand dunes"
- "forest terrain with rolling hills"
- "coastal cliffs with rocky outcrops"
- "alpine landscape with snow-capped peaks"

## 🔧 Troubleshooting

**If you get import errors:**
```bash
pip install --upgrade pip
pip install numpy matplotlib pillow scipy
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

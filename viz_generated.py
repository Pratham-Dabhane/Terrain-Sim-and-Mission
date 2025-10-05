"""
Apply enhanced visualization to generated heightmap
"""

from enhanced_3d_viz import create_enhanced_3d_visualization
import numpy as np
from PIL import Image
import os

def visualize_generated_terrain():
    """Create enhanced 3D visualization of generated terrain"""
    
    heightmap_path = "Output/test_heightmap.png"
    
    if not os.path.exists(heightmap_path):
        print("❌ test_heightmap.png not found")
        return
    
    # Load the generated heightmap
    img = Image.open(heightmap_path).convert('L')
    heightmap = np.array(img) / 255.0
    
    print(f"📊 Loading heightmap: {heightmap.shape}, range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
    
    # Create enhanced visualization
    success = create_enhanced_3d_visualization(
        heightmap,
        "Output/enhanced_generated_terrain.png",
        "Generated Terrain - Enhanced 3D View"
    )
    
    if success:
        print("✅ Enhanced terrain visualization created: Output/enhanced_generated_terrain.png")
    else:
        print("❌ Failed to create enhanced visualization")

if __name__ == "__main__":
    visualize_generated_terrain()
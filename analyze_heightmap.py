"""
Quick visualization test for generated heightmap
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize_generated_heightmap():
    """Load and visualize the generated heightmap in multiple ways"""
    
    heightmap_path = "Output/test_heightmap.png"
    
    if not os.path.exists(heightmap_path):
        print("❌ test_heightmap.png not found")
        return
    
    # Load heightmap
    img = Image.open(heightmap_path).convert('L')
    heightmap = np.array(img) / 255.0
    
    print(f"📊 Heightmap stats:")
    print(f"   Shape: {heightmap.shape}")
    print(f"   Range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
    print(f"   Mean: {heightmap.mean():.3f}")
    print(f"   Std: {heightmap.std():.3f}")
    
    # Create multiple visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original heightmap
    axes[0,0].imshow(heightmap, cmap='gray')
    axes[0,0].set_title('Original Heightmap')
    axes[0,0].axis('off')
    
    # Terrain colormap
    im1 = axes[0,1].imshow(heightmap, cmap='terrain')
    axes[0,1].set_title('Terrain Colors')
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1])
    
    # Contour plot
    x = np.arange(heightmap.shape[1])
    y = np.arange(heightmap.shape[0])
    X, Y = np.meshgrid(x, y)
    contour = axes[0,2].contour(X, Y, heightmap, levels=15, colors='black', alpha=0.7)
    axes[0,2].contourf(X, Y, heightmap, levels=15, cmap='terrain', alpha=0.8)
    axes[0,2].set_title('Contour Map')
    
    # 3D surface (small subplot)
    ax3d = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Downsample for performance  
    step = max(1, heightmap.shape[0] // 50)
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    Z_sub = heightmap[::step, ::step]
    
    surf = ax3d.plot_surface(X_sub, Y_sub, Z_sub, 
                            cmap='terrain', alpha=0.8,
                            rcount=30, ccount=30)
    ax3d.set_title('3D Surface')
    ax3d.view_init(elev=30, azim=45)
    
    # Histogram of elevation values
    axes[1,1].hist(heightmap.flatten(), bins=50, alpha=0.7, color='brown')
    axes[1,1].set_title('Elevation Distribution')
    axes[1,1].set_xlabel('Elevation')
    axes[1,1].set_ylabel('Frequency')
    
    # Gradient magnitude
    gy, gx = np.gradient(heightmap)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    im2 = axes[1,2].imshow(gradient_mag, cmap='hot')
    axes[1,2].set_title('Slope Magnitude')
    axes[1,2].axis('off')
    plt.colorbar(im2, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('Output/heightmap_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Heightmap analysis saved: Output/heightmap_analysis.png")

if __name__ == "__main__":
    visualize_generated_heightmap()
"""
Enhanced 3D Terrain Visualization
Fixes rendering issues and improves visual quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

def create_enhanced_3d_visualization(heightmap, output_path="Output/enhanced_terrain_3d.png", title="Enhanced Terrain"):
    """
    Create enhanced 3D visualization with better lighting and colors.
    
    Args:
        heightmap: 2D numpy array of elevation data
        output_path: Path to save the visualization
        title: Title for the visualization
    """
    
    if HAS_PYVISTA:
        try:
            return create_pyvista_visualization(heightmap, output_path, title)
        except Exception as e:
            print(f"PyVista visualization failed: {e}, falling back to matplotlib...")
            return create_matplotlib_visualization(heightmap, output_path, title)
    else:
        print("PyVista not available, using matplotlib...")
        return create_matplotlib_visualization(heightmap, output_path, title)

def create_pyvista_visualization(heightmap, output_path, title):
    """Create PyVista 3D visualization with enhanced rendering"""
    
    height, width = heightmap.shape
    
    # Create coordinate grids
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height) 
    X, Y = np.meshgrid(x, y)
    
    # Scale elevation for better visualization
    z_scale = max(width, height) * 0.2
    Z = heightmap * z_scale
    
    # Create structured grid
    grid = pv.StructuredGrid(X, Y, Z)
    grid["elevation"] = heightmap.flatten(order='F')
    grid["normalized_elevation"] = ((heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())).flatten(order='F')
    
    # Create plotter with enhanced settings
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    
    # Set background
    plotter.background_color = 'lightblue'
    
    # Add mesh with enhanced rendering
    actor = plotter.add_mesh(
        grid,
        scalars='normalized_elevation',
        cmap='terrain',
        show_scalar_bar=True,
        scalar_bar_args={
            'title': 'Elevation',
            'n_labels': 5,
            'fmt': '%.2f'
        },
        lighting=True,
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
        ambient=0.3,
        diffuse=0.8,
        opacity=1.0,
        show_edges=False
    )
    
    # Enhanced lighting setup
    # Remove default lights
    plotter.remove_all_lights()
    
    # Add multiple lights for better illumination
    # Main light (sun)
    main_light = pv.Light(
        position=(width*2, height*2, z_scale*3),
        focal_point=(width/2, height/2, z_scale*0.3),
        color='white',
        intensity=0.7
    )
    plotter.add_light(main_light)
    
    # Fill light
    fill_light = pv.Light(
        position=(-width/2, -height/2, z_scale*2),
        focal_point=(width/2, height/2, z_scale*0.3),
        color='lightblue',
        intensity=0.3
    )
    plotter.add_light(fill_light)
    
    # Ambient light
    ambient_light = pv.Light(
        light_type='headlight',
        intensity=0.2
    )
    plotter.add_light(ambient_light)
    
    # Set camera position for optimal viewing
    plotter.camera_position = [
        (width*1.8, height*1.8, z_scale*1.5),  # Camera position
        (width/2, height/2, z_scale*0.2),      # Focal point
        (0, 0, 1)                             # Up vector
    ]
    
    # Add title
    plotter.add_title(title, font_size=16, color='black')
    
    # Enable depth peeling for better transparency
    plotter.enable_depth_peeling(number_of_peels=4)
    
    # Take screenshot
    plotter.screenshot(output_path, return_img=False)
    plotter.close()
    
    return True

def create_matplotlib_visualization(heightmap, output_path, title):
    """Create matplotlib 3D visualization as fallback"""
    
    height, width = heightmap.shape
    
    # Create coordinate grids
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    X, Y = np.meshgrid(x, y)
    
    # Scale elevation
    z_scale = max(width, height) * 0.15
    Z = heightmap * z_scale
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for performance
    step = max(1, width // 100)
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step] 
    Z_sub = Z[::step, ::step]
    
    # Create surface plot
    surf = ax.plot_surface(
        X_sub, Y_sub, Z_sub,
        cmap='terrain',
        alpha=0.9,
        linewidth=0,
        antialiased=True,
        rcount=100,
        ccount=100
    )
    
    # Add contour lines for depth perception
    contour = ax.contour(
        X_sub, Y_sub, Z_sub,
        levels=10,
        colors='darkgray',
        alpha=0.6,
        linewidths=0.5
    )
    
    # Set labels and title
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Elevation', fontsize=12)
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add color bar
    cbar = plt.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Elevation', fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True

def test_enhanced_visualization():
    """Test the enhanced visualization"""
    
    print("🎨 Testing Enhanced 3D Visualization...")
    
    # Create test heightmap
    size = 128
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # Create more interesting terrain
    heightmap = 0.4 * (np.sin(X/3) * np.cos(Y/3) + 1)
    heightmap += 0.3 * np.sin(X) * np.sin(Y) 
    heightmap += 0.2 * np.sin(X*2) * np.cos(Y*2)
    heightmap += 0.1 * np.random.random((size, size))
    
    # Normalize
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    
    # Create output directory
    os.makedirs("Output", exist_ok=True)
    
    # Test both methods
    success_pyvista = False
    success_matplotlib = False
    
    if HAS_PYVISTA:
        try:
            create_pyvista_visualization(
                heightmap, 
                "Output/enhanced_terrain_pyvista.png",
                "Enhanced PyVista Terrain Visualization"
            )
            print("✅ PyVista visualization created successfully")
            success_pyvista = True
        except Exception as e:
            print(f"❌ PyVista visualization failed: {e}")
    
    try:
        create_matplotlib_visualization(
            heightmap,
            "Output/enhanced_terrain_matplotlib.png", 
            "Enhanced Matplotlib Terrain Visualization"
        )
        print("✅ Matplotlib visualization created successfully")
        success_matplotlib = True
    except Exception as e:
        print(f"❌ Matplotlib visualization failed: {e}")
    
    return success_pyvista or success_matplotlib

if __name__ == "__main__":
    test_enhanced_visualization()
"""
Matplotlib-based 3D terrain viewer (fallback for PyVista issues)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

logger = logging.getLogger(__name__)


def show_interactive_terrain_matplotlib(heightmap, terrain_prompt="Terrain"):
    """
    Show interactive 3D terrain using matplotlib (more compatible than PyVista).
    
    Args:
        heightmap: 2D numpy array of elevation values
        terrain_prompt: Title for the window
    """
    logger.info(f"Opening matplotlib 3D viewer for: {terrain_prompt}")
    
    height, width = heightmap.shape
    
    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # Scale Z for better visualization
    Z = heightmap * 60.0  # Same scaling as PyVista version
    
    # Create figure
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with colormap
    surf = ax.plot_surface(X, Y, Z, 
                          cmap='gist_earth',  # Earth colors
                          linewidth=0.5,
                          antialiased=True,
                          edgecolors='darkblue',
                          alpha=0.9)
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Elevation (m)')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.set_title(f'Interactive Terrain: {terrain_prompt}\n(Click and drag to rotate, scroll to zoom)', 
                 fontsize=14, pad=20)
    
    # Set viewing angle
    ax.view_init(elev=30, azim=-60)
    
    # Set background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add instructions
    instructions = (
        "CONTROLS:\n"
        "• Click + Drag: Rotate view\n"
        "• Scroll: Zoom in/out\n"
        "• Right-click + Drag: Pan\n"
        "• Close window to exit"
    )
    fig.text(0.02, 0.98, instructions, fontsize=10, 
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    logger.info("✓ Matplotlib viewer opened - you can now interact with the terrain")
    
    # Show interactive window (blocking)
    plt.show()
    
    logger.info("✓ Matplotlib viewer closed")
    return True

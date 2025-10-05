"""
3D Mesh Visualization and Export using PyVista
Converts heightmaps to interactive 3D meshes with realistic rendering.
"""

import numpy as np
import pyvista as pv
from PIL import Image
import torch
from typing import Union, Optional, Tuple, Dict, Any
import logging
import os
from pathlib import Path

# Optional imports for enhanced functionality
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

logger = logging.getLogger(__name__)

class TerrainMeshGenerator:
    """
    Convert heightmaps to 3D meshes using various methods.
    
    Args:
        method: Mesh generation method ('structured_grid', 'marching_cubes', 'delaunay')
        scale_factor: Scaling factor for terrain elevation
        smoothing: Whether to apply mesh smoothing
    """
    
    def __init__(
        self,
        method: str = "structured_grid",
        scale_factor: float = 1.0,
        smoothing: bool = True
    ):
        self.method = method
        self.scale_factor = scale_factor
        self.smoothing = smoothing
        
        # Validate method
        valid_methods = ["structured_grid", "marching_cubes", "delaunay"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        if method == "marching_cubes" and not HAS_SKIMAGE:
            logger.warning("scikit-image not available. Using structured_grid method instead.")
            self.method = "structured_grid"
        
        logger.info(f"TerrainMeshGenerator initialized with method: {self.method}")
    
    def heightmap_to_structured_grid(
        self, 
        heightmap: np.ndarray,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 1.0
    ) -> pv.StructuredGrid:
        """
        Convert heightmap to PyVista StructuredGrid.
        
        Args:
            heightmap: 2D heightmap array (H, W)
            x_scale: X-axis scaling
            y_scale: Y-axis scaling  
            z_scale: Z-axis (elevation) scaling
            
        Returns:
            pv.StructuredGrid: Mesh as structured grid
        """
        if len(heightmap.shape) != 2:
            raise ValueError("Heightmap must be 2D array")
        
        height, width = heightmap.shape
        
        # Create coordinate grids
        x = np.linspace(0, width * x_scale, width)
        y = np.linspace(0, height * y_scale, height)
        X, Y = np.meshgrid(x, y)
        
        # Scale elevation
        Z = heightmap * z_scale * self.scale_factor
        
        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Add heightmap as scalar field
        grid["elevation"] = heightmap.flatten(order='F')
        grid["height_normalized"] = (heightmap / heightmap.max()).flatten(order='F')
        
        return grid
    
    def heightmap_to_marching_cubes(
        self,
        heightmap: np.ndarray,
        level: float = 0.5,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 1.0
    ) -> pv.PolyData:
        """
        Convert heightmap to mesh using marching cubes algorithm.
        
        Args:
            heightmap: 2D heightmap array
            level: Isosurface level
            x_scale: X-axis scaling
            y_scale: Y-axis scaling
            z_scale: Z-axis scaling
            
        Returns:
            pv.PolyData: Mesh as polydata
        """
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image required for marching cubes")
        
        # Create 3D volume by extruding heightmap
        height, width = heightmap.shape
        depth = int(max(height, width) * 0.1)  # 10% of max dimension
        
        volume = np.zeros((height, width, depth))
        for i in range(depth):
            threshold = i / depth
            volume[:, :, i] = (heightmap > threshold).astype(float)
        
        # Apply marching cubes
        try:
            vertices, faces, normals, values = measure.marching_cubes(
                volume, level=level, spacing=(x_scale, y_scale, z_scale)
            )
        except Exception as e:
            logger.warning(f"Marching cubes failed: {e}. Using structured grid instead.")
            return self.heightmap_to_structured_grid(heightmap, x_scale, y_scale, z_scale)
        
        # Create PyVista mesh
        mesh = pv.PolyData(vertices, faces.reshape(-1, 4)[:, 1:])
        mesh["normals"] = normals
        mesh["values"] = values
        
        return mesh
    
    def heightmap_to_delaunay(
        self,
        heightmap: np.ndarray,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 1.0,
        alpha: float = 0.0
    ) -> pv.PolyData:
        """
        Convert heightmap to mesh using Delaunay triangulation.
        
        Args:
            heightmap: 2D heightmap array
            x_scale: X-axis scaling
            y_scale: Y-axis scaling
            z_scale: Z-axis scaling
            alpha: Alpha parameter for Delaunay
            
        Returns:
            pv.PolyData: Triangulated mesh
        """
        height, width = heightmap.shape
        
        # Create point cloud
        x = np.linspace(0, width * x_scale, width)
        y = np.linspace(0, height * y_scale, height)
        X, Y = np.meshgrid(x, y)
        Z = heightmap * z_scale * self.scale_factor
        
        # Flatten to point cloud
        points = np.column_stack([
            X.flatten(),
            Y.flatten(),
            Z.flatten()
        ])
        
        # Create point cloud
        cloud = pv.PolyData(points)
        cloud["elevation"] = heightmap.flatten()
        
        # Delaunay triangulation
        mesh = cloud.delaunay_2d(alpha=alpha)
        
        return mesh
    
    def generate_mesh(
        self,
        heightmap: Union[np.ndarray, torch.Tensor, Image.Image],
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 1.0,
        **kwargs
    ) -> pv.DataSet:
        """
        Generate 3D mesh from heightmap using selected method.
        
        Args:
            heightmap: Input heightmap
            x_scale: X-axis scaling
            y_scale: Y-axis scaling
            z_scale: Z-axis scaling
            **kwargs: Additional method-specific parameters
            
        Returns:
            pv.DataSet: Generated mesh
        """
        # Convert input to numpy array
        if isinstance(heightmap, torch.Tensor):
            heightmap = heightmap.cpu().numpy()
        elif isinstance(heightmap, Image.Image):
            heightmap = np.array(heightmap.convert('L')) / 255.0
        
        # Ensure 2D
        if len(heightmap.shape) == 3:
            heightmap = heightmap.squeeze()
        if len(heightmap.shape) == 4:
            heightmap = heightmap[0, 0]
        
        # Normalize heightmap
        if heightmap.max() > 1.0:
            heightmap = heightmap / heightmap.max()
        
        # Generate mesh based on method
        if self.method == "structured_grid":
            mesh = self.heightmap_to_structured_grid(heightmap, x_scale, y_scale, z_scale)
        elif self.method == "marching_cubes":
            mesh = self.heightmap_to_marching_cubes(heightmap, x_scale, y_scale, z_scale, **kwargs)
        elif self.method == "delaunay":
            mesh = self.heightmap_to_delaunay(heightmap, x_scale, y_scale, z_scale, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Apply smoothing if requested
        if self.smoothing and hasattr(mesh, 'smooth'):
            try:
                mesh = mesh.smooth(n_iter=50, relaxation_factor=0.1)
            except:
                logger.warning("Mesh smoothing failed")
        
        return mesh


class TerrainVisualizer:
    """
    Interactive 3D visualization of terrain meshes using PyVista.
    """
    
    def __init__(
        self,
        window_size: Tuple[int, int] = (1024, 768),
        background_color: str = "white",
        lighting: bool = True,
        shadows: bool = False,  # Disabled for performance
        anti_aliasing: bool = True
    ):
        self.window_size = window_size
        self.background_color = background_color
        self.lighting = lighting
        self.shadows = shadows
        self.anti_aliasing = anti_aliasing
        
        # Default camera position
        self.camera_position = [(0.5, 0.5, 2.0), (0.5, 0.5, 0.0), (0, 0, 1)]
        
        logger.info("TerrainVisualizer initialized")
    
    def create_plotter(self, off_screen: bool = False) -> pv.Plotter:
        """
        Create a PyVista plotter with configured settings.
        
        Args:
            off_screen: Whether to render off-screen (for saving images)
            
        Returns:
            pv.Plotter: Configured plotter
        """
        plotter = pv.Plotter(
            window_size=self.window_size,
            off_screen=off_screen
        )
        
        # Configure plotter
        plotter.background_color = self.background_color
        
        if self.lighting:
            plotter.enable_depth_peeling()
        
        if self.shadows:
            plotter.enable_shadows()
        
        if self.anti_aliasing:
            plotter.enable_anti_aliasing()
        
        return plotter
    
    def apply_terrain_colormap(
        self,
        mesh: pv.DataSet,
        colormap: str = "terrain",
        elevation_field: str = "elevation"
    ) -> pv.DataSet:
        """
        Apply terrain-appropriate colormap to mesh.
        
        Args:
            mesh: Input mesh
            colormap: Colormap name
            elevation_field: Field name for elevation data
            
        Returns:
            pv.DataSet: Mesh with colormap applied
        """
        # Available terrain colormaps
        terrain_cmaps = {
            "terrain": "terrain",
            "earth": "gist_earth", 
            "topographic": "plasma",
            "height": "viridis",
            "realistic": "terrain",
            "ocean": "ocean"
        }
        
        if colormap in terrain_cmaps:
            colormap = terrain_cmaps[colormap]
        
        # Apply colormap if elevation data exists
        if elevation_field in mesh.array_names:
            mesh.set_active_scalars(elevation_field)
        
        return mesh
    
    def apply_realistic_texture(
        self,
        mesh: pv.DataSet,
        enhanced_image: Union[np.ndarray, Image.Image, str]
    ) -> pv.DataSet:
        """
        Apply realistic terrain texture from enhanced image to mesh.
        
        Args:
            mesh: Input mesh
            enhanced_image: Enhanced terrain image (RGB array, PIL Image, or file path)
            
        Returns:
            pv.DataSet: Mesh with realistic texture applied
        """
        # Load image if path provided
        if isinstance(enhanced_image, str):
            enhanced_image = Image.open(enhanced_image)
        
        # Convert PIL Image to numpy array
        if isinstance(enhanced_image, Image.Image):
            enhanced_image = np.array(enhanced_image)
        
        # Ensure RGB format
        if len(enhanced_image.shape) == 2:
            enhanced_image = np.stack([enhanced_image] * 3, axis=-1)
        elif enhanced_image.shape[2] == 4:  # RGBA
            enhanced_image = enhanced_image[:, :, :3]
        
        # Get mesh dimensions
        bounds = mesh.bounds
        x_range = (bounds[0], bounds[1])
        y_range = (bounds[2], bounds[3])
        
        # Get mesh points
        points = mesh.points
        
        # Map mesh coordinates to image coordinates
        img_height, img_width = enhanced_image.shape[:2]
        
        # Normalize coordinates to [0, 1]
        x_norm = (points[:, 0] - x_range[0]) / (x_range[1] - x_range[0])
        y_norm = (points[:, 1] - y_range[0]) / (y_range[1] - y_range[0])
        
        # Convert to image indices
        x_indices = np.clip((x_norm * (img_width - 1)).astype(int), 0, img_width - 1)
        y_indices = np.clip(((1 - y_norm) * (img_height - 1)).astype(int), 0, img_height - 1)
        
        # Sample colors from enhanced image
        colors = enhanced_image[y_indices, x_indices]
        
        # Normalize colors to [0, 1] for PyVista
        colors_normalized = colors.astype(np.float32) / 255.0
        
        # Add colors to mesh with proper naming
        mesh.point_data['RGB'] = colors_normalized
        mesh.set_active_scalars('RGB')
        
        logger.info(f"Applied realistic texture to mesh: {colors_normalized.shape}")
        
        return mesh
    
    def add_terrain_lighting(self, plotter: pv.Plotter):
        """Add realistic lighting for terrain visualization"""
        # Remove default lights first
        plotter.remove_all_lights()
        
        # Sun light (primary)
        sun_light = pv.Light(
            position=(2.0, 2.0, 3.0),
            focal_point=(0.5, 0.5, 0.0),
            color="white",
            intensity=0.7
        )
        plotter.add_light(sun_light)
        
        # Fill light (softer)
        fill_light = pv.Light(
            position=(-1.0, -1.0, 2.0),
            focal_point=(0.5, 0.5, 0.0),
            color="lightblue",
            intensity=0.3
        )
        plotter.add_light(fill_light)
        
        # Ambient light
        ambient_light = pv.Light(
            light_type='headlight',
            intensity=0.2
        )
        plotter.add_light(ambient_light)
        plotter.add_light(sun_light)
        
        # Ambient light
        ambient_light = pv.Light(
            light_type='headlight',
            intensity=0.3
        )
        plotter.add_light(ambient_light)
    
    def visualize_terrain(
        self,
        mesh: pv.DataSet,
        title: str = "Terrain Visualization",
        colormap: str = "terrain",
        show_edges: bool = False,
        show_wireframe: bool = False,
        show_scalar_bar: bool = True,
        camera_position: Optional[list] = None,
        save_path: Optional[str] = None,
        interactive: bool = True,
        enhanced_texture: Optional[Union[np.ndarray, Image.Image, str]] = None
    ) -> Optional[np.ndarray]:
        """
        Visualize terrain mesh interactively.
        
        Args:
            mesh: Terrain mesh to visualize
            title: Window title
            colormap: Colormap for elevation
            show_edges: Whether to show mesh edges
            show_wireframe: Whether to show wireframe
            show_scalar_bar: Whether to show color scale bar
            camera_position: Custom camera position
            save_path: Path to save screenshot
            interactive: Whether to show interactive window
            
        Returns:
            np.ndarray: Screenshot if save_path provided and not interactive
        """
        # Create plotter
        plotter = self.create_plotter(off_screen=not interactive)
        
        # Apply realistic texture if provided, otherwise use elevation colormap
        if enhanced_texture is not None:
            try:
                mesh = self.apply_realistic_texture(mesh, enhanced_texture)
                # Add mesh with realistic texture colors
                actor = plotter.add_mesh(
                    mesh,
                    scalars="RGB",  # Use the RGB texture colors
                    rgb=True,  # Important: tell PyVista these are RGB values
                    show_edges=show_edges,
                    show_scalar_bar=False,  # No scalar bar for RGB colors
                    lighting=self.lighting,
                    smooth_shading=True,
                    opacity=1.0
                )
            except Exception as e:
                print(f"Warning: Failed to apply realistic texture: {e}")
                print("Falling back to elevation-based coloring...")
                # Fall back to elevation coloring
                mesh = self.apply_terrain_colormap(mesh, colormap)
                if "elevation" not in mesh.array_names and mesh.n_points > 0:
                    z_coords = mesh.points[:, 2]
                    mesh["elevation"] = z_coords
                    mesh.set_active_scalars("elevation")
                
                actor = plotter.add_mesh(
                    mesh,
                    scalars="elevation" if "elevation" in mesh.array_names else None,
                    cmap=colormap,
                    show_edges=show_edges,
                    show_scalar_bar=show_scalar_bar,
                    lighting=self.lighting,
                    smooth_shading=True,
                    opacity=1.0
                )
        else:
            # Apply colormap and ensure elevation data is set
            mesh = self.apply_terrain_colormap(mesh, colormap)
            
            # Ensure we have elevation data for coloring
            if "elevation" not in mesh.array_names and mesh.n_points > 0:
                # Create elevation data from Z coordinates
                z_coords = mesh.points[:, 2]
                mesh["elevation"] = z_coords
                mesh.set_active_scalars("elevation")
            
            # Add mesh to scene with explicit scalar field
            actor = plotter.add_mesh(
                mesh,
                scalars="elevation" if "elevation" in mesh.array_names else None,
                cmap=colormap,
                show_edges=show_edges,
                show_scalar_bar=show_scalar_bar,
                lighting=self.lighting,
                smooth_shading=True,
                opacity=1.0
            )
        
        # Add wireframe if requested
        if show_wireframe:
            plotter.add_mesh(
                mesh,
                style='wireframe',
                color='black',
                line_width=1,
                opacity=0.3
            )
        
        # Configure lighting
        if self.lighting:
            self.add_terrain_lighting(plotter)
        
        # Set camera position with better defaults for terrain
        if camera_position:
            plotter.camera_position = camera_position
        else:
            # Better default camera position for elevated terrain
            mesh_bounds = mesh.bounds
            center_x = (mesh_bounds[0] + mesh_bounds[1]) / 2
            center_y = (mesh_bounds[2] + mesh_bounds[3]) / 2
            center_z = (mesh_bounds[4] + mesh_bounds[5]) / 2
            
            cam_distance = max(mesh_bounds[1] - mesh_bounds[0], mesh_bounds[3] - mesh_bounds[2]) * 1.5
            camera_pos = [
                (center_x + cam_distance, center_y + cam_distance, center_z + cam_distance * 0.5),
                (center_x, center_y, center_z),
                (0, 0, 1)
            ]
            plotter.camera_position = camera_pos
        
        # Set title
        plotter.add_title(title, font_size=16)
        
        # Add axes
        plotter.show_axes()
        
        # Show or save
        if interactive:
            plotter.show()
            return None
        else:
            screenshot = plotter.screenshot(save_path, return_img=True)
            plotter.close()
            return screenshot
    
    def create_comparison_view(
        self,
        meshes: list,
        titles: list,
        colormap: str = "terrain",
        save_path: Optional[str] = None
    ):
        """
        Create side-by-side comparison of multiple terrain meshes.
        
        Args:
            meshes: List of terrain meshes
            titles: List of titles for each mesh
            colormap: Colormap to use
            save_path: Path to save comparison image
        """
        n_meshes = len(meshes)
        if n_meshes != len(titles):
            raise ValueError("Number of meshes must match number of titles")
        
        # Create subplot plotter
        plotter = pv.Plotter(
            shape=(1, n_meshes),
            window_size=(self.window_size[0] * n_meshes, self.window_size[1])
        )
        
        for i, (mesh, title) in enumerate(zip(meshes, titles)):
            plotter.subplot(0, i)
            
            # Apply colormap
            mesh = self.apply_terrain_colormap(mesh, colormap)
            
            # Add mesh
            plotter.add_mesh(
                mesh,
                cmap=colormap,
                show_scalar_bar=True,
                lighting=self.lighting,
                smooth_shading=True
            )
            
            # Configure view
            plotter.camera_position = self.camera_position
            plotter.add_title(title, font_size=14)
            
            # Add lighting
            if self.lighting:
                self.add_terrain_lighting(plotter)
        
        # Show or save
        if save_path:
            plotter.screenshot(save_path)
        else:
            plotter.show()


class MeshExporter:
    """
    Export terrain meshes to various 3D file formats.
    """
    
    @staticmethod
    def export_mesh(
        mesh: pv.DataSet,
        file_path: str,
        format: Optional[str] = None
    ):
        """
        Export mesh to file.
        
        Args:
            mesh: Mesh to export
            file_path: Output file path
            format: File format (auto-detected from extension if None)
        """
        file_path = Path(file_path)
        
        # Auto-detect format from extension
        if format is None:
            format = file_path.suffix.lower()
        
        # Create output directory
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format in ['.ply', '.PLY']:
                mesh.save(str(file_path))
            elif format in ['.obj', '.OBJ']:
                mesh.save(str(file_path))
            elif format in ['.stl', '.STL']:
                mesh.save(str(file_path))
            elif format in ['.vtk', '.VTK']:
                mesh.save(str(file_path))
            elif format in ['.vtp', '.VTP']:
                mesh.save(str(file_path))
            else:
                # Try generic save
                mesh.save(str(file_path))
                
            logger.info(f"Mesh exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export mesh: {e}")
            raise
    
    @staticmethod
    def export_heightmap(
        heightmap: np.ndarray,
        file_path: str,
        format: str = "png"
    ):
        """
        Export heightmap as image.
        
        Args:
            heightmap: 2D heightmap array
            file_path: Output file path
            format: Image format
        """
        # Normalize to [0, 255]
        if heightmap.max() <= 1.0:
            heightmap_img = (heightmap * 255).astype(np.uint8)
        else:
            heightmap_img = heightmap.astype(np.uint8)
        
        # Save as image
        img = Image.fromarray(heightmap_img, mode='L')
        img.save(file_path, format=format.upper())
        
        logger.info(f"Heightmap exported to {file_path}")


def test_mesh_generation():
    """Test function for mesh generation and visualization"""
    try:
        # Create test heightmap
        size = 128
        x = np.linspace(0, 4*np.pi, size)
        y = np.linspace(0, 4*np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate terrain-like heightmap
        heightmap = 0.5 * (np.sin(X/3) * np.cos(Y/3) + 1)
        heightmap += 0.2 * np.sin(X) * np.sin(Y)
        heightmap += 0.1 * np.random.random((size, size))
        heightmap = np.clip(heightmap, 0, 1)
        
        print(f"Test heightmap shape: {heightmap.shape}")
        print(f"Height range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
        
        # Test mesh generation
        generator = TerrainMeshGenerator(method="structured_grid")
        mesh = generator.generate_mesh(
            heightmap,
            x_scale=1.0,
            y_scale=1.0,
            z_scale=0.5
        )
        
        print(f"Generated mesh: {mesh}")
        print(f"Mesh points: {mesh.n_points}")
        print(f"Mesh cells: {mesh.n_cells}")
        
        # Test visualization (off-screen)
        visualizer = TerrainVisualizer()
        screenshot = visualizer.visualize_terrain(
            mesh,
            title="Test Terrain",
            save_path="test_terrain_3d.png",
            interactive=False
        )
        
        if screenshot is not None:
            print("Screenshot saved successfully")
        
        # Test export
        exporter = MeshExporter()
        exporter.export_mesh(mesh, "test_terrain.ply")
        exporter.export_heightmap(heightmap, "test_heightmap.png")
        
        print("Mesh generation and visualization test passed!")
        
    except Exception as e:
        print(f"Mesh generation test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mesh_generation()
#!/usr/bin/env python3
"""
Simplified Photorealistic 3D Terrain Renderer
Focuses on core photorealistic features with better compatibility.
"""

import numpy as np
import pyvista as pv
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class AdvancedTerrainRenderer:
    def __init__(self):
        self.setup_advanced_rendering()
    
    def setup_advanced_rendering(self):
        """Configure PyVista for photorealistic rendering"""
        pv.set_plot_theme("document")  # Clean white background
        logger.info("Advanced terrain renderer initialized")
    
    def create_photorealistic_visualization(self, heightmap, enhanced_texture, terrain_prompt, output_path):
        """Create photorealistic 3D terrain"""
        
        logger.info("Creating photorealistic 3D visualization...")
        
        try:
            # 1. Create high-resolution mesh with enhanced scaling
            mesh = self._create_enhanced_mesh(heightmap)
            logger.info(f"Enhanced mesh created: {mesh.n_points} points")
            
            # 2. Apply realistic texture mapping
            mesh = self._apply_texture_mapping(mesh, enhanced_texture, heightmap)
            logger.info("Realistic texture mapping applied")
            
            # 3. Set up photorealistic rendering
            plotter = self._setup_advanced_plotter()
            
            # 4. Add terrain with enhanced materials
            self._add_photorealistic_terrain(plotter, mesh, terrain_prompt)
            
            # 5. Set cinematic camera
            self._set_cinematic_camera(plotter, mesh, heightmap)
            
            # 6. Render high-quality image
            plotter.screenshot(output_path, window_size=(1920, 1080), return_img=False)
            plotter.close()
            
            logger.info(f"✓ Photorealistic visualization saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create photorealistic visualization: {e}")
            raise
    
    def create_interactive_photorealistic_visualization(self, heightmap, enhanced_texture, terrain_prompt):
        """Create INTERACTIVE photorealistic 3D terrain with rotation, zoom, etc."""
        
        logger.info("Creating INTERACTIVE photorealistic 3D visualization...")
        
        try:
            # 1. Create high-resolution mesh with enhanced scaling
            mesh = self._create_enhanced_mesh(heightmap)
            logger.info(f"Enhanced mesh created: {mesh.n_points} points")
            
            # 2. Apply realistic texture mapping
            mesh = self._apply_texture_mapping(mesh, enhanced_texture, heightmap)
            logger.info("Realistic texture mapping applied")
            
            # 3. Set up INTERACTIVE photorealistic plotter
            plotter = self._setup_interactive_plotter(terrain_prompt)
            
            # 4. Add terrain with enhanced materials
            self._add_photorealistic_terrain(plotter, mesh, terrain_prompt)
            
            # 5. Set initial camera position
            self._set_cinematic_camera(plotter, mesh, heightmap)
            
            # 6. Add interactive controls and UI
            self._add_interactive_controls(plotter, mesh, terrain_prompt)
            
            # 7. Show interactive window
            logger.info("🎮 Opening interactive 3D viewer... (close window when done)")
            plotter.show()
            
            logger.info("✓ Interactive visualization session completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create interactive visualization: {e}")
            raise
    
    def _create_enhanced_mesh(self, heightmap):
        """Create enhanced mesh with dramatic scaling"""
        height, width = heightmap.shape
        
        # Create coordinate grids
        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        
        # Enhanced height scaling for dramatic terrain (increased from 20 to 60)
        Z = heightmap * 60.0
        
        # Create points array
        points = np.column_stack([
            X.ravel(),
            Y.ravel(), 
            Z.ravel()
        ])
        
        # Create faces for structured grid
        faces = []
        for j in range(height - 1):
            for i in range(width - 1):
                # Two triangles per quad
                p1 = j * width + i
                p2 = j * width + (i + 1)
                p3 = (j + 1) * width + i
                p4 = (j + 1) * width + (i + 1)
                
                faces.extend([3, p1, p2, p3])  # First triangle
                faces.extend([3, p2, p4, p3])  # Second triangle
        
        # Create PolyData mesh
        mesh = pv.PolyData(points, faces)
        
        return mesh
    
    def _apply_texture_mapping(self, mesh, enhanced_texture, heightmap):
        """Apply realistic texture colors to mesh vertices"""
        
        # Convert texture to numpy array
        if isinstance(enhanced_texture, Image.Image):
            texture_array = np.array(enhanced_texture)
        else:
            texture_array = enhanced_texture
        
        # Get mesh points
        points = mesh.points
        height, width = heightmap.shape
        
        # Map 3D coordinates back to texture coordinates
        x_coords = points[:, 0]
        y_coords = points[:, 1] 
        z_coords = points[:, 2]
        
        # Normalize coordinates to [0, 1]
        x_norm = x_coords / (width - 1)
        y_norm = y_coords / (height - 1)
        
        # Convert to texture indices
        tex_height, tex_width = texture_array.shape[:2]
        x_indices = np.clip((x_norm * (tex_width - 1)).astype(int), 0, tex_width - 1)
        y_indices = np.clip(((1 - y_norm) * (tex_height - 1)).astype(int), 0, tex_height - 1)
        
        # Sample colors from texture
        colors = texture_array[y_indices, x_indices]
        
        # Enhanced coloring based on elevation
        colors_enhanced = self._enhance_colors_by_elevation(colors, z_coords, heightmap)
        
        # Normalize to [0, 1] for PyVista
        colors_normalized = colors_enhanced.astype(np.float32) / 255.0
        
        # Apply to mesh
        mesh.point_data['RGB'] = colors_normalized
        mesh.set_active_scalars('RGB')
        
        return mesh
    
    def _enhance_colors_by_elevation(self, colors, elevations, heightmap):
        """Enhance colors based on elevation for more realism"""
        enhanced = colors.copy()
        
        # Calculate elevation percentiles
        z_min, z_max = elevations.min(), elevations.max()
        elevation_normalized = (elevations - z_min) / (z_max - z_min)
        
        # Add atmospheric perspective (distant objects are more blue/hazy)
        for i in range(len(enhanced)):
            elev = elevation_normalized[i]
            
            # Higher elevations get slightly more atmospheric haze
            if elev > 0.7:  # High areas
                haze_factor = 0.1 * (elev - 0.7) / 0.3  # 0 to 0.1
                enhanced[i] = enhanced[i] * (1 - haze_factor) + np.array([200, 220, 255]) * haze_factor
                
            # Lower elevations get slightly darker (shadows)
            elif elev < 0.3:  # Low areas  
                shadow_factor = 0.2 * (0.3 - elev) / 0.3  # 0 to 0.2
                enhanced[i] = enhanced[i] * (1 - shadow_factor)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _enhance_forest_texture(self, texture, mesh):
        """Add forest-specific enhancements"""
        # Get elevation data for variation
        points = mesh.points
        elevations = points[:, 2].reshape(texture.shape[:2])
        
        # Enhance green variations based on elevation
        enhanced = texture.copy()
        
        # Add darker greens in valleys (lower elevation)
        low_areas = elevations < np.percentile(elevations, 30)
        enhanced[low_areas] = enhanced[low_areas] * 0.8  # Darker forest
        enhanced[low_areas, 1] = np.minimum(enhanced[low_areas, 1] * 1.2, 255)  # More green
        
        # Add lighter areas on peaks (clearings)
        high_areas = elevations > np.percentile(elevations, 80)
        enhanced[high_areas] = enhanced[high_areas] * 1.1  # Lighter areas
        
        return enhanced
    
    def _enhance_desert_texture(self, texture, mesh):
        """Add desert-specific enhancements"""
        points = mesh.points
        elevations = points[:, 2].reshape(texture.shape[:2])
        
        enhanced = texture.copy()
        
        # Add shadow effects in valleys
        low_areas = elevations < np.percentile(elevations, 40)
        enhanced[low_areas] = enhanced[low_areas] * 0.7  # Darker valleys
        
        # Bright sand on peaks
        high_areas = elevations > np.percentile(elevations, 70)
        enhanced[high_areas] = np.minimum(enhanced[high_areas] * 1.3, 255)
        
        return enhanced
    
    def _enhance_mountain_texture(self, texture, mesh):
        """Add mountain-specific enhancements"""
        points = mesh.points
        elevations = points[:, 2].reshape(texture.shape[:2])
        
        enhanced = texture.copy()
        
        # Snow on highest peaks
        snow_areas = elevations > np.percentile(elevations, 85)
        enhanced[snow_areas] = [240, 248, 255]  # Snow white
        
        # Rocky areas on slopes
        mid_areas = (elevations > np.percentile(elevations, 40)) & (elevations < np.percentile(elevations, 85))
        enhanced[mid_areas, :] = enhanced[mid_areas, :] * 0.8  # Darker rocky areas
        enhanced[mid_areas, 2] = np.minimum(enhanced[mid_areas, 2] + 20, 255)  # Slightly more blue for rock
        
        return enhanced
    
    def _setup_photorealistic_lighting(self):
        """Set up advanced lighting like reference image"""
        plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
        
        # Multiple light sources for photorealistic lighting
        try:
            # Remove default lighting
            plotter.remove_all_lights()
            
            # Main sun light (directional) - primary illumination
            sun_light = pv.Light(
                position=(200, 200, 300),
                focal_point=(0, 0, 0),
                color='white',
                intensity=0.9
            )
            plotter.add_light(sun_light)
            
            # Sky light (ambient) - soft fill lighting
            sky_light = pv.Light(
                position=(0, 0, 400),
                color='lightblue',
                intensity=0.4
            )
            plotter.add_light(sky_light)
            
            # Fill light (soften shadows) - reduce harsh shadows
            fill_light = pv.Light(
                position=(-100, -100, 200),
                focal_point=(0, 0, 0),
                color='white',
                intensity=0.3
            )
            plotter.add_light(fill_light)
            
        except Exception as e:
            logger.warning(f"Advanced lighting setup failed, using defaults: {e}")
            # Keep default lighting if advanced setup fails
        
        return plotter
    
    def _add_terrain_with_materials(self, plotter, mesh, terrain_prompt):
        """Add terrain with advanced material properties"""
        
        # Determine material properties based on terrain type
        if any(word in terrain_prompt.lower() for word in ['forest', 'green']):
            material_props = {
                'ambient': 0.3,
                'diffuse': 0.8,
                'specular': 0.1,
                'specular_power': 10,
                'roughness': 0.8
            }
        elif any(word in terrain_prompt.lower() for word in ['desert', 'sand']):
            material_props = {
                'ambient': 0.4,
                'diffuse': 0.9,
                'specular': 0.2,
                'specular_power': 20,
                'roughness': 0.6
            }
        elif any(word in terrain_prompt.lower() for word in ['snow', 'mountain']):
            material_props = {
                'ambient': 0.5,
                'diffuse': 0.8,
                'specular': 0.4,
                'specular_power': 50,
                'roughness': 0.3
            }
        else:
            material_props = {
                'ambient': 0.3,
                'diffuse': 0.8,
                'specular': 0.2,
                'specular_power': 20,
                'roughness': 0.5
            }
        
        # Add mesh with enhanced material properties
        mesh_actor = plotter.add_mesh(
            mesh,
            rgb=True,
            smooth_shading=True,
            show_edges=False,
            ambient=material_props['ambient'],
            diffuse=material_props['diffuse'],
            specular=material_props['specular'],
            specular_power=material_props['specular_power']
        )
        
        return mesh_actor
    
    def _set_cinematic_camera(self, plotter, mesh):
        """Position camera for dramatic cinematic view like reference"""
        bounds = mesh.bounds
        
        # Calculate optimal camera position
        x_center = (bounds[0] + bounds[1]) / 2
        y_center = (bounds[2] + bounds[3]) / 2
        z_max = bounds[5]
        
        # Cinematic angle (slightly elevated, angled view)
        camera_distance = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 1.2
        
        camera_pos = (
            x_center + camera_distance * 0.8,  # Side view
            y_center - camera_distance * 0.6,  # Slightly back
            z_max + camera_distance * 0.4      # Elevated
        )
        
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = (x_center, y_center, z_max * 0.3)
        plotter.camera.up = (0, 0, 1)
        
        # Set field of view for dramatic perspective
        plotter.camera.view_angle = 45
    
    def _setup_advanced_plotter(self):
        """Set up plotter with advanced rendering settings"""
        plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
        
        # Advanced lighting setup
        try:
            # Clear default lights
            plotter.remove_all_lights()
            
            # Primary directional light (sun)
            sun_light = pv.Light(
                position=(500, 500, 800),
                focal_point=(0, 0, 0),
                color=[1.0, 0.95, 0.8],  # Warm sunlight
                intensity=0.8
            )
            plotter.add_light(sun_light)
            
            # Ambient sky light
            sky_light = pv.Light(
                position=(0, 0, 1000),
                color=[0.7, 0.8, 1.0],  # Blue sky light
                intensity=0.4
            )
            plotter.add_light(sky_light)
            
            # Fill light to soften shadows
            fill_light = pv.Light(
                position=(-300, -300, 400),
                focal_point=(0, 0, 0),
                color=[1.0, 1.0, 1.0],
                intensity=0.2
            )
            plotter.add_light(fill_light)
            
        except Exception as e:
            logger.warning(f"Advanced lighting failed, using default: {e}")
        
        return plotter
    
    def _add_photorealistic_terrain(self, plotter, mesh, terrain_prompt):
        """Add terrain with photorealistic material properties"""
        
        # Material properties based on terrain type
        if any(word in terrain_prompt.lower() for word in ['forest', 'green', 'tree', 'valley']):
            # Forest: matte, organic surface
            material_props = {
                'ambient': 0.3,
                'diffuse': 0.8, 
                'specular': 0.05,
                'specular_power': 5
            }
        elif any(word in terrain_prompt.lower() for word in ['desert', 'sand', 'dune', 'oasis']):
            # Desert: slightly reflective sand
            material_props = {
                'ambient': 0.4,
                'diffuse': 0.9,
                'specular': 0.15,
                'specular_power': 15
            }
        elif any(word in terrain_prompt.lower() for word in ['mountain', 'snow', 'peak', 'glacier', 'rock']):
            # Mountain: mixed rock and snow reflectivity
            material_props = {
                'ambient': 0.5,
                'diffuse': 0.7,
                'specular': 0.3,
                'specular_power': 40
            }
        else:
            # Default terrain
            material_props = {
                'ambient': 0.3,
                'diffuse': 0.8,
                'specular': 0.1,
                'specular_power': 10
            }
        
        # Add mesh with enhanced properties
        actor = plotter.add_mesh(
            mesh,
            rgb=True,  # Use RGB colors from texture
            smooth_shading=True,
            show_edges=False,
            **material_props
        )
        
        return actor
    
    def _set_cinematic_camera(self, plotter, mesh, heightmap):
        """Set dramatic camera angle like reference image"""
        bounds = mesh.bounds
        
        # Calculate scene center and size
        x_center = (bounds[0] + bounds[1]) / 2
        y_center = (bounds[2] + bounds[3]) / 2
        z_center = (bounds[4] + bounds[5]) / 2
        z_max = bounds[5]
        
        # Scene dimensions
        x_size = bounds[1] - bounds[0]
        y_size = bounds[3] - bounds[2]
        z_size = bounds[5] - bounds[4]
        max_size = max(x_size, y_size)
        
        # FIXED: Much more elevated camera position for full 3D view
        camera_distance = max_size * 1.2
        camera_height = z_max + max_size * 0.8  # Much higher elevation
        
        camera_position = [
            x_center + camera_distance * 0.6,  # Side offset
            y_center - camera_distance * 0.4,  # Back offset  
            camera_height                       # High elevation
        ]
        
        # Focus on terrain center at ground level
        focus_point = [x_center, y_center, z_center]
        
        # Set camera with better angle
        plotter.camera.position = camera_position
        plotter.camera.focal_point = focus_point
        plotter.camera.up = [0, 0, 1]
        plotter.camera.view_angle = 45  # Wider angle to see full terrain
    
    def _setup_interactive_plotter(self, terrain_prompt):
        """Set up interactive plotter with enhanced controls"""
        # Create interactive plotter (not off_screen!)
        plotter = pv.Plotter(window_size=(1400, 900))
        
        # Set window title
        plotter.add_title(f"Interactive Photorealistic Terrain: {terrain_prompt}", font_size=16)
        
        # Advanced lighting setup (same as static version)
        try:
            # Clear default lights
            plotter.remove_all_lights()
            
            # Primary directional light (sun)
            sun_light = pv.Light(
                position=(500, 500, 800),
                focal_point=(0, 0, 0),
                color=[1.0, 0.95, 0.8],  # Warm sunlight
                intensity=0.8
            )
            plotter.add_light(sun_light)
            
            # Ambient sky light
            sky_light = pv.Light(
                position=(0, 0, 1000),
                color=[0.7, 0.8, 1.0],  # Blue sky light
                intensity=0.4
            )
            plotter.add_light(sky_light)
            
            # Fill light to soften shadows
            fill_light = pv.Light(
                position=(-300, -300, 400),
                focal_point=(0, 0, 0),
                color=[1.0, 1.0, 1.0],
                intensity=0.2
            )
            plotter.add_light(fill_light)
            
        except Exception as e:
            logger.warning(f"Advanced lighting failed, using default: {e}")
        
        return plotter
    
    def _add_interactive_controls(self, plotter, mesh, terrain_prompt):
        """Add interactive controls and UI elements"""
        
        # Add control instructions as text
        controls_text = (
            "INTERACTIVE CONTROLS:\\n"
            "• Mouse: Rotate view\\n"  
            "• Scroll: Zoom in/out\\n"
            "• Right-click + drag: Pan\\n"
            "• 'r': Reset camera\\n"
            "• 'q' or ESC: Quit"
        )
        
        plotter.add_text(
            controls_text,
            position='upper_left',
            font_size=12,
            color='white',
            shadow=True
        )
        
        # Add terrain info
        bounds = mesh.bounds
        z_range = bounds[5] - bounds[4]
        info_text = (
            f"TERRAIN INFO:\\n"
            f"Prompt: {terrain_prompt[:40]}...\\n"
            f"Points: {mesh.n_points:,}\\n"
            f"Height Range: {z_range:.1f} units\\n"
            f"Resolution: High (60x scaling)"
        )
        
        plotter.add_text(
            info_text,
            position='upper_right',
            font_size=10,
            color='lightgray',
            shadow=True
        )
        
        # Add coordinate axes for reference
        plotter.show_axes()
        
        return plotter
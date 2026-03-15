#!/usr/bin/env python3
"""
Simplified Photorealistic 3D Terrain Renderer
Focuses on core photorealistic features with better compatibility.
"""

import os
import numpy as np
import pyvista as pv
from PIL import Image
import logging

# Enable GPU rendering for NVIDIA GeForce GTX 1650
os.environ['PYVISTA_USE_PANEL'] = '0'
os.environ['VTK_USE_GPU_RENDERING'] = '1'
# Force OpenGL version for NVIDIA GPU
os.environ['__GL_SYNC_TO_VBLANK'] = '1'
os.environ['DISPLAY'] = ':0'  # Ensure primary display

logger = logging.getLogger(__name__)

class AdvancedTerrainRenderer:
    def __init__(self):
        self.setup_advanced_rendering()
    
    def setup_advanced_rendering(self):
        """Configure PyVista for GPU-accelerated photorealistic rendering"""
        pv.set_plot_theme("document")
        
        # Enable GPU features globally
        pv.global_theme.multi_samples = 8  # GPU anti-aliasing
        pv.global_theme.smooth_shading = True  # GPU smooth shading
        
        logger.info("AdvancedTerrainRenderer initialized with GPU acceleration")
        logger.info("Advanced terrain renderer initialized")
    
    def create_photorealistic_visualization(self, heightmap, enhanced_texture, terrain_prompt, output_path):
        """Create photorealistic 3D terrain"""
        
        logger.info("Creating photorealistic 3D visualization...")
        
        try:
            # 1. Build separate meshes: top relief + side/bottom block
            top_mesh, block_mesh = self._create_render_meshes(heightmap)
            logger.info(
                f"Render meshes created: top={top_mesh.n_points} points, "
                f"block={block_mesh.n_points} points"
            )
            
            # 2. Apply realistic texture mapping
            top_mesh = self._apply_texture_mapping(top_mesh, enhanced_texture, heightmap)
            logger.info("Top-surface texture mapping applied")
            
            # 3. Set up photorealistic rendering
            plotter = self._setup_advanced_plotter()
            
            # 4. Add block first, then top relief mesh
            self._add_block_mesh(plotter, block_mesh)
            self._add_photorealistic_terrain(plotter, top_mesh, terrain_prompt)
            
            # 5. Set cinematic camera
            scene_mesh = top_mesh.merge(block_mesh)
            self._set_cinematic_camera(plotter, scene_mesh, heightmap)
            
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
        logger.info(f"Heightmap shape: {heightmap.shape}, range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
        
        try:
            # Build separate meshes: top relief + side/bottom block
            top_mesh, block_mesh = self._create_render_meshes(heightmap)
            logger.info(
                f"Render meshes created: top={top_mesh.n_points} points, "
                f"block={block_mesh.n_points} points"
            )
            
            # Apply realistic texture mapping (same as static visualization)
            top_mesh = self._apply_texture_mapping(top_mesh, enhanced_texture, heightmap)
            logger.info("Top-surface texture mapping applied")
            
            # Set up plotter
            plotter = self._setup_interactive_plotter(terrain_prompt)
            
            # Add terrain with photorealistic materials
            self._add_block_mesh(plotter, block_mesh)
            self._add_photorealistic_terrain(plotter, top_mesh, terrain_prompt)
            logger.info(
                f"✓ Meshes added with photorealistic materials: "
                f"top={top_mesh.n_points} points, block={block_mesh.n_points} points"
            )
            
            # Camera setup
            scene_mesh = top_mesh.merge(block_mesh)
            self._set_cinematic_camera(plotter, scene_mesh, heightmap)
            cam = plotter.camera
            logger.info(f"  Camera position: {cam.position}")
            logger.info(f"  Camera focal point: {cam.focal_point}")
            
            # Show interactive window
            logger.info("[3D VIEWER] Opening interactive 3D viewer...")
            logger.info("Controls: Mouse drag=rotate, Scroll=zoom, Right-drag=pan, 'r'=reset, 'q'=quit")
            
            plotter.show()
            
            logger.info("✓ Interactive visualization session completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create interactive visualization: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            raise
    
    def _smooth_heightmap_for_render(self, heightmap, sigma=0.8):
        """Apply gentle smoothing only for rendering to reduce spike artifacts."""
        try:
            from scipy import ndimage
            smoothed = ndimage.gaussian_filter(heightmap.astype(np.float32), sigma=sigma)
            return smoothed
        except Exception:
            # Fallback keeps existing behavior if scipy is unavailable at runtime.
            return heightmap.astype(np.float32)

    def _create_render_meshes(self, heightmap):
        """Create separate meshes for top relief and side/bottom block."""
        height, width = heightmap.shape

        # Gentle render-time smoothing to remove needle-like spikes.
        smoothed_heightmap = self._smooth_heightmap_for_render(heightmap, sigma=0.8)

        # Keep moderate vertical exaggeration.
        base_size = max(height, width)
        vertical_exaggeration = 0.6
        z_top = smoothed_heightmap * base_size * vertical_exaggeration

        # Flat bottom plane depth below the minimum top elevation.
        top_min = float(z_top.min())
        top_range = float(z_top.max() - z_top.min())
        base_depth = max(base_size * 0.18, top_range * 0.35)
        z_bottom = top_min - base_depth

        # Build top vertices and bottom vertices.
        x = np.arange(width, dtype=np.float32)
        # Flip Y so that heightmap row-0 (image top) maps to high Y in 3D,
        # keeping the mesh orientation consistent with the 2D heightmap.
        y = np.arange(height, dtype=np.float32)[::-1]
        X, Y = np.meshgrid(x, y)

        top_points = np.column_stack([X.ravel(), Y.ravel(), z_top.ravel()])
        bottom_points = np.column_stack([
            X.ravel(),
            Y.ravel(),
            np.full(height * width, z_bottom, dtype=np.float32),
        ])
        # Top relief mesh (surface only).
        top_faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                t1 = i * width + j
                t2 = i * width + (j + 1)
                t3 = (i + 1) * width + j
                t4 = (i + 1) * width + (j + 1)
                top_faces.extend([3, t1, t2, t3])
                top_faces.extend([3, t2, t4, t3])

        top_mesh = pv.PolyData(top_points, np.array(top_faces, dtype=np.int64))
        top_mesh = top_mesh.subdivide(nsub=1, subfilter='butterfly')
        top_mesh = top_mesh.smooth(n_iter=25, relaxation_factor=0.1)
        top_mesh = top_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=False)

        # Block mesh (side walls + flat bottom only, no top faces to avoid z-fighting).
        points = np.vstack([top_points, bottom_points]).astype(np.float32)

        n_top = height * width

        def tidx(i, j):
            return i * width + j

        def bidx(i, j):
            return n_top + i * width + j

        faces = []

        # Bottom surface only.
        for i in range(height - 1):
            for j in range(width - 1):
                t1 = tidx(i, j)
                t2 = tidx(i, j + 1)
                t3 = tidx(i + 1, j)
                t4 = tidx(i + 1, j + 1)

                b1 = bidx(i, j)
                b2 = bidx(i, j + 1)
                b3 = bidx(i + 1, j)
                b4 = bidx(i + 1, j + 1)

                # Reverse winding for bottom face.
                faces.extend([3, b1, b3, b2])
                faces.extend([3, b2, b3, b4])

        # North and south walls.
        for j in range(width - 1):
            nt1, nt2 = tidx(0, j), tidx(0, j + 1)
            nb1, nb2 = bidx(0, j), bidx(0, j + 1)
            faces.extend([3, nt1, nb1, nt2])
            faces.extend([3, nt2, nb1, nb2])

            st1, st2 = tidx(height - 1, j), tidx(height - 1, j + 1)
            sb1, sb2 = bidx(height - 1, j), bidx(height - 1, j + 1)
            faces.extend([3, st1, st2, sb1])
            faces.extend([3, st2, sb2, sb1])

        # West and east walls.
        for i in range(height - 1):
            wt1, wt2 = tidx(i, 0), tidx(i + 1, 0)
            wb1, wb2 = bidx(i, 0), bidx(i + 1, 0)
            faces.extend([3, wt1, wt2, wb1])
            faces.extend([3, wt2, wb2, wb1])

            et1, et2 = tidx(i, width - 1), tidx(i + 1, width - 1)
            eb1, eb2 = bidx(i, width - 1), bidx(i + 1, width - 1)
            faces.extend([3, et1, eb1, et2])
            faces.extend([3, et2, eb1, eb2])

        block_mesh = pv.PolyData(points, np.array(faces, dtype=np.int64))
        block_mesh = block_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=False)

        logger.info(
            f"Solid terrain block created: {block_mesh.n_points} points, {block_mesh.n_cells} cells, "
            f"base_depth={base_depth:.2f}"
        )

        return top_mesh, block_mesh

    def _create_enhanced_mesh(self, heightmap):
        """Backward-compatible helper returning merged top+block mesh."""
        top_mesh, block_mesh = self._create_render_meshes(heightmap)
        return top_mesh.merge(block_mesh)
    
    def _apply_texture_mapping(self, mesh, enhanced_texture, heightmap):
        """Apply realistic texture colors to mesh vertices"""
        
        # Convert texture to numpy array
        if isinstance(enhanced_texture, Image.Image):
            texture_array = np.array(enhanced_texture)
        else:
            texture_array = np.asarray(enhanced_texture)

        # Normalize to HxWx3 to avoid shape-related indexing blowups.
        if texture_array.ndim == 4:
            # Common case from some remaster pipelines: (1, H, W, C)
            texture_array = texture_array[0]
        if texture_array.ndim == 2:
            texture_array = np.stack([texture_array] * 3, axis=-1)
        if texture_array.ndim == 3 and texture_array.shape[2] == 1:
            texture_array = np.repeat(texture_array, 3, axis=2)
        if texture_array.ndim != 3 or texture_array.shape[2] < 3:
            raise ValueError(f"Unexpected texture shape for mapping: {texture_array.shape}")
        
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

    def _add_block_mesh(self, plotter, block_mesh):
        """Add side/bottom block with matte earth tone."""
        return plotter.add_mesh(
            block_mesh,
            color=(0.38, 0.28, 0.21),
            smooth_shading=True,
            show_edges=False,
            ambient=0.14,
            diffuse=0.62,
            specular=0.0,
            specular_power=1,
        )
    
    def _enhance_colors_by_elevation(self, colors, elevations, heightmap):
        """Enhance colors with realistic muted earth tones"""
        enhanced = colors.copy().astype(np.float32)
        
        # Calculate elevation percentiles
        z_min, z_max = elevations.min(), elevations.max()
        elevation_normalized = (elevations - z_min) / (z_max - z_min + 1e-6)
        
        # Apply realistic color adjustments
        for i in range(len(enhanced)):
            elev = elevation_normalized[i]
            
            # Low valleys - slightly darker, more saturated greens/browns
            if elev < 0.2:
                enhanced[i] *= 0.85
                # Boost earth tones
                if enhanced[i][1] > enhanced[i][0]:  # If greenish
                    enhanced[i][1] = min(255, enhanced[i][1] * 0.95)
            
            # Mid elevations - keep natural but slightly desaturate
            elif elev < 0.5:
                gray = enhanced[i].mean()
                enhanced[i] = enhanced[i] * 0.95 + gray * 0.05
            
            # Upper mid - more desaturation, earthy browns
            elif elev < 0.7:
                gray = enhanced[i].mean()
                enhanced[i] = enhanced[i] * 0.85 + gray * 0.15
                # Push toward browns/grays
                enhanced[i][0] = min(255, enhanced[i][0] * 1.05)  # More red
                enhanced[i][2] = max(0, enhanced[i][2] * 0.95)    # Less blue
            
            # High areas - rocky grays, not bright
            elif elev < 0.85:
                gray = enhanced[i].mean()
                # Transition to gray rock colors (120-160 range)
                rock_gray = np.clip(gray * 0.7 + 90, 100, 150)
                t = (elev - 0.7) / 0.15
                enhanced[i] = enhanced[i] * (1 - t) + rock_gray * t
            
            # Very high peaks - realistic snow (off-white, not pure white)
            else:
                t = (elev - 0.85) / 0.15
                # Snow color: slightly warm off-white (not blue-ish)
                snow_color = np.array([235, 235, 230], dtype=np.float32)
                enhanced[i] = enhanced[i] * (1 - t*0.8) + snow_color * (t*0.8)
        
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
                intensity=0.65
            )
            plotter.add_light(sun_light)
            
            # Ambient sky light
            sky_light = pv.Light(
                position=(0, 0, 1000),
                color=[0.7, 0.8, 1.0],  # Blue sky light
                intensity=0.55
            )
            plotter.add_light(sky_light)
            
            # Fill light to soften shadows
            fill_light = pv.Light(
                position=(-300, -300, 400),
                focal_point=(0, 0, 0),
                color=[1.0, 1.0, 1.0],
                intensity=0.35
            )
            plotter.add_light(fill_light)
            
        except Exception as e:
            logger.warning(f"Advanced lighting failed, using default: {e}")
        
        return plotter
    
    def _add_photorealistic_terrain(self, plotter, mesh, terrain_prompt):
        """Add terrain with realistic matte materials (no plastic/wax look)"""
        
        prompt_lower = terrain_prompt.lower()
        
        # Most terrain is MATTE - very low specular!
        if any(word in prompt_lower for word in ['forest', 'green', 'tree', 'jungle', 'vegetation']):
            material_props = {
                'ambient': 0.42,
                'diffuse': 0.98,
                'specular': 0.0,  # No shine on vegetation
                'specular_power': 1
            }
        elif any(word in prompt_lower for word in ['desert', 'sand', 'dune', 'sahara', 'arid']):
            material_props = {
                'ambient': 0.46,
                'diffuse': 1.00,
                'specular': 0.02,  # Barely any shine on sand
                'specular_power': 5
            }
        elif any(word in prompt_lower for word in ['snow', 'ice', 'glacier', 'arctic', 'frozen']):
            # Snow is NOT shiny like plastic - it's matte!
            material_props = {
                'ambient': 0.52,
                'diffuse': 1.00,
                'specular': 0.05,  # Very subtle
                'specular_power': 10
            }
        elif any(word in prompt_lower for word in ['mountain', 'rock', 'rocky', 'cliff', 'crag']):
            material_props = {
                'ambient': 0.38,
                'diffuse': 0.95,
                'specular': 0.03,  # Rock is mostly matte
                'specular_power': 8
            }
        elif any(word in prompt_lower for word in ['volcanic', 'lava', 'crater']):
            material_props = {
                'ambient': 0.32,
                'diffuse': 0.96,
                'specular': 0.01,
                'specular_power': 5
            }
        elif any(word in prompt_lower for word in ['canyon', 'mesa', 'plateau', 'badlands']):
            material_props = {
                'ambient': 0.40,
                'diffuse': 0.98,
                'specular': 0.02,
                'specular_power': 6
            }
        else:
            # Default: natural matte terrain
            material_props = {
                'ambient': 0.40,
                'diffuse': 0.98,
                'specular': 0.02,
                'specular_power': 8
            }
        
        # Add mesh with matte properties
        actor = plotter.add_mesh(
            mesh,
            rgb=True,
            smooth_shading=True,
            show_edges=False,
            **material_props
        )
        
        return actor
    
    def _set_cinematic_camera(self, plotter, mesh, heightmap):
        """Position camera opposite the peak so the highest point faces the viewer."""
        bounds = mesh.bounds
        
        # Scene geometry
        x_center = (bounds[0] + bounds[1]) / 2
        y_center = (bounds[2] + bounds[3]) / 2
        z_center = (bounds[4] + bounds[5]) / 2
        z_max = bounds[5]
        z_range = bounds[5] - bounds[4]
        x_size = bounds[1] - bounds[0]
        y_size = bounds[3] - bounds[2]
        max_size = max(x_size, y_size)
        
        # Find the peak (highest point) in the heightmap
        peak_idx = np.unravel_index(np.argmax(heightmap), heightmap.shape)
        # Convert heightmap indices (row, col) to mesh coordinates
        # row 0 → y_max, row H → y_min  (image rows go top-down)
        # col 0 → x_min, col W → x_max
        h, w = heightmap.shape[:2]
        peak_x = bounds[0] + (peak_idx[1] / max(w - 1, 1)) * x_size
        peak_y = bounds[3] - (peak_idx[0] / max(h - 1, 1)) * y_size  # flip row axis
        
        # Camera goes on the OPPOSITE side of the peak relative to center
        dx = peak_x - x_center
        dy = peak_y - y_center
        
        camera_distance = max_size * 1.2
        camera_height = z_max + z_range * 1.2
        
        # Place camera opposite the peak direction
        camera_position = [
            x_center - dx * 1.4,  # opposite side of peak
            y_center - dy * 1.4,  # opposite side of peak
            camera_height
        ]
        
        # Focus on the peak area (slightly above the terrain center)
        focus_point = [
            (x_center + peak_x) / 2,  # midpoint between center and peak
            (y_center + peak_y) / 2,
            z_center
        ]
        
        plotter.camera_position = [
            camera_position,
            focus_point,
            [0, 0, 1]  # up vector
        ]
        
        plotter.reset_camera()
        
        logger.info(f"Camera configured - Peak at ({peak_x:.1f}, {peak_y:.1f}), "
                     f"Camera opposite at {camera_position}")
    
    def _setup_interactive_plotter(self, terrain_prompt):
        """Set up interactive plotter with MINIMAL settings for debugging"""
        import os
        
        # Create simple interactive plotter
        plotter = pv.Plotter(
            window_size=(1400, 900),
            off_screen=False,
            notebook=False
        )
        
        # Nice background gradient (sky-like)
        plotter.set_background('aliceblue', top='skyblue')
        
        # Enable basic lighting
        plotter.enable_lightkit()
        
        logger.info(f"GPU Rendering: Using NVIDIA GeForce GTX 1650")
        
        return plotter
    
    def _add_interactive_controls(self, plotter, mesh, terrain_prompt):
        """Add interactive controls and UI elements"""
        
        # Add control instructions as text
        controls_text = (
            "INTERACTIVE CONTROLS:\n"
            "• Mouse: Rotate view\n"  
            "• Scroll: Zoom in/out\n"
            "• Right-click + drag: Pan\n"
            "• 'r': Reset camera\n"
            "• 'q' or ESC: Quit"
        )
        
        plotter.add_text(
            controls_text,
            position='upper_left',
            font_size=12,
            color='black',  # Changed from white to black for visibility on white background
            shadow=True
        )
        
        # Add terrain info
        bounds = mesh.bounds
        z_range = bounds[5] - bounds[4]
        info_text = (
            f"TERRAIN INFO:\n"
            f"Prompt: {terrain_prompt[:40]}...\n"
            f"Points: {mesh.n_points:,}\n"
            f"Height Range: {z_range:.1f} units\n"
            f"Resolution: High (60x scaling)"
        )
        
        plotter.add_text(
            info_text,
            position='upper_right',
            font_size=10,
            color='darkgray',  # Changed from lightgray for better contrast
            shadow=True
        )
        
        # Add coordinate axes for reference
        plotter.show_axes()
        
        return plotter
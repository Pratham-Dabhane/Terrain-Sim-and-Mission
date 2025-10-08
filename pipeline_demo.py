"""
Complete Terrain Generation Pipeline Example
Demonstrates the full workflow from text prompt to 3D mesh.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add pipeline to path
sys.path.append('pipeline')

try:
    from pipeline.clip_encoder import CLIPTextEncoderWithProcessor
    from pipeline.models_awcgan import CLIPConditionedGenerator
    from pipeline.mesh_visualize import TerrainMeshGenerator, TerrainVisualizer, MeshExporter
    from pipeline.data import create_synthetic_data
    from pipeline.train_awcgan import create_training_config
    from pipeline.realistic_terrain_enhancer import RealisticTerrainEnhancer
    from pipeline.advanced_terrain_renderer import AdvancedTerrainRenderer
    from pipeline.config import PipelineConfig, get_default_config
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Make sure all pipeline files are in the 'pipeline' directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TerrainPipelineDemo:
    """
    Complete demonstration of the terrain generation pipeline.
    """
    
    def __init__(self, output_dir: str = "Output", device: str = None, config: PipelineConfig = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store configuration (use default if none provided)
        self.config = config or get_default_config()
        logger.info(f"Pipeline initialized with config: {self.config._count_enabled_features()} features enabled")
        
        # Auto-detect device the old way (maintaining existing behavior)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.clip_encoder = None
        self.generator = None
        self.enhancer = None
        self.mesh_generator = None
        self.visualizer = None
        self.exporter = None
        
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        try:
            # CLIP encoder
            self.clip_encoder = CLIPTextEncoderWithProcessor(
                embedding_dim=512,
                device=self.device
            )
            logger.info("✓ CLIP encoder initialized")
            
            # GAN generator
            self.generator = CLIPConditionedGenerator(
                noise_dim=128,
                clip_dim=512,
                output_size=256
            ).to(self.device)
            
            # Try to load pre-trained weights
            checkpoint_path = Path("checkpoints/best_checkpoint.pth")
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                logger.info("✓ GAN generator initialized with pre-trained weights")
            else:
                logger.info("✓ GAN generator initialized (using random weights)")
            
            self.generator.eval()
            
            # Realistic terrain enhancer
            try:
                self.enhancer = RealisticTerrainEnhancer()
                logger.info("✓ Realistic terrain enhancer initialized")
            except Exception as e:
                logger.warning(f"Terrain enhancer initialization failed: {e}")
                self.enhancer = None
            
            # 3D mesh components
            self.mesh_generator = TerrainMeshGenerator(method="structured_grid")
            self.visualizer = TerrainVisualizer()
            self.exporter = MeshExporter()
            self.advanced_renderer = AdvancedTerrainRenderer()
            logger.info("✓ 3D mesh components initialized")
            logger.info("✓ Advanced photorealistic renderer initialized")
            
            logger.info("Pipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def generate_heightmap(self, prompt: str, seed: int = None) -> np.ndarray:
        """
        Generate heightmap from text prompt using AI, procedural, or GAN methods.
        
        Args:
            prompt: Text description of terrain
            seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Generated heightmap (256x256)
        """
        logger.info(f"Generating heightmap for: '{prompt}'")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Check if AI heightmap generation is enabled
        if self.config.ai_heightmap_enabled:
            logger.info("Using AI diffusion model for heightmap generation")
            try:
                from pipeline.ai_heightmap_generator import create_ai_heightmap_generator
                
                # Create AI generator with memory checks
                ai_generator = create_ai_heightmap_generator(device=self.device)
                
                if ai_generator is not None:
                    # Check memory requirements
                    memory_info = ai_generator.check_memory_requirements()
                    
                    if memory_info["recommended"]:
                        # Generate heightmap using AI
                        heightmap = ai_generator.generate_heightmap(
                            prompt=prompt,
                            size=self.config.base_heightmap_size,
                            seed=seed
                        )
                        
                        if heightmap is not None:
                            logger.info(f"✓ AI heightmap generated: shape={heightmap.shape}, range=[{heightmap.min():.3f}, {heightmap.max():.3f}]")
                            # Clean up to free memory
                            ai_generator.cleanup()
                            return heightmap
                        else:
                            logger.warning("AI heightmap generation returned None, falling back")
                    else:
                        logger.warning(f"AI heightmap generation not recommended: {memory_info.get('reason', 'Unknown')}")
                        logger.info("Falling back to procedural/GAN generation")
                    
                    # Clean up resources
                    ai_generator.cleanup()
                else:
                    logger.warning("AI heightmap generator not available, falling back")
                    
            except Exception as e:
                logger.warning(f"AI heightmap generation failed, falling back: {e}")
                # Continue to other generation methods
        
        # Check if procedural generation is enabled
        if self.config.procedural_enabled:
            logger.info("Using procedural terrain generation")
            try:
                import pipeline.procedural_noise_utils as pn
                from pipeline.prompt_parser import parse_prompt_to_params
                
                # Parse prompt to extract terrain parameters
                parsed = parse_prompt_to_params(prompt)
                
                # Create default noise parameters
                default_params = {
                    'scale': 100.0,
                    'octaves': 6,
                    'persistence': 0.5,
                    'lacunarity': 2.0,
                    'mountain_weight': 0.8,
                    'valley_weight': 0.2,
                    'river_strength': 0.3,
                    'river_frequency': 0.05,
                    'seed': seed
                }
                
                # Merge parsed parameters with defaults
                procedural_params = default_params.copy()
                
                # Map parsed parameters to NoiseParams fields
                if 'mountain_octaves' in parsed:
                    procedural_params['octaves'] = parsed['mountain_octaves']
                if 'scale' in parsed:
                    procedural_params['scale'] = parsed['scale'] * 20.0  # Scale up for better terrain
                if 'persistence' in parsed:
                    procedural_params['persistence'] = parsed['persistence']
                if 'lacunarity' in parsed:
                    procedural_params['lacunarity'] = parsed['lacunarity']
                if 'mountain_weight' in parsed:
                    procedural_params['mountain_weight'] = parsed['mountain_weight']
                if 'valley_weight' in parsed:
                    procedural_params['valley_weight'] = parsed['valley_weight']
                if 'river_strength' in parsed:
                    procedural_params['river_strength'] = parsed['river_strength']
                if 'river_freq' in parsed:
                    procedural_params['river_frequency'] = parsed['river_freq']
                
                # Log the final parameters
                logger.info(f"Procedural params: {procedural_params}")
                
                # Create NoiseParams object
                noise_params = pn.NoiseParams(
                    scale=procedural_params['scale'],
                    octaves=procedural_params['octaves'],
                    persistence=procedural_params['persistence'],
                    lacunarity=procedural_params['lacunarity'],
                    mountain_weight=procedural_params['mountain_weight'],
                    valley_weight=procedural_params['valley_weight'],
                    river_strength=procedural_params['river_strength'],
                    river_frequency=procedural_params['river_frequency'],
                    seed=seed
                )
                
                # Generate procedural heightmap
                heightmap = pn.generate_procedural_heightmap(
                    shape=self.config.base_heightmap_size,
                    params=noise_params
                )
                
                logger.info(f"✓ Procedural heightmap generated: shape={heightmap.shape}, range=[{heightmap.min():.3f}, {heightmap.max():.3f}]")
                return heightmap
                
            except Exception as e:
                logger.warning(f"Procedural generation failed, falling back to GAN: {e}")
                # Continue to GAN generation below
        
        # Original GAN-based generation path
        with torch.no_grad():
            # Encode text prompt
            clip_embedding = self.clip_encoder.encode_text_with_enhancement(
                prompt, enhance_prompts=True
            )
            
            # Generate noise
            noise = torch.randn(1, 128, device=self.device)
            
            # Generate heightmap
            heightmap_tensor = self.generator(noise, clip_embedding)
            heightmap = heightmap_tensor.cpu().numpy()[0, 0]
            
            # Normalize to [0, 1]
            heightmap = (heightmap + 1) / 2
            heightmap = np.clip(heightmap, 0, 1)
        
        return heightmap
    
    def remaster_heightmap(
        self, 
        heightmap: np.ndarray, 
        prompt: str,
        output_size: tuple = (512, 512),
        seed: int = None
    ) -> tuple:
        """
        Enhance heightmap with realistic terrain coloring and textures.
        
        Args:
            heightmap: Input heightmap
            prompt: Text prompt for enhancement
            output_size: Target output size
            seed: Random seed
            
        Returns:
            tuple: (enhanced_image, enhanced_heightmap)
        """
        if not self.enhancer:
            logger.info("Terrain enhancer disabled")
            
            # Check if texture mapping is enabled
            if self.config.texture_mapping_enabled:
                logger.info("Using slope-based color mapping for terrain visualization")
                from pipeline.terrain_texture_mapper import colorize_by_elevation_and_slope
                enhanced = colorize_by_elevation_and_slope(heightmap)
                enhanced = enhanced.resize(output_size)
                return enhanced, heightmap
            
            # Fallback to basic grayscale
            logger.info("Using basic grayscale fallback")
            enhanced = Image.fromarray((heightmap * 255).astype(np.uint8))
            enhanced = enhanced.convert('RGB').resize(output_size)
            return enhanced, heightmap
        
        logger.info(f"Enhancing heightmap with realistic terrain colors...")
        
        try:
            # Convert heightmap to proper format for enhancer
            heightmap_uint8 = (heightmap * 255).astype(np.uint8)
            
            # Apply realistic terrain enhancement
            enhanced_rgb = self.enhancer.enhance_heightmap(
                heightmap=heightmap_uint8,
                prompt=prompt,
                apply_lighting=True,
                texture_strength=0.08
            )
            
            # Convert to PIL Image and resize if needed
            enhanced_image = Image.fromarray(enhanced_rgb, mode='RGB')
            if enhanced_image.size != output_size:
                enhanced_image = enhanced_image.resize(output_size, Image.Resampling.LANCZOS)
            
            logger.info(f"✓ Terrain enhancement complete: {enhanced_image.size}")
            return enhanced_image, heightmap
            
        except Exception as e:
            logger.error(f"Terrain enhancement failed: {e}")
            # Fallback to basic enhancement
            enhanced = Image.fromarray((heightmap * 255).astype(np.uint8))
            enhanced = enhanced.convert('RGB').resize(output_size)
            return enhanced, heightmap
    
    def create_3d_mesh(self, heightmap: np.ndarray, scale_factor: float = 20.0):
        """
        Generate 3D mesh from heightmap.
        
        Args:
            heightmap: Input heightmap
            scale_factor: Z-axis scaling for elevation
            
        Returns:
            pv.DataSet: 3D mesh
        """
        logger.info("Generating 3D mesh...")
        
        mesh = self.mesh_generator.generate_mesh(
            heightmap=heightmap,
            x_scale=1.0,
            y_scale=1.0,
            z_scale=scale_factor
        )
        
        return mesh
    
    def save_results(
        self, 
        prompt: str,
        heightmap: np.ndarray,
        enhanced_image: Image.Image,
        mesh,
        session_id: str
    ) -> Dict[str, str]:
        """
        Save all generated results to files.
        
        Args:
            prompt: Original text prompt
            heightmap: Generated heightmap
            enhanced_image: Enhanced terrain image
            mesh: 3D mesh
            session_id: Unique session identifier
            
        Returns:
            Dict[str, str]: Paths to saved files
        """
        logger.info("Saving results...")
        
        # Create session directory
        session_dir = self.output_dir / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)
        
        paths = {}
        
        # Save heightmap
        heightmap_path = session_dir / "heightmap.png"
        heightmap_img = Image.fromarray((heightmap * 255).astype(np.uint8))
        heightmap_img.save(heightmap_path)
        paths["heightmap"] = str(heightmap_path)
        
        # Save enhanced terrain image
        enhanced_path = session_dir / "enhanced_terrain.png"
        enhanced_image.save(enhanced_path)
        paths["enhanced"] = str(enhanced_path)
        
        # Save 3D mesh
        mesh_vtk_path = session_dir / "mesh.vtk"
        try:
            self.exporter.export_mesh(mesh, str(mesh_vtk_path))
            paths["mesh_vtk"] = str(mesh_vtk_path)
        except Exception as e:
            logger.warning(f"Failed to export mesh: {e}")
            paths["mesh_vtk"] = None
        
        # Save standard 3D visualization
        viz_path = session_dir / "visualization_3d.png"
        self.visualizer.visualize_terrain(
            mesh=mesh,
            title=f"Terrain: {prompt}",
            save_path=str(viz_path),
            interactive=False,
            enhanced_texture=enhanced_image
        )
        paths["visualization"] = str(viz_path)
        
        # Save PHOTOREALISTIC 3D visualization
        photorealistic_path = session_dir / "visualization_3d_PHOTOREALISTIC.png"
        try:
            self.advanced_renderer.create_photorealistic_visualization(
                heightmap=heightmap,
                enhanced_texture=enhanced_image,
                terrain_prompt=prompt,
                output_path=str(photorealistic_path)
            )
            paths["photorealistic_viz"] = str(photorealistic_path)
            logger.info(f"✓ Photorealistic visualization saved: {photorealistic_path}")
        except Exception as e:
            logger.warning(f"Failed to create photorealistic visualization: {e}")
            paths["photorealistic_viz"] = None
        
        # Save metadata
        metadata = {
            "prompt": prompt,
            "session_id": session_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "heightmap_shape": list(heightmap.shape),
            "heightmap_range": [float(heightmap.min()), float(heightmap.max())],
            "has_photorealistic_viz": paths.get("photorealistic_viz") is not None,
            "files": paths
        }
        
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        paths["metadata"] = str(metadata_path)
        
        return paths
    
    def run_complete_pipeline(
        self, 
        prompt: str, 
        seed: int = None,
        enable_remastering: bool = True,
        output_size: tuple = (512, 512)
    ) -> Dict[str, Any]:
        """
        Run the complete terrain generation pipeline.
        
        Args:
            prompt: Text description of terrain
            seed: Random seed for reproducibility
            enable_remastering: Whether to use realistic terrain enhancement
            output_size: Size for final outputs
            
        Returns:
            Dict with all results and file paths
        """
        session_id = f"{int(time.time())}"
        logger.info(f"Starting complete pipeline for session {session_id}")
        logger.info(f"Prompt: '{prompt}'")
        
        start_time = time.time()
        
        try:
            # Step 1: Generate heightmap
            heightmap = self.generate_heightmap(prompt, seed)
            logger.info(f"Heightmap generated: {heightmap.shape}, range [{heightmap.min():.3f}, {heightmap.max():.3f}]")
            
            # Step 2: Enhance with realistic terrain coloring (optional)
            if enable_remastering:
                enhanced_image, enhanced_heightmap = self.remaster_heightmap(
                    heightmap, prompt, output_size, seed
                )
                final_heightmap = enhanced_heightmap
            else:
                # Check if texture mapping is enabled for fallback coloring
                if self.config.texture_mapping_enabled:
                    logger.info("Remastering disabled - using slope-based color mapping")
                    from pipeline.terrain_texture_mapper import colorize_by_elevation_and_slope
                    enhanced_image = colorize_by_elevation_and_slope(heightmap)
                    enhanced_image = enhanced_image.resize(output_size)
                else:
                    # Basic grayscale fallback
                    logger.info("Remastering disabled - using basic grayscale")
                    enhanced_image = Image.fromarray((heightmap * 255).astype(np.uint8))
                    enhanced_image = enhanced_image.convert('RGB').resize(output_size)
                final_heightmap = heightmap
            
            # Step 3: Generate 3D mesh
            mesh = self.create_3d_mesh(final_heightmap)
            logger.info(f"3D mesh generated: {mesh.n_points} points, {mesh.n_cells} cells")
            
            # Step 4: Save all results
            file_paths = self.save_results(
                prompt, heightmap, enhanced_image, mesh, session_id
            )
            
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            results = {
                "session_id": session_id,
                "prompt": prompt,
                "seed": seed,
                "heightmap": heightmap,
                "enhanced_image": enhanced_image,
                "mesh": mesh,
                "file_paths": file_paths,
                "elapsed_time": elapsed_time,
                "success": True
            }
            
            logger.info(f"Pipeline completed successfully in {elapsed_time:.1f} seconds")
            logger.info(f"Results saved to: {self.output_dir / f'session_{session_id}'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "session_id": session_id,
                "prompt": prompt,
                "error": str(e),
                "success": False
            }
    
    def create_comparison_visualization(self, results_list: List[Dict]) -> str:
        """
        Create a comparison visualization of multiple results.
        
        Args:
            results_list: List of pipeline results
            
        Returns:
            str: Path to comparison image
        """
        logger.info("Creating comparison visualization...")
        
        n_results = len(results_list)
        fig, axes = plt.subplots(2, n_results, figsize=(4*n_results, 8))
        
        if n_results == 1:
            axes = axes.reshape(-1, 1)
        
        for i, result in enumerate(results_list):
            if not result["success"]:
                continue
            
            heightmap = result["heightmap"]
            remastered = result["remastered_image"]
            prompt = result["prompt"]
            
            # Plot heightmap
            axes[0, i].imshow(heightmap, cmap='terrain', vmin=0, vmax=1)
            axes[0, i].set_title(f"Heightmap\n{prompt[:30]}...", fontsize=10)
            axes[0, i].axis('off')
            
            # Plot remastered
            axes[1, i].imshow(remastered)
            axes[1, i].set_title(f"Remastered\n{prompt[:30]}...", fontsize=10)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        comparison_path = self.output_dir / "comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison saved to: {comparison_path}")
        return str(comparison_path)


def run_demo_examples():
    """Run demonstration with example prompts"""
    
    # Example prompts for different terrain types
    example_prompts = [
        "Majestic mountain ranges with snow-capped peaks and deep valleys",
        "Rolling green hills with a meandering river through the landscape",
        "Volcanic terrain with craters and lava rock formations",
        "Desert landscape with sand dunes and rocky outcrops",
        "Coastal cliffs overlooking the ocean with beaches",
        "Forest-covered mountains with lakes and waterfalls"
    ]
    
    # Initialize pipeline
    logger.info("=== Terrain Generation Pipeline Demo ===")
    pipeline = TerrainPipelineDemo()
    
    # Run examples
    results = []
    for i, prompt in enumerate(example_prompts[:3]):  # Limit to 3 for demo
        logger.info(f"\n--- Example {i+1}/3 ---")
        
        result = pipeline.run_complete_pipeline(
            prompt=prompt,
            seed=42 + i,  # Different seed for each
            enable_remastering=True,  # Enable SD enhancement
            output_size=(512, 512)
        )
        
        results.append(result)
        
        if result["success"]:
            logger.info(f"✓ Generated terrain for: {prompt}")
        else:
            logger.error(f"✗ Failed to generate terrain: {result.get('error', 'Unknown error')}")
    
    # Create comparison
    successful_results = [r for r in results if r["success"]]
    if len(successful_results) > 1:
        comparison_path = pipeline.create_comparison_visualization(successful_results)
        logger.info(f"Comparison visualization: {comparison_path}")
    
    # Summary
    logger.info(f"\n=== Demo Summary ===")
    logger.info(f"Total examples: {len(example_prompts[:3])}")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(results) - len(successful_results)}")
    logger.info(f"Results saved to: Output/")
    
    return results


def run_interactive_demo():
    """Run interactive demo where user can input prompts"""
    
    pipeline = TerrainPipelineDemo()
    
    print("\n=== Interactive Terrain Generation ===")
    print("Enter terrain descriptions to generate 3D landscapes!")
    print("Type 'quit' to exit.\n")
    
    session_count = 0
    
    while True:
        try:
            prompt = input("Enter terrain description: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                print("Please enter a valid description.")
                continue
            
            session_count += 1
            print(f"\nGenerating terrain {session_count}...")
            
            result = pipeline.run_complete_pipeline(
                prompt=prompt,
                seed=None,  # Random seed
                enable_remastering=True
            )
            
            if result["success"]:
                print(f"✓ Success! Generated in {result['elapsed_time']:.1f}s")
                print(f"Files saved to: {result['file_paths']['metadata']}")
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nGenerated {session_count} terrains. Thank you!")


def run_batch_demo():
    """Run batch generation with predefined prompts"""
    
    print("=== Batch Terrain Generation ===")
    print("Generating multiple terrain examples...")
    
    pipeline = TerrainPipelineDemo()
    
    # Predefined batch prompts
    batch_prompts = [
        "rocky mountains with snow peaks",
        "rolling hills with gentle slopes",
        "volcanic terrain with craters",
        "desert landscape with sand dunes",
        "coastal cliffs with rocky outcrops",
        "forest terrain with rivers and valleys",
        "alpine landscape with glaciers",
        "canyon landscape with deep ravines"
    ]
    
    results = []
    
    for i, prompt in enumerate(batch_prompts, 1):
        print(f"\nGenerating terrain {i}/{len(batch_prompts)}: '{prompt}'")
        
        try:
            result = pipeline.run_complete_pipeline(
                prompt=prompt,
                seed=i * 42,  # Consistent seeds for reproducibility
                enable_remastering=True
            )
            
            if result["success"]:
                print(f"✓ Success! Generated in {result['elapsed_time']:.1f}s")
                results.append({
                    'prompt': prompt,
                    'success': True,
                    'time': result['elapsed_time'],
                    'files': result['file_paths']
                })
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")
                results.append({
                    'prompt': prompt,
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
                
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                'prompt': prompt,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r.get('time', 0) for r in results if r['success'])
    
    print(f"\n=== Batch Generation Complete ===")
    print(f"Successful: {successful}/{len(batch_prompts)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time: {total_time/successful:.1f}s per terrain" if successful > 0 else "")
    print(f"Output folder: {pipeline.output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Terrain Generation Pipeline Demo")
    parser.add_argument("--mode", choices=["demo", "interactive", "batch"], default="demo",
                       help="Demo mode: demo (examples), interactive (user input), or batch (multiple examples)")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--no-remaster", action="store_true", 
                       help="Disable Stable Diffusion remastering")
    parser.add_argument("--interactive-3d", action="store_true",
                       help="Launch interactive 3D viewer after generation")
    parser.add_argument("--simulate-path", action="store_true",
                       help="Enable interactive mission path simulation")
    
    args = parser.parse_args()
    
    try:
        if args.prompt:
            # Single prompt mode
            pipeline = TerrainPipelineDemo()
            result = pipeline.run_complete_pipeline(
                prompt=args.prompt,
                seed=args.seed,
                enable_remastering=not args.no_remaster
            )
            
            if result["success"]:
                print(f"✓ Generated terrain for: '{args.prompt}'")
                print(f"Results: {result['file_paths']['metadata']}")
                
                # Launch interactive 3D viewer if requested
                if args.interactive_3d:
                    try:
                        print("\n🎮 Launching interactive 3D viewer...")
                        session_dir = Path(result['file_paths']['metadata']).parent
                        
                        # Load terrain data for interactive viewer
                        heightmap_path = session_dir / "heightmap.png"
                        enhanced_path = session_dir / "enhanced_terrain.png"
                        
                        if heightmap_path.exists() and enhanced_path.exists():
                            heightmap_img = Image.open(heightmap_path).convert('L')
                            heightmap = np.array(heightmap_img).astype(np.float32) / 255.0
                            enhanced_texture = Image.open(enhanced_path)
                            
                            pipeline.advanced_renderer.create_interactive_photorealistic_visualization(
                                heightmap=heightmap,
                                enhanced_texture=enhanced_texture,
                                terrain_prompt=args.prompt
                            )
                            print("✓ Interactive session completed!")
                        else:
                            print("❌ Could not find terrain files for interactive viewer")
                            
                    except Exception as e:
                        print(f"❌ Failed to launch interactive viewer: {e}")
                
                # Launch mission simulation if requested
                if args.simulate_path or pipeline.config.mission_sim_enabled:
                    try:
                        print("\n🎯 Launching mission path simulation...")
                        from pipeline.mission_simulator import select_points_and_simulate
                        
                        # Use the generated heightmap for mission planning
                        heightmap = result['heightmap']
                        session_dir = Path(result['file_paths']['metadata']).parent
                        
                        # Run interactive mission simulation
                        mission_output = select_points_and_simulate(
                            heightmap=heightmap, 
                            output_dir=str(session_dir)
                        )
                        
                        if mission_output:
                            print(f"✓ Mission simulation completed: {mission_output}")
                        else:
                            print("❌ Mission simulation was cancelled or failed")
                            
                    except Exception as e:
                        print(f"❌ Failed to launch mission simulation: {e}")
                        logger.error(f"Mission simulation error: {e}")
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")
                
        elif args.mode == "interactive":
            run_interactive_demo()
        elif args.mode == "batch":
            run_batch_demo()
        else:
            run_demo_examples()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
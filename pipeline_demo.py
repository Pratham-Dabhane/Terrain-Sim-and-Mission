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
    from pipeline.remaster_sd_controlnet import TerrainRemaster
    from pipeline.mesh_visualize import TerrainMeshGenerator, TerrainVisualizer, MeshExporter
    from pipeline.data import create_synthetic_data
    from pipeline.train_awcgan import create_training_config
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
    
    def __init__(self, output_dir: str = "Output", device: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.clip_encoder = None
        self.generator = None
        self.remaster = None
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
            
            # Stable Diffusion remaster (optional)
            try:
                self.remaster = TerrainRemaster(
                    controlnet_type="depth",
                    device=self.device,
                    use_fp16=True,
                    enable_cpu_offload=True
                )
                logger.info("✓ Stable Diffusion remaster initialized")
            except Exception as e:
                logger.warning(f"Stable Diffusion not available: {e}")
                self.remaster = None
            
            # 3D mesh components
            self.mesh_generator = TerrainMeshGenerator(method="structured_grid")
            self.visualizer = TerrainVisualizer()
            self.exporter = MeshExporter()
            logger.info("✓ 3D mesh components initialized")
            
            logger.info("Pipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def generate_heightmap(self, prompt: str, seed: int = None) -> np.ndarray:
        """
        Generate heightmap from text prompt using GAN.
        
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
        Enhance heightmap with Stable Diffusion + ControlNet.
        
        Args:
            heightmap: Input heightmap
            prompt: Text prompt for enhancement
            output_size: Target output size
            seed: Random seed
            
        Returns:
            tuple: (remastered_image, enhanced_heightmap)
        """
        if not self.remaster:
            logger.warning("Stable Diffusion not available, skipping remastering")
            # Create fallback enhanced image
            enhanced = Image.fromarray((heightmap * 255).astype(np.uint8), mode='L')
            enhanced = enhanced.convert('RGB').resize(output_size)
            return enhanced, heightmap
        
        logger.info(f"Remastering heightmap with Stable Diffusion...")
        
        try:
            remastered_image, enhanced_heightmap = self.remaster.remaster_heightmap(
                heightmap=heightmap,
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, unrealistic",
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.8,
                output_size=output_size,
                seed=seed,
                preserve_geometry=True
            )
            
            return remastered_image, enhanced_heightmap
            
        except Exception as e:
            logger.error(f"Remastering failed: {e}")
            # Fallback to basic enhancement
            enhanced = Image.fromarray((heightmap * 255).astype(np.uint8), mode='L')
            enhanced = enhanced.convert('RGB').resize(output_size)
            return enhanced, heightmap
    
    def create_3d_mesh(self, heightmap: np.ndarray, scale_factor: float = 0.3):
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
        remastered_image: Image.Image,
        mesh,
        session_id: str
    ) -> Dict[str, str]:
        """
        Save all generated results to files.
        
        Args:
            prompt: Original text prompt
            heightmap: Generated heightmap
            remastered_image: Enhanced image
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
        heightmap_img = Image.fromarray((heightmap * 255).astype(np.uint8), mode='L')
        heightmap_img.save(heightmap_path)
        paths["heightmap"] = str(heightmap_path)
        
        # Save remastered image
        remastered_path = session_dir / "remastered.png"
        remastered_image.save(remastered_path)
        paths["remastered"] = str(remastered_path)
        
        # Save 3D mesh
        mesh_ply_path = session_dir / "mesh.ply"
        mesh_obj_path = session_dir / "mesh.obj"
        self.exporter.export_mesh(mesh, str(mesh_ply_path))
        self.exporter.export_mesh(mesh, str(mesh_obj_path))
        paths["mesh_ply"] = str(mesh_ply_path)
        paths["mesh_obj"] = str(mesh_obj_path)
        
        # Save 3D visualization
        viz_path = session_dir / "visualization_3d.png"
        self.visualizer.visualize_terrain(
            mesh=mesh,
            title=f"Terrain: {prompt}",
            save_path=str(viz_path),
            interactive=False
        )
        paths["visualization"] = str(viz_path)
        
        # Save metadata
        metadata = {
            "prompt": prompt,
            "session_id": session_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "heightmap_shape": list(heightmap.shape),
            "heightmap_range": [float(heightmap.min()), float(heightmap.max())],
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
            enable_remastering: Whether to use Stable Diffusion enhancement
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
            
            # Step 2: Remaster with Stable Diffusion (optional)
            if enable_remastering:
                remastered_image, enhanced_heightmap = self.remaster_heightmap(
                    heightmap, prompt, output_size, seed
                )
                final_heightmap = enhanced_heightmap
            else:
                # Use original heightmap
                remastered_image = Image.fromarray((heightmap * 255).astype(np.uint8), mode='L')
                remastered_image = remastered_image.convert('RGB').resize(output_size)
                final_heightmap = heightmap
            
            # Step 3: Generate 3D mesh
            mesh = self.create_3d_mesh(final_heightmap)
            logger.info(f"3D mesh generated: {mesh.n_points} points, {mesh.n_cells} cells")
            
            # Step 4: Save all results
            file_paths = self.save_results(
                prompt, heightmap, remastered_image, mesh, session_id
            )
            
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            results = {
                "session_id": session_id,
                "prompt": prompt,
                "seed": seed,
                "heightmap": heightmap,
                "remastered_image": remastered_image,
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Terrain Generation Pipeline Demo")
    parser.add_argument("--mode", choices=["demo", "interactive"], default="demo",
                       help="Demo mode: run examples or interactive input")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--no-remaster", action="store_true", 
                       help="Disable Stable Diffusion remastering")
    
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
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")
                
        elif args.mode == "interactive":
            run_interactive_demo()
        else:
            run_demo_examples()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
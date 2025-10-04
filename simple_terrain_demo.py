"""
Simple Terrain Generation Demo
Tests the core pipeline components step by step.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Add pipeline to path
sys.path.append("pipeline")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_generation():
    """Test basic terrain generation without complex dependencies"""
    print("🚀 Starting Basic Terrain Generation Test")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("Output", exist_ok=True)
    
    try:
        # 1. Test CLIP Encoder
        print("1️⃣  Testing CLIP Text Encoder...")
        from clip_encoder import CLIPTextEncoderWithProcessor
        
        clip_encoder = CLIPTextEncoderWithProcessor(embedding_dim=512)
        test_prompt = "mountainous terrain with steep rocky peaks and deep valleys"
        
        with torch.no_grad():
            clip_embedding = clip_encoder.encode_text_with_enhancement(test_prompt, enhance_prompts=True)
        
        print(f"   ✅ CLIP encoding successful: {clip_embedding.shape}")
        
        # 2. Test aWCGAN Generator
        print("2️⃣  Testing aWCGAN Generator...")
        from models_awcgan import CLIPConditionedGenerator
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = CLIPConditionedGenerator(noise_dim=128, clip_dim=512).to(device)
        generator.eval()
        
        # Generate heightmap
        torch.manual_seed(42)
        noise = torch.randn(1, 128, device=device)
        
        with torch.no_grad():
            heightmap_tensor = generator(noise, clip_embedding)
        
        # Convert to numpy and normalize
        heightmap = heightmap_tensor.cpu().numpy()[0, 0]
        heightmap = (heightmap + 1) / 2  # [-1, 1] to [0, 1]
        heightmap = np.clip(heightmap, 0, 1)
        
        print(f"   ✅ Heightmap generated: {heightmap.shape}, range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
        
        # Save heightmap
        heightmap_path = "Output/test_heightmap.png"
        heightmap_img = Image.fromarray((heightmap * 255).astype(np.uint8), mode='L')
        heightmap_img.save(heightmap_path)
        print(f"   ✅ Heightmap saved: {heightmap_path}")
        
        # 3. Test Basic 3D Visualization (fallback method)
        print("3️⃣  Testing 3D Mesh Generation...")
        try:
            # Try PyVista first
            import pyvista as pv
            
            # Create structured grid
            height, width = heightmap.shape
            x = np.linspace(0, width, width)
            y = np.linspace(0, height, height)
            X, Y = np.meshgrid(x, y)
            Z = heightmap * 30  # Scale elevation
            
            # Create mesh
            grid = pv.StructuredGrid(X, Y, Z)
            grid["elevation"] = heightmap.flatten(order='F')
            
            # Save as VTK (PyVista native format)
            mesh_path = "Output/test_terrain.vtk"
            grid.save(mesh_path)
            print(f"   ✅ 3D mesh saved: {mesh_path} ({grid.n_points} points)")
            
            # Create visualization
            try:
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(grid, cmap='terrain', show_scalar_bar=True)
                plotter.camera_position = [(width/2, height/2, 100), (width/2, height/2, 0), (0, 0, 1)]
                viz_path = "Output/test_terrain_3d.png"
                plotter.screenshot(viz_path)
                plotter.close()
                print(f"   ✅ 3D visualization saved: {viz_path}")
            except:
                print("   ⚠️  3D visualization failed (likely headless environment)")
                
        except ImportError:
            print("   ⚠️  PyVista not available, creating matplotlib 3D plot...")
            
            # Fallback: matplotlib 3D plot
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Downsample for performance
            step = 4
            X_sub = X[::step, ::step]
            Y_sub = Y[::step, ::step]
            Z_sub = Z[::step, ::step]
            
            surf = ax.plot_surface(X_sub, Y_sub, Z_sub, cmap='terrain', alpha=0.8)
            ax.set_title(f'Terrain: {test_prompt[:40]}...')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Elevation')
            
            plt.colorbar(surf, shrink=0.5, aspect=5)
            
            viz_path = "Output/test_terrain_matplotlib.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Matplotlib 3D plot saved: {viz_path}")
        
        # 4. Create comparison visualization
        print("4️⃣  Creating comparison visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Heightmap
        im1 = axes[0].imshow(heightmap, cmap='terrain', vmin=0, vmax=1)
        axes[0].set_title('Generated Heightmap')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Contour plot
        contour = axes[1].contour(heightmap, levels=10, colors='black', alpha=0.6, linewidths=0.5)
        contourf = axes[1].contourf(heightmap, levels=20, cmap='terrain', alpha=0.8)
        axes[1].set_title('Elevation Contours')
        axes[1].axis('off')
        plt.colorbar(contourf, ax=axes[1], shrink=0.8)
        
        plt.tight_layout()
        comparison_path = "Output/terrain_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comparison visualization saved: {comparison_path}")
        
        # 5. Generate multiple terrains
        print("5️⃣  Generating multiple terrain examples...")
        
        test_prompts = [
            "rolling hills with gentle slopes and meadows",
            "desert landscape with massive sand dunes",
            "volcanic terrain with craters and rocky formations",
            "coastal cliffs with steep drops to the ocean"
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, prompt in enumerate(test_prompts):
            # Encode prompt
            with torch.no_grad():
                clip_emb = clip_encoder.encode_text_with_enhancement(prompt, enhance_prompts=True)
                
                # Generate with different seed
                torch.manual_seed(42 + i)
                noise = torch.randn(1, 128, device=device)
                heightmap_tensor = generator(noise, clip_emb)
                
                heightmap = heightmap_tensor.cpu().numpy()[0, 0]
                heightmap = (heightmap + 1) / 2
                heightmap = np.clip(heightmap, 0, 1)
            
            # Plot heightmap
            im = axes[i].imshow(heightmap, cmap='terrain', vmin=0, vmax=1)
            axes[i].set_title(f"{i+1}. {prompt[:30]}...", fontsize=10)
            axes[i].axis('off')
            
            # Plot contour
            axes[i+4].contourf(heightmap, levels=15, cmap='terrain')
            axes[i+4].contour(heightmap, levels=8, colors='black', alpha=0.4, linewidths=0.5)
            axes[i+4].set_title(f"Contours {i+1}", fontsize=10)
            axes[i+4].axis('off')
            
            # Save individual heightmap
            individual_path = f"Output/terrain_{i+1}_heightmap.png"
            Image.fromarray((heightmap * 255).astype(np.uint8), mode='L').save(individual_path)
        
        plt.tight_layout()
        gallery_path = "Output/terrain_gallery.png"
        plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Terrain gallery saved: {gallery_path}")
        
        # Results summary
        print("\n" + "=" * 50)
        print("🎉 TERRAIN GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📁 Output folder: {os.path.abspath('Output')}")
        print(f"📊 Prompt: {test_prompt}")
        print(f"🖥️  Device: {device.upper()}")
        print(f"🧠 Model: aWCGAN with CLIP conditioning")
        print(f"📐 Resolution: {heightmap.shape}")
        
        # List generated files
        output_files = [f for f in os.listdir('Output') if f.startswith('test_') or f.startswith('terrain_')]
        print(f"\n📄 Generated {len(output_files)} files:")
        for file in sorted(output_files):
            print(f"   • {file}")
        
        print("\n🎯 Next steps:")
        print("   1. Open heightmap images to see generated terrain")
        print("   2. Load .vtk files in ParaView or Blender for 3D viewing")
        print("   3. Try different prompts by modifying the script")
        print("   4. Train your own model with real DEM data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_generation()
    if success:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed. Check the error messages above.")
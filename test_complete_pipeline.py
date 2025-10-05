"""
Final comprehensive test of the terrain generation pipeline
"""

import os
import warnings
import numpy as np
from PIL import Image
import torch

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def test_complete_pipeline():
    """Test the complete terrain generation pipeline"""
    
    print("🏔️  COMPREHENSIVE TERRAIN PIPELINE TEST")
    print("=" * 50)
    
    try:
        # Test 1: CLIP encoder
        print("\n1️⃣  Testing CLIP Text Encoder...")
        from pipeline.clip_encoder import CLIPTextEncoderWithProcessor
        
        clip_encoder = CLIPTextEncoderWithProcessor()
        test_prompts = [
            "rocky mountains with snow peaks",
            "rolling hills with gentle slopes", 
            "volcanic terrain with craters",
            "desert landscape with sand dunes"
        ]
        
        for prompt in test_prompts:
            embedding = clip_encoder.encode_text(prompt)
            print(f"   ✅ '{prompt}' -> {embedding.shape}")
        
        # Test 2: aWCGAN Generator
        print("\n2️⃣  Testing aWCGAN Generator...")
        from pipeline.models_awcgan import CLIPConditionedGenerator
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = CLIPConditionedGenerator(
            noise_dim=128,
            clip_dim=512,
            output_size=256
        ).to(device)
        
        # Generate terrain for each prompt
        results = []
        for i, prompt in enumerate(test_prompts):
            embedding = clip_encoder.encode_text(prompt)
            noise = torch.randn(1, 128, device=device)
            
            with torch.no_grad():
                heightmap = generator(noise, embedding)
                heightmap_np = heightmap.squeeze().cpu().numpy()
                heightmap_np = (heightmap_np + 1) / 2  # [-1,1] -> [0,1]
            
            # Save heightmap
            heightmap_img = (heightmap_np * 255).astype(np.uint8)
            img_path = f"Output/test_prompt_{i+1}_heightmap.png"
            Image.fromarray(heightmap_img, mode='L').save(img_path)
            
            results.append((prompt, heightmap_np, img_path))
            print(f"   ✅ '{prompt}' -> {heightmap_np.shape}, range: [{heightmap_np.min():.3f}, {heightmap_np.max():.3f}]")
        
        # Test 3: 3D Visualization
        print("\n3️⃣  Testing 3D Visualizations...")
        from enhanced_3d_viz import create_enhanced_3d_visualization
        
        for i, (prompt, heightmap, img_path) in enumerate(results):
            viz_path = f"Output/test_prompt_{i+1}_3d.png"
            success = create_enhanced_3d_visualization(
                heightmap, 
                viz_path,
                f"Terrain: {prompt}"
            )
            if success:
                print(f"   ✅ 3D visualization saved: {viz_path}")
        
        # Test 4: Create summary grid
        print("\n4️⃣  Creating Summary Grid...")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(test_prompts), 2, figsize=(12, 4*len(test_prompts)))
        
        for i, (prompt, heightmap, _) in enumerate(results):
            # Heightmap
            axes[i,0].imshow(heightmap, cmap='terrain')
            axes[i,0].set_title(f"Heightmap: {prompt}", fontsize=10)
            axes[i,0].axis('off')
            
            # 3D view (create small matplotlib 3D)
            ax3d = fig.add_subplot(len(test_prompts), 2, 2*i+2, projection='3d')
            
            h, w = heightmap.shape
            x = np.linspace(0, w, w)
            y = np.linspace(0, h, h)
            X, Y = np.meshgrid(x, y)
            
            step = max(1, w // 30)
            X_sub = X[::step, ::step]
            Y_sub = Y[::step, ::step]
            Z_sub = heightmap[::step, ::step] * max(w,h) * 0.2
            
            ax3d.plot_surface(X_sub, Y_sub, Z_sub, cmap='terrain', alpha=0.8, rcount=20, ccount=20)
            ax3d.set_title(f"3D: {prompt}", fontsize=10)
            ax3d.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.savefig('Output/pipeline_test_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Test 5: Stats summary
        print("\n5️⃣  Pipeline Statistics:")
        print(f"   💾 Device: {device}")
        print(f"   🧠 Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"   📐 Output resolution: 256x256")
        print(f"   🎨 Test prompts: {len(test_prompts)}")
        
        # List output files
        output_files = [f for f in os.listdir("Output") if f.startswith("test_prompt")]
        print(f"\n📁 Generated {len(output_files)} test files:")
        for f in sorted(output_files):
            print(f"   • {f}")
        
        print("\n🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    os.makedirs("Output", exist_ok=True)
    test_complete_pipeline()
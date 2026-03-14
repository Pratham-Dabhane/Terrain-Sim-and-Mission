"""
Simple Demo Script for Realistic Terrain Generation
Shows off the different biomes and terrain types supported by the pipeline.
"""

import sys
import time
sys.path.append('.')

from pipeline_demo import TerrainPipelineDemo

def run_terrain_demos():
    """Run a series of terrain generation demos."""
    
    print("🏔️  REALISTIC TERRAIN GENERATION PIPELINE DEMO")
    print("=" * 60)
    print()
    
    # Initialize pipeline
    print("🔧 Initializing terrain generation pipeline...")
    pipeline = TerrainPipelineDemo(output_dir="demo_output")
    print("✅ Pipeline ready!")
    print()
    
    # Demo prompts for different biomes
    demo_prompts = [
        ("❄️  Snow-Covered Mountains", "snow-covered mountain peaks with deep valleys"),
        ("🏜️  Desert Landscape", "vast desert landscape with rolling sand dunes"),  
        ("🌲 Forest Hills", "dense forest landscape with rolling green hills"),
        ("🧊 Arctic Tundra", "frozen arctic tundra with ice formations"),
        ("🏔️  Rocky Peaks", "jagged rocky mountain peaks with stone cliffs"),
    ]
    
    results = []
    
    for i, (name, prompt) in enumerate(demo_prompts, 1):
        print(f"{name}")
        print(f"📝 Prompt: '{prompt}'")
        print("🔄 Generating terrain...")
        
        start_time = time.time()
        
        try:
            # Use deterministic seed based on prompt (different for each due to different prompts)
            # Override seed parameter removed - will use hash-based deterministic seed
            result = pipeline.run_complete_pipeline(
                prompt=prompt,
                seed=None,  # Use deterministic seed from prompt
                enable_remastering=True
            )
            
            if result["success"]:
                elapsed = time.time() - start_time
                print(f"✅ Generated in {elapsed:.1f}s")
                print(f"📁 Saved to: {result['file_paths']['enhanced']}")
                results.append(result)
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print("-" * 40)
        print()
    
    print(f"🎉 Demo Complete! Generated {len(results)} terrain variations")
    print("📋 Results Summary:")
    
    for i, result in enumerate(results, 1):
        session_id = result['session_id']
        prompt = result['prompt']
        elapsed = result['elapsed_time']
        print(f"  {i}. {prompt[:50]}... ({elapsed:.1f}s)")
        print(f"     Session: {session_id}")
    
    print()
    print("🖼️  Check the 'demo_output' folder to see your realistic terrain images!")
    print("💡 Each terrain automatically gets:")
    print("   • Biome-appropriate coloring (snow, desert sand, forest green, etc.)")
    print("   • Elevation-based textures (peaks, valleys, slopes)")
    print("   • Realistic lighting and shadows")
    print("   • 3D mesh visualization")
    print()
    print("🚀 Your text-to-terrain pipeline is fully operational!")

if __name__ == "__main__":
    run_terrain_demos()
"""
Simple script to open and display the enhanced terrain images
"""

from PIL import Image
import os

def show_enhanced_images():
    """Open the enhanced terrain images to see the realistic coloring."""
    
    # Get the session directories
    sessions = [
        "Output/session_1759680108",  # Snow mountains
        "Output/session_1759680151",  # Desert
        "Output/session_1759680195"   # Forest
    ]
    
    prompts = [
        "snow-covered mountain peaks with deep valleys",
        "vast desert landscape with rolling sand dunes", 
        "dense forest landscape with rolling green hills"
    ]
    
    for i, (session_dir, prompt) in enumerate(zip(sessions, prompts)):
        enhanced_path = f"{session_dir}/enhanced_terrain.png"
        
        if os.path.exists(enhanced_path):
            print(f"\n🎯 Terrain {i+1}: {prompt}")
            print(f"📁 Opening: {enhanced_path}")
            
            try:
                # Open and show the image
                img = Image.open(enhanced_path)
                print(f"📏 Size: {img.size[0]}x{img.size[1]}")
                print(f"🎨 Mode: {img.mode}")
                
                # Calculate file size
                file_size = os.path.getsize(enhanced_path) / 1024
                print(f"💾 File size: {file_size:.1f} KB")
                
                # Show the image
                img.show(title=f"Enhanced Terrain: {prompt}")
                
                print("✅ Image opened in default viewer")
                
            except Exception as e:
                print(f"❌ Error opening {enhanced_path}: {e}")
        else:
            print(f"❌ Enhanced image not found: {enhanced_path}")
    
    print("\n💡 These are your REALISTIC TERRAIN images!")
    print("🎨 They should show proper colors (snow=white, desert=brown, forest=green)")
    print("📊 The rainbow-colored images you saw are 3D mesh visualizations")

if __name__ == "__main__":
    show_enhanced_images()
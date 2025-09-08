#!/usr/bin/env python3
"""
Lightweight Interactive Terrain Prototype
Runs without heavy diffusion models, focusing on GAN + enhancement.
"""

import numpy as np
import matplotlib.pyplot as plt
from terrain_prototype import (
    prompt_to_heightmap_gan,
    _basic_enhancement,
    visualize_terrain
)

def interactive_terrain_generation():
    """Interactive terrain generation without diffusion models."""
    print("=" * 70)
    print("🎮 Interactive Terrain Generation")
    print("=" * 70)
    print("This version focuses on GAN + enhancement pipeline")
    print("(No heavy diffusion models required)")
    
    # Example prompts
    example_prompts = [
        "mountainous terrain with rivers and valleys",
        "desert landscape with sand dunes",
        "forest terrain with rolling hills",
        "coastal cliffs with rocky outcrops",
        "alpine landscape with snow-capped peaks",
        "volcanic terrain with lava flows",
        "swampy marshland with islands",
        "arctic tundra with ice formations"
    ]
    
    print("\n💡 Example terrain prompts:")
    for i, prompt in enumerate(example_prompts, 1):
        print(f"{i:2d}. {prompt}")
    
    print("\n" + "-" * 70)
    
    while True:
        try:
            # Get user input
            user_prompt = input("\n🎯 Enter your terrain description (or 'quit' to exit): ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for using the terrain generator!")
                break
            
            if not user_prompt:
                user_prompt = "mountainous terrain with rivers and valleys"
                print(f"🎲 Using default prompt: '{user_prompt}'")
            
            print(f"\n🚀 Generating terrain for: '{user_prompt}'")
            print("-" * 70)
            
            # Step 1: Generate base heightmap using GAN
            print("\n📊 Step 1: Generating base heightmap...")
            heightmap = prompt_to_heightmap_gan(user_prompt, size=128)
            
            # Step 2: Apply enhancement (skip diffusion)
            print("\n🎨 Step 2: Applying terrain enhancement...")
            enhanced_terrain = _basic_enhancement(heightmap, user_prompt)
            
            # Step 3: Visualize results
            print("\n🖼️  Step 3: Creating visualization...")
            visualize_terrain(enhanced_terrain, heightmap, user_prompt, 
                           save_path=f"terrain_{len(user_prompt.split())}_words.png")
            
            print("\n" + "=" * 70)
            print("🎉 Terrain generation completed successfully!")
            print("=" * 70)
            
            # Ask if user wants to continue
            continue_gen = input("\n🔄 Generate another terrain? (y/n): ").strip().lower()
            if continue_gen not in ['y', 'yes', 'yeah', 'sure']:
                print("👋 Thanks for using the terrain generator!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Generation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during terrain generation: {e}")
            print("Please try a different prompt or check your installation.")
            
            retry = input("\n🔄 Try again? (y/n): ").strip().lower()
            if retry not in ['y', 'yes', 'yeah', 'sure']:
                break

def quick_demo():
    """Quick demo of different terrain types."""
    print("\n" + "=" * 70)
    print("🎬 Quick Demo Mode")
    print("=" * 70)
    
    demo_prompts = [
        "mountainous terrain with rivers and valleys",
        "desert landscape with sand dunes",
        "forest terrain with rolling hills"
    ]
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n🎯 Demo {i}/3: {prompt}")
        print("-" * 50)
        
        try:
            # Generate terrain
            heightmap = prompt_to_heightmap_gan(prompt, size=64)
            enhanced = _basic_enhancement(heightmap, prompt)
            
            # Create simple visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Demo {i}: {prompt}', fontsize=14, fontweight='bold')
            
            # Heightmap
            axes[0].imshow(heightmap, cmap='terrain', aspect='equal')
            axes[0].set_title('Heightmap', fontweight='bold')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            axes[0].grid(True, alpha=0.3)
            
            # Enhanced terrain
            if len(enhanced.shape) == 3:
                axes[1].imshow(enhanced)
            else:
                axes[1].imshow(enhanced, cmap='terrain')
            axes[1].set_title('Enhanced Terrain', fontweight='bold')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save demo
            save_path = f"demo_{i}_{prompt.split()[0]}.png"
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
            
            # Show plot briefly
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"❌ Demo {i} failed: {e}")
    
    print("\n🎉 Quick demo completed!")

def main():
    """Main function."""
    print("🎮 Welcome to the Lightweight Interactive Terrain Prototype!")
    print("\nThis version provides:")
    print("✓ Fast terrain generation (no heavy models)")
    print("✓ Interactive prompt input")
    print("✓ GAN + enhancement pipeline")
    print("✓ Multiple visualization options")
    print("✓ Automatic image saving")
    
    while True:
        print("\n" + "=" * 70)
        print("📋 Choose an option:")
        print("1. 🎮 Interactive terrain generation")
        print("2. 🎬 Quick demo (3 terrain types)")
        print("3. 📚 Show pipeline explanation")
        print("4. 🚪 Exit")
        
        try:
            choice = input("\n🎯 Enter your choice (1-4): ").strip()
            
            if choice == "1":
                interactive_terrain_generation()
            elif choice == "2":
                quick_demo()
            elif choice == "3":
                print("\n📚 Running pipeline explanation...")
                import subprocess
                subprocess.run(["python", "pipeline_explanation.py"])
            elif choice == "4":
                print("👋 Thanks for using the terrain generator!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()

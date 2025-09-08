#!/usr/bin/env python3
"""
Demo script for the Generative AI Terrain Prototype
Shows various terrain generation examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from terrain_prototype import (
    prompt_to_heightmap_gan,
    refine_with_diffusion,
    visualize_terrain
)

def demo_mountainous_terrain():
    """Demo: Mountainous terrain with rivers and valleys."""
    print("\n" + "="*60)
    print("DEMO 1: Mountainous Terrain with Rivers and Valleys")
    print("="*60)
    
    prompt = "mountainous terrain with rivers and valleys"
    
    # Generate terrain
    heightmap = prompt_to_heightmap_gan(prompt, size=128)
    
    # Refine (will use basic enhancement if diffusion not available)
    refined_terrain = refine_with_diffusion(heightmap, prompt)
    
    # Visualize
    visualize_terrain(refined_terrain, heightmap, prompt, 
                     save_path="demo_mountainous.png")
    
    return heightmap, refined_terrain

def demo_desert_landscape():
    """Demo: Desert landscape with sand dunes."""
    print("\n" + "="*60)
    print("DEMO 2: Desert Landscape with Sand Dunes")
    print("="*60)
    
    prompt = "desert landscape with sand dunes"
    
    # Generate terrain
    heightmap = prompt_to_heightmap_gan(prompt, size=128)
    
    # Refine
    refined_terrain = refine_with_diffusion(heightmap, prompt)
    
    # Visualize
    visualize_terrain(refined_terrain, heightmap, prompt, 
                     save_path="demo_desert.png")
    
    return heightmap, refined_terrain

def demo_forest_terrain():
    """Demo: Forest terrain with rolling hills."""
    print("\n" + "="*60)
    print("DEMO 3: Forest Terrain with Rolling Hills")
    print("="*60)
    
    prompt = "forest terrain with rolling hills"
    
    # Generate terrain
    heightmap = prompt_to_heightmap_gan(prompt, size=128)
    
    # Refine
    refined_terrain = refine_with_diffusion(heightmap, prompt)
    
    # Visualize
    visualize_terrain(refined_terrain, heightmap, prompt, 
                     save_path="demo_forest.png")
    
    return heightmap, refined_terrain

def demo_coastal_cliffs():
    """Demo: Coastal cliffs with rocky outcrops."""
    print("\n" + "="*60)
    print("DEMO 4: Coastal Cliffs with Rocky Outcrops")
    print("="*60)
    
    prompt = "coastal cliffs with rocky outcrops"
    
    # Generate terrain
    heightmap = prompt_to_heightmap_gan(prompt, size=128)
    
    # Refine
    refined_terrain = refine_with_diffusion(heightmap, prompt)
    
    # Visualize
    visualize_terrain(refined_terrain, heightmap, prompt, 
                     save_path="demo_coastal.png")
    
    return heightmap, refined_terrain

def demo_alpine_landscape():
    """Demo: Alpine landscape with snow-capped peaks."""
    print("\n" + "="*60)
    print("DEMO 5: Alpine Landscape with Snow-capped Peaks")
    print("="*60)
    
    prompt = "alpine landscape with snow-capped peaks"
    
    # Generate terrain
    heightmap = prompt_to_heightmap_gan(prompt, size=128)
    
    # Refine
    refined_terrain = refine_with_diffusion(heightmap, prompt)
    
    # Visualize
    visualize_terrain(refined_terrain, heightmap, prompt, 
                     save_path="demo_alpine.png")
    
    return heightmap, refined_terrain

def run_all_demos():
    """Run all demo examples."""
    print("=" * 80)
    print("Generative AI Terrain Prototype - Demo Suite")
    print("=" * 80)
    print("This demo will generate 5 different terrain types and save them as images.")
    print("Each demo will show:")
    print("- The generated heightmap")
    print("- The refined 2.5D terrain")
    print("- A 3D visualization")
    print("- Save the result as a PNG file")
    
    # Turn off interactive mode for batch processing
    plt.ioff()
    
    demos = [
        demo_mountainous_terrain,
        demo_desert_landscape,
        demo_forest_terrain,
        demo_coastal_cliffs,
        demo_alpine_landscape
    ]
    
    results = []
    
    for i, demo_func in enumerate(demos, 1):
        try:
            print(f"\nRunning Demo {i}/5...")
            heightmap, terrain = demo_func()
            results.append((heightmap, terrain))
            print(f"Demo {i} completed successfully!")
            
        except Exception as e:
            print(f"Demo {i} failed: {e}")
            results.append((None, None))
        
        # Close plots to free memory
        plt.close('all')
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    
    successful_demos = sum(1 for h, t in results if h is not None and t is not None)
    print(f"Successful demos: {successful_demos}/5")
    
    if successful_demos > 0:
        print("\nGenerated files:")
        print("- demo_mountainous.png")
        print("- demo_desert.png")
        print("- demo_forest.png")
        print("- demo_coastal.png")
        print("- demo_alpine.png")
        
        print("\nTo run the interactive prototype:")
        print("python terrain_prototype.py")
    
    print("\nDemo completed!")

def run_single_demo():
    """Run a single demo based on user choice."""
    print("=" * 60)
    print("Single Demo Selection")
    print("=" * 60)
    
    demos = [
        ("Mountainous terrain with rivers and valleys", demo_mountainous_terrain),
        ("Desert landscape with sand dunes", demo_desert_landscape),
        ("Forest terrain with rolling hills", demo_forest_terrain),
        ("Coastal cliffs with rocky outcrops", demo_coastal_cliffs),
        ("Alpine landscape with snow-capped peaks", demo_alpine_landscape)
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    
    print("6. Run all demos")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-6): ").strip()
            
            if choice == "0":
                print("Exiting demo...")
                return
            elif choice == "6":
                run_all_demos()
                return
            elif choice in ["1", "2", "3", "4", "5"]:
                idx = int(choice) - 1
                name, demo_func = demos[idx]
                
                print(f"\nRunning: {name}")
                demo_func()
                return
            else:
                print("Invalid choice. Please enter 0-6.")
                
        except KeyboardInterrupt:
            print("\nDemo interrupted.")
            return
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Welcome to the Terrain Prototype Demo!")
    print("\nChoose an option:")
    print("1. Run a single demo")
    print("2. Run all demos")
    
    while True:
        try:
            choice = input("\nSelect option (1 or 2): ").strip()
            
            if choice == "1":
                run_single_demo()
                break
            elif choice == "2":
                run_all_demos()
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\nDemo interrupted.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

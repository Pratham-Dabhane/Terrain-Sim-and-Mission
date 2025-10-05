#!/usr/bin/env python3
"""
Interactive Photorealistic Terrain Viewer
Launch interactive 3D visualization with rotate, zoom, and pan controls.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
import logging

# Import the pipeline components
try:
    from pipeline.advanced_terrain_renderer import AdvancedTerrainRenderer
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Make sure all pipeline files are in the 'pipeline' directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveTerrainViewer:
    """Interactive viewer for photorealistic 3D terrain visualizations."""
    
    def __init__(self):
        self.renderer = AdvancedTerrainRenderer()
    
    def launch_interactive_viewer(self, session_path):
        """Launch interactive viewer for a specific terrain session."""
        
        session_dir = Path(session_path)
        
        if not session_dir.exists():
            logger.error(f"Session directory not found: {session_dir}")
            return False
        
        # Load session data
        metadata_path = session_dir / "metadata.json"
        heightmap_path = session_dir / "heightmap.png"
        enhanced_path = session_dir / "enhanced_terrain.png"
        
        if not all([metadata_path.exists(), heightmap_path.exists(), enhanced_path.exists()]):
            logger.error("Required files not found in session directory")
            logger.error(f"Need: metadata.json, heightmap.png, enhanced_terrain.png")
            return False
        
        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        prompt = metadata['prompt']
        
        # Load terrain data
        heightmap_img = Image.open(heightmap_path).convert('L')
        heightmap = np.array(heightmap_img).astype(np.float32) / 255.0
        
        enhanced_texture = Image.open(enhanced_path)
        
        logger.info(f"Loaded terrain: '{prompt}'")
        logger.info(f"Heightmap: {heightmap.shape}, Enhanced texture: {enhanced_texture.size}")
        
        # Launch interactive viewer
        try:
            self.renderer.create_interactive_photorealistic_visualization(
                heightmap=heightmap,
                enhanced_texture=enhanced_texture,
                terrain_prompt=prompt
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch interactive viewer: {e}")
            return False
    
    def list_available_sessions(self):
        """List all available terrain sessions."""
        
        output_dir = Path("Output")
        if not output_dir.exists():
            logger.info("No Output directory found")
            return []
        
        sessions = []
        for session_dir in output_dir.glob("session_*"):
            metadata_path = session_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    sessions.append({
                        'path': str(session_dir),
                        'id': session_dir.name,
                        'prompt': metadata.get('prompt', 'Unknown'),
                        'timestamp': metadata.get('generated_at', 'Unknown')
                    })
                except Exception as e:
                    logger.warning(f"Could not read metadata for {session_dir}: {e}")
        
        return sorted(sessions, key=lambda x: x['id'], reverse=True)

def main():
    parser = argparse.ArgumentParser(description='Interactive Photorealistic Terrain Viewer')
    parser.add_argument('--session', type=str, help='Path to specific session directory')
    parser.add_argument('--list', action='store_true', help='List available sessions')
    parser.add_argument('--latest', action='store_true', help='Open latest session')
    
    args = parser.parse_args()
    
    viewer = InteractiveTerrainViewer()
    
    if args.list:
        # List available sessions
        sessions = viewer.list_available_sessions()
        if not sessions:
            print("No terrain sessions found.")
        else:
            print("\\n🏔️  AVAILABLE TERRAIN SESSIONS:")
            print("="*70)
            for i, session in enumerate(sessions, 1):
                print(f"{i:2d}. {session['id']}")
                print(f"    Prompt: '{session['prompt']}'")
                print(f"    Generated: {session['timestamp']}")
                print(f"    Path: {session['path']}")
                print()
            
            print("💡 Usage:")
            print(f"   python {sys.argv[0]} --session Output/session_XXXXXX")
            print(f"   python {sys.argv[0]} --latest")
        return
    
    if args.latest:
        # Open latest session
        sessions = viewer.list_available_sessions()
        if not sessions:
            print("No terrain sessions found.")
            return
        
        latest_session = sessions[0]
        print(f"🎮 Opening latest session: {latest_session['id']}")
        print(f"   Prompt: '{latest_session['prompt']}'")
        
        success = viewer.launch_interactive_viewer(latest_session['path'])
        if success:
            print("✅ Interactive session completed successfully!")
        else:
            print("❌ Failed to launch interactive viewer")
        return
    
    if args.session:
        # Open specific session
        success = viewer.launch_interactive_viewer(args.session)
        if success:
            print("✅ Interactive session completed successfully!")
        else:
            print("❌ Failed to launch interactive viewer")
        return
    
    # No arguments - show help and list sessions
    parser.print_help()
    print()
    
    sessions = viewer.list_available_sessions()
    if sessions:
        print("🏔️  Recent terrain sessions:")
        for session in sessions[:3]:  # Show last 3
            print(f"   • {session['id']}: '{session['prompt']}'")
        print(f"\\n💡 Use --list to see all sessions, --latest for most recent")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import sys
import os
sys.path.append('pipeline')

print("Testing TerrainRemaster import...")

try:
    from pipeline.remaster_sd_controlnet import TerrainRemaster, HAS_DIFFUSERS
    print(f"✅ TerrainRemaster import successful")
    print(f"HAS_DIFFUSERS: {HAS_DIFFUSERS}")
    
    if HAS_DIFFUSERS:
        print("Attempting to create TerrainRemaster...")
        try:
            remaster = TerrainRemaster(
                controlnet_type="depth",
                device="cuda" if __import__('torch').cuda.is_available() else "cpu",
                use_fp16=True,
                enable_cpu_offload=True
            )
            print("✅ TerrainRemaster created successfully")
        except Exception as e:
            print(f"❌ TerrainRemaster creation failed: {e}")
    else:
        print("❌ Diffusers not available")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
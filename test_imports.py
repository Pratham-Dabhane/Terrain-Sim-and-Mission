#!/usr/bin/env python3

import sys
sys.path.append('pipeline')

try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    print('✅ Diffusers import successful')
    HAS_DIFFUSERS = True
except ImportError as e:
    print(f'❌ Diffusers import failed: {e}')
    HAS_DIFFUSERS = False

try:
    from transformers import pipeline
    print('✅ Transformers import successful')
    HAS_TRANSFORMERS = True
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')
    HAS_TRANSFORMERS = False

print(f"Final status: Diffusers={HAS_DIFFUSERS}, Transformers={HAS_TRANSFORMERS}")
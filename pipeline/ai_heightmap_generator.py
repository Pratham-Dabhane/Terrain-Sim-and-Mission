"""
AI Heightmap Generator using Diffusion Models
Optional module that generates heightmaps from text prompts using diffusion models.
Gracefully falls back if diffusers library is not available.
"""

import numpy as np
import logging
from typing import Optional, Union, Tuple
from PIL import Image
import torch

logger = logging.getLogger(__name__)

# Global flag to track diffusers availability
_DIFFUSERS_AVAILABLE = False
_DIFFUSION_PIPELINE = None

# Try to import diffusers with graceful fallback
try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    from diffusers.utils import logging as diffusers_logging
    import transformers
    
    # Suppress excessive logging from diffusers
    diffusers_logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    
    _DIFFUSERS_AVAILABLE = True
    logger.info("✓ Diffusers library available for AI heightmap generation")
    
except ImportError as e:
    logger.warning(f"Diffusers library not available: {e}")
    logger.info("AI heightmap generation will be disabled (fallback to GAN/procedural)")
    _DIFFUSERS_AVAILABLE = False


class AIHeightmapGenerator:
    """
    AI-powered heightmap generator using diffusion models.
    
    Generates grayscale heightmaps from text prompts using Stable Diffusion
    or similar diffusion models. Falls back gracefully if libraries unavailable.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize AI heightmap generator.
        
        Args:
            device (str): Device to use for inference ("cuda" or "cpu")
        """
        self.device = device
        self.pipeline = None
        self.is_available = _DIFFUSERS_AVAILABLE
        
        if self.is_available:
            self._initialize_pipeline()
        else:
            logger.warning("AI heightmap generator disabled - diffusers not available")
    
    def _initialize_pipeline(self):
        """Initialize the diffusion pipeline for heightmap generation."""
        if not self.is_available:
            return
        
        try:
            logger.info("Initializing AI heightmap generation pipeline...")
            
            # Check GPU memory availability
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU memory available: {gpu_memory:.1f} GB")
                
                if gpu_memory < 6.0:  # Less than 6GB GPU memory
                    logger.warning("Low GPU memory detected - using CPU for AI heightmap generation")
                    self.device = "cpu"
            
            # Use a lightweight model for heightmap generation
            model_id = "runwayml/stable-diffusion-v1-5"
            
            # Initialize with memory-efficient settings
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable safety checker for heightmaps
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to("cuda")
                # Enable memory efficient attention if available
                try:
                    self.pipeline.enable_memory_efficient_attention()
                    logger.info("✓ Memory efficient attention enabled")
                except:
                    logger.info("Memory efficient attention not available")
                
                # Enable model CPU offload to save VRAM
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("✓ Model CPU offload enabled")
                except:
                    logger.info("Model CPU offload not available")
            else:
                self.pipeline = self.pipeline.to("cpu")
            
            logger.info(f"✓ AI heightmap generator initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI heightmap generator: {e}")
            self.is_available = False
            self.pipeline = None
    
    def generate_heightmap(
        self, 
        prompt: str,
        size: Tuple[int, int] = (256, 256),
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Generate heightmap from text prompt using diffusion model.
        
        Args:
            prompt (str): Text description of terrain
            size (Tuple[int, int]): Output size (width, height)
            num_inference_steps (int): Number of diffusion steps (fewer = faster)
            guidance_scale (float): How closely to follow the prompt
            seed (Optional[int]): Random seed for reproducibility
            
        Returns:
            Optional[np.ndarray]: Generated heightmap (0-1) or None if failed
        """
        if not self.is_available or self.pipeline is None:
            logger.warning("AI heightmap generation not available - falling back")
            return None
        
        try:
            # Enhance prompt for heightmap generation
            heightmap_prompt = self._enhance_prompt_for_heightmap(prompt)
            logger.info(f"Generating AI heightmap for: '{heightmap_prompt}'")
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            
            # Generate image using diffusion model
            with torch.no_grad():
                result = self.pipeline(
                    prompt=heightmap_prompt,
                    height=size[1],
                    width=size[0],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    output_type="pil"
                )
                
                generated_image = result.images[0]
            
            # Convert to grayscale heightmap
            heightmap = self._image_to_heightmap(generated_image)
            
            logger.info(f"✓ AI heightmap generated: {heightmap.shape}, range [{heightmap.min():.3f}, {heightmap.max():.3f}]")
            return heightmap
            
        except Exception as e:
            logger.error(f"AI heightmap generation failed: {e}")
            return None
    
    def _enhance_prompt_for_heightmap(self, prompt: str) -> str:
        """
        Enhance user prompt for better heightmap generation.
        
        Args:
            prompt (str): Original user prompt
            
        Returns:
            str: Enhanced prompt optimized for heightmap generation
        """
        # Add heightmap-specific keywords to improve generation
        heightmap_keywords = [
            "topographic map", "elevation map", "height field", 
            "grayscale terrain", "black and white landscape",
            "aerial view", "satellite imagery", "contour map"
        ]
        
        # Remove color-related words that might interfere
        color_words = ["red", "blue", "green", "colorful", "bright", "vibrant"]
        enhanced_prompt = prompt.lower()
        
        for word in color_words:
            enhanced_prompt = enhanced_prompt.replace(word, "")
        
        # Add heightmap context
        enhanced_prompt = f"{enhanced_prompt}, topographic elevation map, grayscale heightfield, aerial view"
        
        return enhanced_prompt.strip()
    
    def _image_to_heightmap(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to normalized heightmap array.
        
        Args:
            image (Image.Image): Generated PIL image
            
        Returns:
            np.ndarray: Normalized heightmap (0-1)
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        heightmap = np.array(image, dtype=np.float32)
        
        # Normalize to 0-1 range
        heightmap = heightmap / 255.0
        
        # Apply some enhancement for better terrain characteristics
        heightmap = self._enhance_heightmap(heightmap)
        
        return heightmap
    
    def _enhance_heightmap(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Apply enhancements to make heightmap more terrain-like.
        
        Args:
            heightmap (np.ndarray): Raw heightmap
            
        Returns:
            np.ndarray: Enhanced heightmap
        """
        # Apply slight contrast enhancement
        heightmap = np.clip(heightmap * 1.2 - 0.1, 0, 1)
        
        # Smooth slightly to remove noise
        from scipy import ndimage
        try:
            heightmap = ndimage.gaussian_filter(heightmap, sigma=0.5)
        except:
            # Fallback if scipy not available
            pass
        
        # Ensure proper normalization
        if heightmap.max() > heightmap.min():
            heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        
        return heightmap
    
    def check_memory_requirements(self) -> dict:
        """
        Check memory requirements and availability for AI generation.
        
        Returns:
            dict: Memory information and recommendations
        """
        info = {
            "diffusers_available": self.is_available,
            "pipeline_loaded": self.pipeline is not None,
            "device": self.device,
            "recommended": True
        }
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            
            info.update({
                "gpu_total_gb": gpu_memory,
                "gpu_allocated_gb": gpu_allocated,
                "gpu_reserved_gb": gpu_reserved,
                "gpu_available_gb": gpu_memory - gpu_reserved
            })
            
            # Recommend based on available memory
            if gpu_memory < 6.0:
                info["recommended"] = False
                info["reason"] = "GPU memory < 6GB - may cause OOM errors"
            elif gpu_memory - gpu_reserved < 3.0:
                info["recommended"] = False
                info["reason"] = "Insufficient free GPU memory"
        
        return info
    
    def cleanup(self):
        """Clean up resources to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("AI heightmap generator resources cleaned up")


def create_ai_heightmap_generator(device: str = "cuda") -> Optional[AIHeightmapGenerator]:
    """
    Factory function to create AI heightmap generator with error handling.
    
    Args:
        device (str): Device to use for inference
        
    Returns:
        Optional[AIHeightmapGenerator]: Generator instance or None if unavailable
    """
    if not _DIFFUSERS_AVAILABLE:
        logger.info("AI heightmap generation not available - diffusers library missing")
        return None
    
    try:
        generator = AIHeightmapGenerator(device=device)
        if generator.is_available:
            return generator
        else:
            return None
    except Exception as e:
        logger.error(f"Failed to create AI heightmap generator: {e}")
        return None


# Convenience function for quick checks
def is_ai_heightmap_available() -> bool:
    """Check if AI heightmap generation is available."""
    return _DIFFUSERS_AVAILABLE


def get_ai_requirements() -> list:
    """Get list of required packages for AI heightmap generation."""
    return [
        "diffusers>=0.21.0",
        "transformers>=4.25.0", 
        "accelerate>=0.20.0",
        "scipy>=1.9.0"  # For heightmap enhancement
    ]
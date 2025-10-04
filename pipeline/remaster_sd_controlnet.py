"""
Stable Diffusion + ControlNet Remastering Pipeline
Enhances GAN-generated heightmaps with realistic textures while preserving geometry.
Optimized for low VRAM (GTX 1650, 4GB) with memory-efficient settings.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Union, Optional, Tuple
import logging
import gc

# Diffusers imports
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
    from diffusers.utils import load_image
    import xformers
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    logging.warning("Diffusers not available. Install with: pip install diffusers")

# Transformers imports
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

class MemoryEfficientSD:
    """
    Memory-efficient Stable Diffusion wrapper for low VRAM systems.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-depth",
        device: str = "cuda",
        enable_memory_efficient_attention: bool = True,
        enable_cpu_offload: bool = True,
        use_fp16: bool = True
    ):
        self.device = device
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        if not HAS_DIFFUSERS:
            raise ImportError("Diffusers is required. Install with: pip install diffusers")
        
        # Load ControlNet
        logger.info(f"Loading ControlNet: {controlnet_id}")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=self.dtype,
            use_safetensors=True
        )
        
        # Load Stable Diffusion pipeline
        logger.info(f"Loading Stable Diffusion: {model_id}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if use_fp16 else None
        )
        
        # Memory optimizations
        if enable_memory_efficient_attention:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except:
                logger.warning("xformers not available, using default attention")
        
        # CPU offloading for low VRAM
        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            logger.info("Enabled CPU offloading")
        else:
            self.pipe = self.pipe.to(device)
        
        # Use memory efficient scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        # Enable attention slicing for lower memory usage
        self.pipe.enable_attention_slicing(1)
        
        logger.info("Stable Diffusion pipeline initialized with memory optimizations")
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        negative_prompt: str = "blurry, low quality, distorted, artifacts",
        num_inference_steps: int = 20,  # Reduced for speed
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate image with ControlNet conditioning.
        
        Args:
            prompt: Text prompt for generation
            control_image: ControlNet conditioning image
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            controlnet_conditioning_scale: ControlNet conditioning strength
            width: Output width
            height: Output height
            seed: Random seed
            
        Returns:
            PIL.Image: Generated image
        """
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Clear memory before generation
        self.clear_memory()
        
        try:
            # Generate
            with torch.autocast(self.device, dtype=self.dtype):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    return_dict=False
                )[0]
            
            return result
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM during generation, trying with lower resolution...")
                self.clear_memory()
                
                # Try with lower resolution
                return self.generate(
                    prompt=prompt,
                    control_image=control_image.resize((width//2, height//2)),
                    negative_prompt=negative_prompt,
                    num_inference_steps=max(10, num_inference_steps//2),
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    width=width//2,
                    height=height//2,
                    seed=seed
                )
            else:
                raise e


class HeightmapProcessor:
    """
    Heightmap preprocessing utilities for ControlNet.
    """
    
    @staticmethod
    def heightmap_to_depth(heightmap: np.ndarray) -> np.ndarray:
        """
        Convert heightmap to depth map for ControlNet.
        
        Args:
            heightmap: Heightmap array (H, W) or (H, W, 1)
            
        Returns:
            np.ndarray: Depth map for ControlNet
        """
        if len(heightmap.shape) == 3:
            heightmap = heightmap.squeeze()
        
        # Normalize to [0, 1]
        if heightmap.max() > 1.0:
            heightmap = heightmap / 255.0
        
        # Apply gamma correction to enhance depth perception
        depth = np.power(heightmap, 0.7)
        
        # Convert to 3-channel image
        depth_rgb = np.stack([depth] * 3, axis=-1)
        depth_rgb = (depth_rgb * 255).astype(np.uint8)
        
        return depth_rgb
    
    @staticmethod
    def heightmap_to_normal(heightmap: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Convert heightmap to normal map for ControlNet.
        
        Args:
            heightmap: Heightmap array (H, W)
            strength: Normal map strength
            
        Returns:
            np.ndarray: Normal map RGB image
        """
        if len(heightmap.shape) == 3:
            heightmap = heightmap.squeeze()
        
        # Calculate gradients
        grad_x = cv2.Scharr(heightmap, cv2.CV_64F, 1, 0) * strength
        grad_y = cv2.Scharr(heightmap, cv2.CV_64F, 0, 1) * strength
        
        # Calculate normals
        normal_x = -grad_x
        normal_y = -grad_y
        normal_z = np.ones_like(heightmap)
        
        # Normalize
        length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= length
        normal_y /= length
        normal_z /= length
        
        # Convert to [0, 255] RGB
        normal_map = np.zeros((heightmap.shape[0], heightmap.shape[1], 3), dtype=np.uint8)
        normal_map[:, :, 0] = ((normal_x + 1) * 127.5).astype(np.uint8)  # R
        normal_map[:, :, 1] = ((normal_y + 1) * 127.5).astype(np.uint8)  # G
        normal_map[:, :, 2] = ((normal_z + 1) * 127.5).astype(np.uint8)  # B
        
        return normal_map
    
    @staticmethod
    def prepare_control_image(
        heightmap: np.ndarray, 
        control_type: str = "depth",
        target_size: Tuple[int, int] = (512, 512)
    ) -> Image.Image:
        """
        Prepare control image for ControlNet.
        
        Args:
            heightmap: Input heightmap
            control_type: Type of control ('depth' or 'normal')
            target_size: Target image size
            
        Returns:
            PIL.Image: Control image
        """
        if control_type == "depth":
            control_array = HeightmapProcessor.heightmap_to_depth(heightmap)
        elif control_type == "normal":
            control_array = HeightmapProcessor.heightmap_to_normal(heightmap)
        else:
            raise ValueError(f"Unknown control type: {control_type}")
        
        # Convert to PIL and resize
        control_image = Image.fromarray(control_array)
        control_image = control_image.resize(target_size, Image.Resampling.LANCZOS)
        
        return control_image


class TerrainRemaster:
    """
    Main terrain remastering class that combines heightmap processing with Stable Diffusion.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_type: str = "depth",  # "depth" or "normal"
        device: str = "cuda",
        use_fp16: bool = True,
        enable_cpu_offload: bool = True
    ):
        self.controlnet_type = controlnet_type
        self.device = device
        
        # ControlNet model mapping
        controlnet_models = {
            "depth": "lllyasviel/sd-controlnet-depth",
            "normal": "lllyasviel/sd-controlnet-normal",
        }
        
        if controlnet_type not in controlnet_models:
            raise ValueError(f"Unsupported ControlNet type: {controlnet_type}")
        
        controlnet_id = controlnet_models[controlnet_type]
        
        # Initialize Stable Diffusion pipeline
        self.sd_pipeline = MemoryEfficientSD(
            model_id=model_id,
            controlnet_id=controlnet_id,
            device=device,
            use_fp16=use_fp16,
            enable_cpu_offload=enable_cpu_offload
        )
        
        self.processor = HeightmapProcessor()
        
        logger.info(f"TerrainRemaster initialized with {controlnet_type} ControlNet")
    
    def enhance_prompt_for_terrain(self, prompt: str) -> str:
        """
        Enhance prompt for better terrain generation.
        
        Args:
            prompt: Original prompt
            
        Returns:
            str: Enhanced prompt
        """
        # Add terrain-specific style keywords
        style_keywords = [
            "highly detailed",
            "realistic",
            "natural landscape",
            "photorealistic",
            "sharp focus",
            "8k resolution",
            "professional photography"
        ]
        
        enhanced = f"{prompt}, {', '.join(style_keywords)}"
        return enhanced
    
    def remaster_heightmap(
        self,
        heightmap: Union[np.ndarray, torch.Tensor, Image.Image],
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, artifacts, unrealistic, cartoon",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,  # Lower for geometry preservation
        output_size: Tuple[int, int] = (512, 512),
        seed: Optional[int] = None,
        preserve_geometry: bool = True
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Remaster a heightmap with Stable Diffusion + ControlNet.
        
        Args:
            heightmap: Input heightmap
            prompt: Text description for remastering
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            controlnet_conditioning_scale: ControlNet strength
            output_size: Output image size
            seed: Random seed
            preserve_geometry: Whether to preserve original geometry
            
        Returns:
            Tuple[PIL.Image, np.ndarray]: (remastered_image, preserved_heightmap)
        """
        # Convert input to numpy array
        if isinstance(heightmap, torch.Tensor):
            heightmap = heightmap.cpu().numpy()
        elif isinstance(heightmap, Image.Image):
            heightmap = np.array(heightmap)
        
        # Ensure 2D heightmap
        if len(heightmap.shape) == 3:
            heightmap = heightmap.squeeze()
        if len(heightmap.shape) == 4:  # (B, C, H, W)
            heightmap = heightmap[0, 0]
        
        # Normalize heightmap to [0, 1]
        if heightmap.max() > 1.0:
            heightmap = heightmap / heightmap.max()
        
        # Prepare control image
        control_image = self.processor.prepare_control_image(
            heightmap, 
            control_type=self.controlnet_type,
            target_size=output_size
        )
        
        # Enhance prompt
        enhanced_prompt = self.enhance_prompt_for_terrain(prompt)
        
        # Generate remastered image
        remastered_image = self.sd_pipeline.generate(
            prompt=enhanced_prompt,
            control_image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=output_size[0],
            height=output_size[1],
            seed=seed
        )
        
        # Preserve original geometry if requested
        if preserve_geometry:
            # Resize original heightmap to match output
            preserved_heightmap = cv2.resize(heightmap, output_size, interpolation=cv2.INTER_LINEAR)
        else:
            # Extract heightmap from remastered image (simplified approach)
            gray = np.array(remastered_image.convert('L')) / 255.0
            preserved_heightmap = gray
        
        return remastered_image, preserved_heightmap
    
    def batch_remaster(
        self,
        heightmaps: list,
        prompts: list,
        **kwargs
    ) -> list:
        """
        Remaster multiple heightmaps in batch.
        
        Args:
            heightmaps: List of heightmaps
            prompts: List of prompts
            **kwargs: Additional arguments for remaster_heightmap
            
        Returns:
            list: List of (remastered_image, heightmap) tuples
        """
        results = []
        
        for i, (heightmap, prompt) in enumerate(zip(heightmaps, prompts)):
            logger.info(f"Remastering {i+1}/{len(heightmaps)}: {prompt[:50]}...")
            
            try:
                result = self.remaster_heightmap(heightmap, prompt, **kwargs)
                results.append(result)
                
                # Clear memory between generations
                self.sd_pipeline.clear_memory()
                
            except Exception as e:
                logger.error(f"Failed to remaster heightmap {i}: {e}")
                # Create fallback result
                fallback_image = Image.fromarray((heightmap * 255).astype(np.uint8), mode='L')
                fallback_image = fallback_image.convert('RGB').resize(kwargs.get('output_size', (512, 512)))
                results.append((fallback_image, heightmap))
        
        return results


def test_remastering():
    """Test function for terrain remastering"""
    if not HAS_DIFFUSERS:
        print("Diffusers not available. Skipping test.")
        return
    
    try:
        # Create test heightmap
        size = 256
        x = np.linspace(0, 4*np.pi, size)
        y = np.linspace(0, 4*np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate simple terrain
        heightmap = 0.5 * (np.sin(X/2) * np.cos(Y/2) + 1)
        heightmap += 0.3 * np.random.random((size, size))  # Add noise
        heightmap = np.clip(heightmap, 0, 1)
        
        # Initialize remaster
        remaster = TerrainRemaster(
            controlnet_type="depth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Test remastering
        prompt = "mountainous terrain with rocky peaks and green valleys"
        remastered_image, preserved_heightmap = remaster.remaster_heightmap(
            heightmap=heightmap,
            prompt=prompt,
            num_inference_steps=10,  # Fast test
            output_size=(512, 512)
        )
        
        print(f"Remastered image size: {remastered_image.size}")
        print(f"Preserved heightmap shape: {preserved_heightmap.shape}")
        print("Terrain remastering test passed!")
        
        # Save test results
        remastered_image.save("test_remastered.png")
        heightmap_img = Image.fromarray((preserved_heightmap * 255).astype(np.uint8), mode='L')
        heightmap_img.save("test_heightmap.png")
        
    except Exception as e:
        print(f"Terrain remastering test failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_remastering()
"""
Pipeline Configuration System
Defines feature flags and settings for the terrain generation pipeline.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """
    Configuration dataclass for terrain generation pipeline.
    
    Feature flags control which components are enabled/disabled.
    All flags default to True for existing functionality except experimental features.
    """
    
    # =============================================================================
    # 🚩 FEATURE FLAGS
    # =============================================================================
    
    # Core pipeline features (default: enabled to maintain existing behavior)
    procedural_enabled: bool = True            # Enable procedural terrain generation
    prompt_parser_enabled: bool = True         # Enable advanced prompt parsing
    texture_mapping_enabled: bool = True       # Enable realistic texture mapping
    photorealistic_render_enabled: bool = True # Enable photorealistic 3D rendering
    
    # Experimental/future features (default: disabled)
    ai_heightmap_enabled: bool = False         # Enable AI-enhanced heightmap processing
    mission_sim_enabled: bool = False          # Enable mission simulation features
    
    # =============================================================================
    # 📐 SIZE CONFIGURATIONS
    # =============================================================================
    
    # Heightmap generation settings
    base_heightmap_size: Tuple[int, int] = (256, 256)    # Base heightmap resolution
    output_size: Tuple[int, int] = (512, 512)            # Final output resolution
    
    # =============================================================================
    # ⚙️ PROCESSING SETTINGS
    # =============================================================================
    
    # Performance settings
    batch_size: int = 1                        # Batch processing size
    use_gpu_acceleration: bool = True          # Enable GPU acceleration when available
    memory_efficient_mode: bool = False       # Enable memory-efficient processing
    
    # Quality settings
    mesh_quality: str = "high"                 # Mesh generation quality: "low", "medium", "high"
    texture_resolution: int = 512              # Texture map resolution
    
    # =============================================================================
    # 🔧 VALIDATION & UTILITY METHODS
    # =============================================================================
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        logger.info(f"PipelineConfig initialized with {self._count_enabled_features()} features enabled")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate heightmap size
        if not (isinstance(self.base_heightmap_size, tuple) and len(self.base_heightmap_size) == 2):
            raise ValueError("base_heightmap_size must be a tuple of (width, height)")
        
        if not all(isinstance(x, int) and x > 0 for x in self.base_heightmap_size):
            raise ValueError("base_heightmap_size values must be positive integers")
        
        # Validate output size
        if not (isinstance(self.output_size, tuple) and len(self.output_size) == 2):
            raise ValueError("output_size must be a tuple of (width, height)")
        
        if not all(isinstance(x, int) and x > 0 for x in self.output_size):
            raise ValueError("output_size values must be positive integers")
        
        # Validate mesh quality
        valid_qualities = ["low", "medium", "high"]
        if self.mesh_quality not in valid_qualities:
            raise ValueError(f"mesh_quality must be one of {valid_qualities}")
        
        # Validate texture resolution
        if not isinstance(self.texture_resolution, int) or self.texture_resolution <= 0:
            raise ValueError("texture_resolution must be a positive integer")
    
    def _count_enabled_features(self) -> int:
        """Count number of enabled features."""
        feature_flags = [
            self.procedural_enabled,
            self.prompt_parser_enabled, 
            self.texture_mapping_enabled,
            self.photorealistic_render_enabled,
            self.ai_heightmap_enabled,
            self.mission_sim_enabled
        ]
        return sum(feature_flags)
    
    def get_feature_summary(self) -> dict:
        """Get summary of enabled/disabled features."""
        return {
            "procedural_enabled": self.procedural_enabled,
            "prompt_parser_enabled": self.prompt_parser_enabled,
            "texture_mapping_enabled": self.texture_mapping_enabled,
            "photorealistic_render_enabled": self.photorealistic_render_enabled,
            "ai_heightmap_enabled": self.ai_heightmap_enabled,
            "mission_sim_enabled": self.mission_sim_enabled,
            "total_enabled": self._count_enabled_features()
        }
    
    def is_experimental_mode(self) -> bool:
        """Check if any experimental features are enabled."""
        return self.ai_heightmap_enabled or self.mission_sim_enabled
    
    @classmethod
    def create_minimal_config(cls) -> 'PipelineConfig':
        """Create minimal configuration with only essential features enabled."""
        return cls(
            procedural_enabled=True,
            prompt_parser_enabled=False,
            texture_mapping_enabled=False,
            photorealistic_render_enabled=False,
            ai_heightmap_enabled=False,
            mission_sim_enabled=False,
            mesh_quality="low"
        )
    
    @classmethod
    def create_full_config(cls) -> 'PipelineConfig':
        """Create full configuration with all features enabled."""
        return cls(
            procedural_enabled=True,
            prompt_parser_enabled=True,
            texture_mapping_enabled=True,
            photorealistic_render_enabled=True,
            ai_heightmap_enabled=True,
            mission_sim_enabled=True,
            mesh_quality="high"
        )

# =============================================================================
# 🎯 DEFAULT CONFIGURATION INSTANCE
# =============================================================================

# Default configuration for backward compatibility
DEFAULT_CONFIG = PipelineConfig()

def get_default_config() -> PipelineConfig:
    """Get the default pipeline configuration."""
    return DEFAULT_CONFIG
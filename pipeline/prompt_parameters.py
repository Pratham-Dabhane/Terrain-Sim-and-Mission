"""
Prompt-to-Parameter Layer for Terrain Generation
Converts text prompts into deterministic terrain generation parameters.
"""

import re
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Import existing prompt parsing utilities
try:
    from .prompt_parser import (
        KEYWORD_MAP, 
        parse_prompt_to_params, 
        get_terrain_style_from_prompt,
        analyze_prompt_complexity
    )
except ImportError:
    # Fallback for direct execution
    from prompt_parser import (
        KEYWORD_MAP,
        parse_prompt_to_params,
        get_terrain_style_from_prompt, 
        analyze_prompt_complexity
    )


@dataclass
class TerrainParameters:
    """
    Comprehensive terrain generation parameters extracted from prompt.
    """
    # Core terrain characteristics
    elevation_scale: float = 1.0        # Overall height scaling (0.1 - 3.0)
    roughness: float = 0.5              # Terrain roughness/detail (0.0 - 1.0)
    water_level: float = 0.2            # Water/low-elevation threshold (0.0 - 0.5)
    biome_type: str = "mountain"        # Primary biome classification
    
    # Procedural noise parameters
    scale: float = 100.0                # Noise scale
    octaves: int = 6                    # Detail levels
    persistence: float = 0.5            # Amplitude decay per octave
    lacunarity: float = 2.0             # Frequency multiplier per octave
    
    # Terrain mixing weights
    mountain_weight: float = 0.8        # Mountain feature strength
    valley_weight: float = 0.2          # Valley feature strength
    river_strength: float = 0.3         # River carving strength
    river_frequency: float = 0.05       # River pattern frequency
    
    # Mesh/rendering parameters
    mesh_z_scale: float = 20.0          # Z-axis scaling for 3D mesh
    
    # Resolution
    heightmap_size: Tuple[int, int] = (512, 512)  # Generation resolution
    
    # Deterministic seed
    seed: int = 42                      # Random seed for reproducibility
    
    # Style hints
    terrain_style: str = "mixed"        # Overall style classification
    complexity: str = "medium"          # Complexity level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy passing to functions."""
        return asdict(self)


def generate_deterministic_seed(prompt: str) -> int:
    """
    Generate deterministic seed from prompt text.
    
    Args:
        prompt: Input text prompt
        
    Returns:
        int: Deterministic seed in range [0, 1e9)
    """
    # Use SHA-256 hash for consistency
    hash_object = hashlib.sha256(prompt.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    # Convert first 8 hex chars to integer and mod by 1e9
    hash_int = int(hash_hex[:8], 16)
    seed = hash_int % int(1e9)
    
    logger.debug(f"Generated deterministic seed {seed} from prompt")
    return seed


def extract_intensity_modifiers(prompt: str) -> Dict[str, float]:
    """
    Extract intensity/scale modifiers from prompt text.
    
    Args:
        prompt: Input text prompt
        
    Returns:
        dict: Intensity modifiers for various parameters
    """
    prompt_lower = prompt.lower()
    modifiers = {}
    
    # Elevation scale modifiers
    if any(word in prompt_lower for word in ['extreme', 'dramatic', 'towering', 'massive']):
        modifiers['elevation_scale'] = 2.0
        modifiers['mesh_z_scale'] = 40.0
    elif any(word in prompt_lower for word in ['high', 'tall', 'steep']):
        modifiers['elevation_scale'] = 1.5
        modifiers['mesh_z_scale'] = 30.0
    elif any(word in prompt_lower for word in ['low', 'flat', 'gentle', 'subtle']):
        modifiers['elevation_scale'] = 0.5
        modifiers['mesh_z_scale'] = 10.0
    
    # Roughness modifiers
    if any(word in prompt_lower for word in ['rugged', 'jagged', 'rough', 'rocky', 'craggy']):
        modifiers['roughness'] = 0.8
        modifiers['persistence'] = 0.7
        modifiers['octaves'] = 10
    elif any(word in prompt_lower for word in ['smooth', 'gentle', 'rolling', 'soft']):
        modifiers['roughness'] = 0.3
        modifiers['persistence'] = 0.3
        modifiers['octaves'] = 4
    
    # Water level modifiers
    if any(word in prompt_lower for word in ['ocean', 'sea', 'lake', 'coastal', 'island']):
        modifiers['water_level'] = 0.4
    elif any(word in prompt_lower for word in ['river', 'stream', 'creek']):
        modifiers['water_level'] = 0.2
        modifiers['river_strength'] = 0.5
    elif any(word in prompt_lower for word in ['desert', 'arid', 'dry']):
        modifiers['water_level'] = 0.1
        modifiers['river_strength'] = 0.1
    
    # Scale/detail modifiers
    if any(word in prompt_lower for word in ['detailed', 'complex', 'intricate']):
        modifiers['octaves'] = 9
        modifiers['scale'] = 80.0
    elif any(word in prompt_lower for word in ['simple', 'basic']):
        modifiers['octaves'] = 4
        modifiers['scale'] = 150.0
    
    # Resolution hints
    if any(word in prompt_lower for word in ['high-resolution', 'high-res', 'detailed', 'hires']):
        modifiers['heightmap_size'] = (1024, 1024)
    elif any(word in prompt_lower for word in ['low-resolution', 'low-res', 'fast', 'preview']):
        modifiers['heightmap_size'] = (512, 512)
    
    return modifiers


def extract_biome_type(prompt: str) -> str:
    """
    Extract primary biome type from prompt.
    
    Args:
        prompt: Input text prompt
        
    Returns:
        str: Biome type classification
    """
    prompt_lower = prompt.lower()
    
    # Priority order - more specific first
    if any(word in prompt_lower for word in ['snow', 'ice', 'frozen', 'arctic', 'glacier', 'tundra']):
        return 'arctic'
    elif any(word in prompt_lower for word in ['desert', 'sand', 'dune', 'arid', 'sahara']):
        return 'desert'
    elif any(word in prompt_lower for word in ['forest', 'jungle', 'woods', 'trees', 'woodland']):
        return 'forest'
    elif any(word in prompt_lower for word in ['mountain', 'peak', 'summit', 'alpine', 'highland']):
        return 'mountain'
    elif any(word in prompt_lower for word in ['volcano', 'volcanic', 'lava', 'crater']):
        return 'volcanic'
    elif any(word in prompt_lower for word in ['grass', 'meadow', 'prairie', 'plains', 'savanna']):
        return 'grassland'
    elif any(word in prompt_lower for word in ['swamp', 'marsh', 'wetland', 'bog']):
        return 'wetland'
    elif any(word in prompt_lower for word in ['coast', 'beach', 'cliff', 'shore']):
        return 'coastal'
    else:
        return 'mountain'  # Default


def parse_prompt(prompt: str, override_seed: Optional[int] = None) -> TerrainParameters:
    """
    Main function: Parse text prompt into comprehensive terrain parameters.
    
    This function combines multiple analysis techniques:
    1. Deterministic seed generation from prompt hash
    2. Keyword-based parameter extraction (from existing prompt_parser)
    3. Intensity modifier detection
    4. Biome classification
    5. Complexity analysis
    
    Args:
        prompt: Text description of desired terrain
        override_seed: Optional seed to override deterministic generation
        
    Returns:
        TerrainParameters: Complete parameter set for terrain generation
    """
    logger.info(f"Parsing prompt: '{prompt}'")
    
    # Start with default parameters
    params = TerrainParameters()
    
    # Generate deterministic seed from prompt (unless overridden)
    if override_seed is not None:
        params.seed = override_seed
        logger.info(f"Using override seed: {override_seed}")
    else:
        params.seed = generate_deterministic_seed(prompt)
        logger.info(f"Generated deterministic seed: {params.seed}")
    
    # Extract biome type
    params.biome_type = extract_biome_type(prompt)
    logger.info(f"Detected biome: {params.biome_type}")
    
    # Get terrain style and complexity from existing analyzer
    params.terrain_style = get_terrain_style_from_prompt(prompt)
    complexity_info = analyze_prompt_complexity(prompt)
    params.complexity = complexity_info['complexity']
    
    # Use existing keyword parser for base parameters
    keyword_params = parse_prompt_to_params(prompt)
    
    # Apply keyword-based parameters
    if 'mountain_octaves' in keyword_params:
        params.octaves = keyword_params['mountain_octaves']
    if 'scale' in keyword_params:
        params.scale = keyword_params['scale'] * 20.0  # Scale up for better terrain
    if 'persistence' in keyword_params:
        params.persistence = keyword_params['persistence']
    if 'lacunarity' in keyword_params:
        params.lacunarity = keyword_params['lacunarity']
    if 'mountain_weight' in keyword_params:
        params.mountain_weight = keyword_params['mountain_weight']
    if 'valley_weight' in keyword_params:
        params.valley_weight = keyword_params['valley_weight']
    if 'river_strength' in keyword_params:
        params.river_strength = keyword_params['river_strength']
    if 'river_freq' in keyword_params:
        params.river_frequency = keyword_params['river_freq']
    
    # Extract and apply intensity modifiers (these override keyword params)
    intensity_mods = extract_intensity_modifiers(prompt)
    for key, value in intensity_mods.items():
        if hasattr(params, key):
            setattr(params, key, value)
            logger.debug(f"Applied intensity modifier: {key} = {value}")
    
    # Apply complexity-based adjustments
    if params.complexity == 'high':
        params.octaves = max(params.octaves, complexity_info['suggested_octaves'])
        params.roughness = min(params.roughness + 0.2, 1.0)
    elif params.complexity == 'low':
        params.octaves = min(params.octaves, complexity_info['suggested_octaves'])
        params.roughness = max(params.roughness - 0.2, 0.0)
    
    # Biome-specific adjustments
    if params.biome_type == 'desert':
        params.water_level = 0.1
        params.river_strength = 0.1
    elif params.biome_type == 'arctic':
        params.water_level = 0.3
        params.roughness = max(params.roughness, 0.6)
    elif params.biome_type == 'forest':
        params.mountain_weight = 0.6
        params.valley_weight = 0.4
    elif params.biome_type == 'coastal':
        params.water_level = 0.4
        params.elevation_scale = 0.7
    
    # Log final parameters
    logger.info(f"Terrain parameters: style={params.terrain_style}, complexity={params.complexity}")
    logger.info(f"  elevation_scale={params.elevation_scale:.2f}, roughness={params.roughness:.2f}")
    logger.info(f"  octaves={params.octaves}, scale={params.scale:.1f}")
    logger.info(f"  water_level={params.water_level:.2f}, biome={params.biome_type}")
    
    return params


def get_parameters_dict(prompt: str, override_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function that returns parameters as a plain dictionary.
    
    Args:
        prompt: Text description of desired terrain
        override_seed: Optional seed to override deterministic generation
        
    Returns:
        dict: Terrain parameters as dictionary
    """
    params = parse_prompt(prompt, override_seed)
    return params.to_dict()


# Backward compatibility functions for existing code
def get_seed_from_prompt(prompt: str) -> int:
    """Get deterministic seed for a prompt."""
    return generate_deterministic_seed(prompt)


def get_procedural_params_from_prompt(prompt: str) -> Dict[str, Any]:
    """
    Extract procedural generation parameters from prompt.
    Compatible with procedural_noise_utils.NoiseParams.
    """
    params = parse_prompt(prompt)
    return {
        'scale': params.scale,
        'octaves': params.octaves,
        'persistence': params.persistence,
        'lacunarity': params.lacunarity,
        'mountain_weight': params.mountain_weight,
        'valley_weight': params.valley_weight,
        'river_strength': params.river_strength,
        'river_frequency': params.river_frequency,
        'seed': params.seed
    }


if __name__ == "__main__":
    # Test the prompt parser
    logging.basicConfig(level=logging.INFO)
    
    test_prompts = [
        "dramatic snow-covered mountain peaks with deep valleys",
        "gentle rolling hills with meadows",
        "rugged desert landscape with sand dunes",
        "smooth coastal terrain with beaches",
        "extreme jagged rocky peaks with steep cliffs",
        "complex forest terrain with rivers and lakes"
    ]
    
    print("=" * 80)
    print("PROMPT-TO-PARAMETER PARSER TEST")
    print("=" * 80)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 80)
        
        params = parse_prompt(prompt)
        
        print(f"Seed: {params.seed}")
        print(f"Biome: {params.biome_type}")
        print(f"Style: {params.terrain_style}")
        print(f"Complexity: {params.complexity}")
        print(f"Elevation Scale: {params.elevation_scale:.2f}")
        print(f"Roughness: {params.roughness:.2f}")
        print(f"Octaves: {params.octaves}")
        print(f"Water Level: {params.water_level:.2f}")
        print(f"Mesh Z-Scale: {params.mesh_z_scale:.1f}")
        
        # Test deterministic seed generation
        seed1 = generate_deterministic_seed(prompt)
        seed2 = generate_deterministic_seed(prompt)
        assert seed1 == seed2, "Seeds should be deterministic!"
        print(f"✓ Deterministic seed verified: {seed1}")

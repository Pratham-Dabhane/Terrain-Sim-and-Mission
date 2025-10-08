"""
Lightweight Prompt Parser for Terrain Generation
Maps text keywords to procedural generation parameters.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Keyword mapping to procedural parameters
KEYWORD_MAP = {
    "mountain": {"mountain_octaves": 8, "scale": 6.0},
    "river": {"river_strength": 0.3, "river_freq": 0.03},
    "plain": {"mountain_octaves": 3, "scale": 2.5},
    "desert": {"texture": "sand", "mountain_octaves": 3},
    "forest": {"texture": "forest", "valley_octaves": 4},
    "hill": {"mountain_octaves": 5, "scale": 4.0},
    "valley": {"valley_octaves": 6, "mountain_weight": 0.5, "valley_weight": 0.5},
    "plateau": {"mountain_octaves": 4, "scale": 8.0, "persistence": 0.3},
    "rugged": {"mountain_octaves": 10, "persistence": 0.7, "lacunarity": 2.5},
    "smooth": {"mountain_octaves": 3, "persistence": 0.3, "lacunarity": 1.8},
    "steep": {"mountain_octaves": 8, "persistence": 0.6, "lacunarity": 2.8},
    "rolling": {"mountain_octaves": 4, "persistence": 0.4, "scale": 5.0},
    "canyon": {"valley_octaves": 8, "river_strength": 0.5, "valley_weight": 0.7},
    "meadow": {"mountain_octaves": 2, "scale": 3.0, "persistence": 0.2},
    "highland": {"mountain_octaves": 7, "scale": 7.0, "mountain_weight": 0.9},
    "lowland": {"mountain_octaves": 2, "scale": 2.0, "valley_weight": 0.8},
    "jagged": {"mountain_octaves": 12, "persistence": 0.8, "lacunarity": 3.0},
    "gentle": {"mountain_octaves": 3, "persistence": 0.3, "scale": 4.0},
    "dramatic": {"mountain_octaves": 9, "persistence": 0.7, "scale": 8.0},
    "subtle": {"mountain_octaves": 2, "persistence": 0.2, "scale": 2.0}
}

def parse_prompt_to_params(prompt: str) -> dict:
    """
    Parse text prompt and extract procedural generation parameters.
    
    Args:
        prompt: Text description of terrain
        
    Returns:
        dict: Extracted parameters based on keywords found
    """
    out = {}
    prompt_lower = prompt.lower()
    
    # Track which keywords were found for logging
    found_keywords = []
    
    for keyword, params in KEYWORD_MAP.items():
        if keyword in prompt_lower:
            out.update(params)
            found_keywords.append(keyword)
    
    if found_keywords:
        logger.info(f"Prompt keywords detected: {found_keywords}")
    else:
        # If no keywords found, use default mixed terrain
        out = {"mountain_octaves": 6, "scale": 5.0, "persistence": 0.5}
        logger.info("No specific keywords detected, using default mixed terrain")
    
    return out

def get_terrain_style_from_prompt(prompt: str) -> str:
    """
    Determine overall terrain style from prompt.
    
    Args:
        prompt: Text description of terrain
        
    Returns:
        str: Terrain style ('mountainous', 'hills', 'plains', 'rugged', 'mixed')
    """
    prompt_lower = prompt.lower()
    
    # Priority order - more specific terms first
    if any(word in prompt_lower for word in ['mountain', 'peak', 'summit', 'highland']):
        return 'mountainous'
    elif any(word in prompt_lower for word in ['rugged', 'jagged', 'rough', 'rocky']):
        return 'rugged'
    elif any(word in prompt_lower for word in ['hill', 'rolling', 'gentle']):
        return 'hills'
    elif any(word in prompt_lower for word in ['plain', 'flat', 'meadow', 'lowland']):
        return 'plains'
    else:
        return 'mixed'

def analyze_prompt_complexity(prompt: str) -> Dict[str, Any]:
    """
    Analyze prompt complexity and suggest generation parameters.
    
    Args:
        prompt: Text description of terrain
        
    Returns:
        dict: Analysis results with complexity metrics
    """
    prompt_lower = prompt.lower()
    
    # Count terrain features
    terrain_features = ['mountain', 'hill', 'valley', 'river', 'canyon', 'plateau']
    feature_count = sum(1 for feature in terrain_features if feature in prompt_lower)
    
    # Count descriptive adjectives
    descriptors = ['rugged', 'smooth', 'steep', 'gentle', 'dramatic', 'rolling']
    descriptor_count = sum(1 for desc in descriptors if desc in prompt_lower)
    
    # Determine complexity level
    total_complexity = feature_count + descriptor_count
    
    if total_complexity >= 4:
        complexity = 'high'
        suggested_octaves = 10
    elif total_complexity >= 2:
        complexity = 'medium'
        suggested_octaves = 6
    else:
        complexity = 'low'
        suggested_octaves = 4
    
    analysis = {
        'complexity': complexity,
        'feature_count': feature_count,
        'descriptor_count': descriptor_count,
        'suggested_octaves': suggested_octaves,
        'terrain_style': get_terrain_style_from_prompt(prompt)
    }
    
    logger.debug(f"Prompt analysis: {analysis}")
    return analysis

def merge_params_with_defaults(parsed_params: dict, default_params: dict) -> dict:
    """
    Merge parsed parameters with defaults, giving priority to parsed values.
    
    Args:
        parsed_params: Parameters extracted from prompt
        default_params: Default parameter values
        
    Returns:
        dict: Merged parameters
    """
    merged = default_params.copy()
    merged.update(parsed_params)
    
    logger.debug(f"Merged params: parsed={len(parsed_params)} keys, defaults={len(default_params)} keys")
    return merged

def validate_parsed_params(params: dict) -> dict:
    """
    Validate and clamp parsed parameters to reasonable ranges.
    
    Args:
        params: Parameters to validate
        
    Returns:
        dict: Validated parameters
    """
    validated = params.copy()
    
    # Clamp octaves to reasonable range
    if 'mountain_octaves' in validated:
        validated['mountain_octaves'] = max(1, min(15, validated['mountain_octaves']))
    
    if 'valley_octaves' in validated:
        validated['valley_octaves'] = max(1, min(12, validated['valley_octaves']))
    
    # Clamp scale to reasonable range
    if 'scale' in validated:
        validated['scale'] = max(1.0, min(20.0, validated['scale']))
    
    # Clamp persistence to [0, 1]
    if 'persistence' in validated:
        validated['persistence'] = max(0.1, min(1.0, validated['persistence']))
    
    # Clamp lacunarity to reasonable range
    if 'lacunarity' in validated:
        validated['lacunarity'] = max(1.1, min(4.0, validated['lacunarity']))
    
    # Clamp weights to [0, 1]
    for weight_key in ['mountain_weight', 'valley_weight']:
        if weight_key in validated:
            validated[weight_key] = max(0.0, min(1.0, validated[weight_key]))
    
    # Clamp river parameters
    if 'river_strength' in validated:
        validated['river_strength'] = max(0.0, min(1.0, validated['river_strength']))
    
    if 'river_freq' in validated:
        validated['river_freq'] = max(0.01, min(0.1, validated['river_freq']))
    
    return validated
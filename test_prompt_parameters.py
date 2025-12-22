"""
Quick test script to verify prompt-to-parameter refactoring works correctly.
Tests the new deterministic seed generation and parameter extraction.
"""

import sys
sys.path.append('.')

from pipeline.prompt_parameters import parse_prompt, get_seed_from_prompt
import logging

logging.basicConfig(level=logging.INFO)

def test_deterministic_seeds():
    """Test that same prompt always generates same seed."""
    print("=" * 80)
    print("TEST 1: Deterministic Seed Generation")
    print("=" * 80)
    
    test_prompt = "dramatic mountain peaks with snow"
    
    # Generate seed multiple times
    seeds = [get_seed_from_prompt(test_prompt) for _ in range(5)]
    
    # All should be identical
    assert all(s == seeds[0] for s in seeds), "Seeds should be deterministic!"
    
    print(f"✓ Prompt: '{test_prompt}'")
    print(f"✓ Generated seed: {seeds[0]}")
    print(f"✓ All 5 generations produced identical seed")
    print()
    
    # Different prompts should produce different seeds
    prompt2 = "gentle rolling hills"
    seed2 = get_seed_from_prompt(prompt2)
    
    assert seed2 != seeds[0], "Different prompts should produce different seeds!"
    print(f"✓ Different prompt produces different seed: {seed2}")
    print()


def test_parameter_extraction():
    """Test parameter extraction from various prompts."""
    print("=" * 80)
    print("TEST 2: Parameter Extraction")
    print("=" * 80)
    
    test_cases = [
        {
            "prompt": "dramatic steep mountain peaks",
            "expected": {
                "elevation_scale": 2.0,
                "biome_type": "mountain",
                "terrain_style": "mountainous"
            }
        },
        {
            "prompt": "gentle rolling desert dunes",
            "expected": {
                "biome_type": "desert",
                "terrain_style": "hills",
                "roughness": 0.3
            }
        },
        {
            "prompt": "complex forest terrain with rivers",
            "expected": {
                "biome_type": "forest",
                "water_level": 0.2  # Rivers set water_level to 0.2
            }
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test case {i}: '{test['prompt']}'")
        params = parse_prompt(test['prompt'])
        
        # Check expected values
        for key, expected_value in test['expected'].items():
            actual_value = getattr(params, key)
            if isinstance(expected_value, float):
                # Allow small floating point differences
                assert abs(actual_value - expected_value) < 0.01, \
                    f"Expected {key}={expected_value}, got {actual_value}"
            else:
                assert actual_value == expected_value, \
                    f"Expected {key}={expected_value}, got {actual_value}"
        
        print(f"  ✓ Biome: {params.biome_type}")
        print(f"  ✓ Style: {params.terrain_style}")
        print(f"  ✓ Elevation Scale: {params.elevation_scale}")
        print(f"  ✓ Roughness: {params.roughness}")
        print(f"  ✓ Octaves: {params.octaves}")
        print(f"  ✓ Seed: {params.seed}")
        print()


def test_parameter_consistency():
    """Test that parameters are consistent with prompt content."""
    print("=" * 80)
    print("TEST 3: Parameter Consistency")
    print("=" * 80)
    
    # High complexity prompt should have high octaves
    complex_prompt = "extremely detailed rugged jagged mountain terrain with steep cliffs and deep valleys"
    params1 = parse_prompt(complex_prompt)
    
    # Simple prompt should have lower octaves
    simple_prompt = "flat plain"
    params2 = parse_prompt(simple_prompt)
    
    print(f"Complex prompt octaves: {params1.octaves}")
    print(f"Simple prompt octaves: {params2.octaves}")
    
    assert params1.octaves > params2.octaves, \
        "Complex prompts should have more octaves than simple ones!"
    
    print("✓ Complex prompt generates more detail levels (octaves)")
    print()
    
    # Dramatic prompt should have higher elevation scale
    dramatic_prompt = "dramatic towering mountain peaks"
    params3 = parse_prompt(dramatic_prompt)
    
    gentle_prompt = "gentle hills"
    params4 = parse_prompt(gentle_prompt)
    
    print(f"Dramatic prompt elevation scale: {params3.elevation_scale}")
    print(f"Gentle prompt elevation scale: {params4.elevation_scale}")
    
    assert params3.elevation_scale > params4.elevation_scale, \
        "Dramatic prompts should have higher elevation scale!"
    
    print("✓ Dramatic prompt generates higher elevation scaling")
    print()


def test_override_seed():
    """Test that override seed parameter works."""
    print("=" * 80)
    print("TEST 4: Override Seed")
    print("=" * 80)
    
    prompt = "mountain terrain"
    
    # Get default deterministic seed
    params_default = parse_prompt(prompt)
    default_seed = params_default.seed
    
    # Override with custom seed
    custom_seed = 12345
    params_override = parse_prompt(prompt, override_seed=custom_seed)
    
    print(f"Default deterministic seed: {default_seed}")
    print(f"Override seed: {params_override.seed}")
    
    assert params_override.seed == custom_seed, \
        "Override seed should be used when provided!"
    
    assert params_override.seed != default_seed, \
        "Override seed should differ from deterministic seed!"
    
    print("✓ Override seed parameter works correctly")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PROMPT-TO-PARAMETER REFACTORING TESTS" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        test_deterministic_seeds()
        test_parameter_extraction()
        test_parameter_consistency()
        test_override_seed()
        
        print("=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("Summary:")
        print("  ✓ Deterministic seed generation working")
        print("  ✓ Parameter extraction from prompts working")
        print("  ✓ Parameters adapt to prompt complexity")
        print("  ✓ Override seed functionality working")
        print()
        print("The prompt-to-parameter refactoring is complete and functional!")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

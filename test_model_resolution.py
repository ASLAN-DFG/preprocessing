#!/usr/bin/env python3
"""Test script to demonstrate the new model name resolution functionality."""

from preprocessing.spacy import _resolve_model_name

def test_model_resolution():
    """Test various model name resolution scenarios."""
    
    print("=== Model Name Resolution Tests ===\n")
    
    # Test 1: Language only (should default to md)
    print("Test 1: Language only (defaults to md)")
    result = _resolve_model_name("en")
    print(f"  _resolve_model_name('en') -> {result}")
    assert result == "en_core_web_md", f"Expected 'en_core_web_md', got '{result}'"
    print("  ✓ Passed\n")
    
    # Test 2: Language with specific size
    print("Test 2: Language with specific size")
    result = _resolve_model_name("de", size="lg")
    print(f"  _resolve_model_name('de', size='lg') -> {result}")
    assert result == "de_core_news_lg", f"Expected 'de_core_news_lg', got '{result}'"
    print("  ✓ Passed\n")
    
    # Test 3: Small model
    print("Test 3: Small model")
    result = _resolve_model_name("fr", size="sm")
    print(f"  _resolve_model_name('fr', size='sm') -> {result}")
    assert result == "fr_core_news_sm", f"Expected 'fr_core_news_sm', got '{result}'"
    print("  ✓ Passed\n")
    
    # Test 4: Custom model name (ignores language and size)
    print("Test 4: Custom model name (ignores language and size)")
    result = _resolve_model_name("en", model_name="my_custom_model", size="lg")
    print(f"  _resolve_model_name('en', model_name='my_custom_model', size='lg') -> {result}")
    assert result == "my_custom_model", f"Expected 'my_custom_model', got '{result}'"
    print("  ✓ Passed\n")
    
    # Test 5: Slovenian support
    print("Test 5: Slovenian support")
    result = _resolve_model_name("sl")
    print(f"  _resolve_model_name('sl') -> {result}")
    assert result == "sl_core_news_md", f"Expected 'sl_core_news_md', got '{result}'"
    print("  ✓ Passed\n")
    
    # Test 6: Invalid size should raise error
    print("Test 6: Invalid size raises ValueError")
    try:
        _resolve_model_name("en", size="xl")
        print("  ✗ Failed - should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ValueError raised as expected: {e}")
        print("  ✓ Passed\n")
    
    # Test 7: Unsupported language should raise error
    print("Test 7: Unsupported language raises ValueError")
    try:
        _resolve_model_name("xx")
        print("  ✗ Failed - should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ValueError raised as expected: {e}")
        print("  ✓ Passed\n")
    
    print("=== All Tests Passed! ===\n")
    
    # Print supported languages
    print("Supported languages and their base model names:")
    from preprocessing.spacy import LANGUAGE_MODELS, VALID_SIZES, DEFAULT_SIZE
    for lang, base_model in LANGUAGE_MODELS.items():
        print(f"  {lang}: {base_model}_{{{','.join(sorted(VALID_SIZES))}}} (default: {base_model}_{DEFAULT_SIZE})")

if __name__ == "__main__":
    test_model_resolution()

#!/usr/bin/env python3
"""
Test script for Ollama model detection
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_system.llm_handler import get_available_ollama_models, LLMHandler

def test_ollama_detection():
    """Test Ollama model detection."""

    print("=" * 80)
    print("OLLAMA MODEL DETECTION TEST")
    print("=" * 80)

    # Test 1: Get available models
    print("\n1. Detecting available Ollama models...")
    models = get_available_ollama_models()

    if models:
        print(f"✓ Found {len(models)} models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
    else:
        print("✗ No models found (Ollama may not be running or no models installed)")

    # Test 2: Test LLMHandler auto-detection
    print("\n2. Testing LLMHandler auto-detection...")
    try:
        handler = LLMHandler(provider="ollama")
        print(f"✓ LLMHandler initialized successfully")
        print(f"  Provider: {handler.provider}")

        # Try to get model name
        if hasattr(handler.llm, 'model'):
            print(f"  Model: {handler.llm.model}")
        else:
            print("  Model: (could not detect model name)")

    except ConnectionError as e:
        print(f"✗ Ollama not running: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 3: Test with specific model
    if models:
        print(f"\n3. Testing with specific model: {models[0]}")
        try:
            handler = LLMHandler(provider="ollama", model=models[0])
            print(f"✓ LLMHandler initialized with specific model")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_ollama_detection()

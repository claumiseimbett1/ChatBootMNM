#!/usr/bin/env python3
"""Test script to verify transformers import"""

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ SUCCESS: transformers imports work correctly")
    print("✅ AutoTokenizer imported successfully")
    print("✅ AutoModelForCausalLM imported successfully")
except ImportError as e:
    print(f"❌ ERROR: {e}")
    print("❌ transformers import failed")
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {e}")
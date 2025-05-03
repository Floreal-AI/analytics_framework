#!/usr/bin/env python3
"""
Test script to verify imports are working correctly.
"""

# Test basic imports
try:
    import conversion_subnet
    print(f"✅ Successfully imported conversion_subnet (version: {conversion_subnet.__version__})")
    
    from conversion_subnet import ConversionSynapse, BinaryClassificationMiner, BaseValidator
    print("✅ Successfully imported main classes")
    
    # Test imports from modules
    from conversion_subnet.protocol import ConversionSynapse
    print("✅ Successfully imported ConversionSynapse from protocol")
    
    from conversion_subnet.miner.miner import BinaryClassificationMiner
    print("✅ Successfully imported BinaryClassificationMiner from miner.miner")
    
    from conversion_subnet.validator.reward import Validator
    print("✅ Successfully imported Validator from validator.reward")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    
print("\nAll import tests completed.") 
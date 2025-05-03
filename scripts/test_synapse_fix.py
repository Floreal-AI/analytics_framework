#!/usr/bin/env python3
# Test script to verify fixes to the ConversionSynapse class

import bittensor as bt
from conversion_subnet.protocol import ConversionSynapse, PredictionOutput
from conversion_subnet.validator.generate import generate_conversation

# Configure logging
bt.logging.set_trace(True)
bt.logging.set_debug(True)

def test_empty_prediction():
    """Test that an empty prediction is handled correctly"""
    print("\n=== Testing Empty Prediction ===")
    
    # Create a synapse with features
    features = generate_conversation()
    synapse = ConversionSynapse(features=features)
    
    # Set an empty prediction
    synapse.set_prediction({})
    
    # Check if prediction is validated
    print(f"Initial prediction: {{}}")
    print(f"Validated prediction: {synapse.prediction}")
    
    # Verify required fields are present
    assert 'conversion_happened' in synapse.prediction, "conversion_happened should be added to empty prediction"
    assert 'time_to_conversion_seconds' in synapse.prediction, "time_to_conversion_seconds should be added to empty prediction"
    
    # Verify types are correct
    assert isinstance(synapse.prediction['conversion_happened'], int), "conversion_happened should be an integer"
    assert isinstance(synapse.prediction['time_to_conversion_seconds'], float), "time_to_conversion_seconds should be a float"
    
    print("Empty prediction test passed!")

def test_deserialize_none_prediction():
    """Test that deserializing a None prediction returns a valid dict"""
    print("\n=== Testing Deserialize None Prediction ===")
    
    # Create a synapse with features
    features = generate_conversation()
    synapse = ConversionSynapse(features=features)
    
    # Don't set a prediction
    synapse.set_prediction(None)
    
    # Deserialize
    deserialized = synapse.deserialize()
    print(f"Deserialized None prediction: {deserialized}")
    
    # Verify required fields are present
    assert 'conversion_happened' in deserialized, "conversion_happened should be in deserialized output"
    assert 'time_to_conversion_seconds' in deserialized, "time_to_conversion_seconds should be in deserialized output"
    
    # Verify types are correct
    assert isinstance(deserialized['conversion_happened'], int), "conversion_happened should be an integer"
    assert isinstance(deserialized['time_to_conversion_seconds'], float), "time_to_conversion_seconds should be a float"
    
    print("Deserialize None prediction test passed!")

def test_invalid_prediction_types():
    """Test that invalid prediction types are handled correctly"""
    print("\n=== Testing Invalid Prediction Types ===")
    
    # Create a synapse with features
    features = generate_conversation()
    synapse = ConversionSynapse(features=features)
    
    # Set a prediction with invalid types
    synapse.set_prediction({
        'conversion_happened': '1',  # String instead of int
        'time_to_conversion_seconds': '45.2'  # String instead of float
    })
    
    # Check if prediction is validated
    print(f"Initial prediction: {{'conversion_happened': '1', 'time_to_conversion_seconds': '45.2'}}")
    print(f"Validated prediction: {synapse.prediction}")
    
    # Verify types are corrected
    assert isinstance(synapse.prediction['conversion_happened'], int), "conversion_happened should be converted to integer"
    assert isinstance(synapse.prediction['time_to_conversion_seconds'], float), "time_to_conversion_seconds should be converted to float"
    
    print("Invalid prediction types test passed!")

def test_missing_prediction_fields():
    """Test that missing prediction fields are handled correctly"""
    print("\n=== Testing Missing Prediction Fields ===")
    
    # Create a synapse with features
    features = generate_conversation()
    synapse = ConversionSynapse(features=features)
    
    # Set predictions with missing fields
    test_cases = [
        {'conversion_happened': 1},  # Missing time_to_conversion_seconds
        {'time_to_conversion_seconds': 45.2}  # Missing conversion_happened
    ]
    
    for i, case in enumerate(test_cases):
        synapse.set_prediction(case.copy())
        print(f"\nTest case {i+1}:")
        print(f"Initial prediction: {case}")
        print(f"Validated prediction: {synapse.prediction}")
        
        # Verify all required fields are present
        assert 'conversion_happened' in synapse.prediction, "conversion_happened should be added if missing"
        assert 'time_to_conversion_seconds' in synapse.prediction, "time_to_conversion_seconds should be added if missing"
    
    print("Missing prediction fields test passed!")

def main():
    """Run all tests"""
    print("Starting tests for ConversionSynapse fixes...")
    
    test_empty_prediction()
    test_deserialize_none_prediction()
    test_invalid_prediction_types()
    test_missing_prediction_fields()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 
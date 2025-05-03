#!/usr/bin/env python3
# Test script to verify fixes for integer validation and dict object errors

import bittensor as bt
import numpy as np
from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.validator.utils import validate_features, preprocess_prediction
from conversion_subnet.validator.generate import generate_conversation, preprocess_features

# Configure logging
bt.logging.set_trace(True)
bt.logging.set_debug(True)

def test_integer_validation():
    """Test the integer validation fixes"""
    print("\n=== Testing Integer Validation Fixes ===")
    
    # Create test features with float values for integer fields
    test_features = {
        'session_id': 'test-session',
        'conversation_duration_seconds': 120.5,
        'has_target_entity': 1.0,  # Float that should be an integer
        'entities_collected_count': 4.7,  # Float that should be an integer
        'message_ratio': 1.2,
        'hour_of_day': 15.3,  # Float that should be an integer
        'day_of_week': 2.9,  # Float that should be an integer
        'max_message_length_user': 50.46,  # The problematic field from the error log
        'min_message_length_user': 30.5,
        'avg_message_length_user': 35.7,
        'max_message_length_agent': 96.53,  # The problematic field from the error log
        'min_message_length_agent': 40.2,
        'avg_message_length_agent': 60.8,
        'total_messages': 10,
        'user_messages_count': 5,
        'agent_messages_count': 5,
        'question_count_agent': 2.3,
        'question_count_user': 1.7
    }
    
    # Test preprocess_features function
    print("\nTesting preprocess_features function:")
    processed = preprocess_features(test_features)
    for field in ['hour_of_day', 'day_of_week', 'max_message_length_user', 'max_message_length_agent']:
        print(f"{field}: {type(processed[field]).__name__} = {processed[field]}")
        assert isinstance(processed[field], int), f"{field} should be an integer"
    
    # Test validate_features function
    print("\nTesting validate_features function:")
    validated = validate_features(test_features)
    for field in ['hour_of_day', 'day_of_week', 'max_message_length_user', 'max_message_length_agent']:
        print(f"{field}: {type(validated[field]).__name__} = {validated[field]}")
        assert isinstance(validated[field], int), f"{field} should be an integer"
    
    # Test ConversionSynapse instantiation
    print("\nTesting ConversionSynapse instantiation:")
    synapse = ConversionSynapse(features=test_features)
    for field in ['hour_of_day', 'day_of_week', 'max_message_length_user', 'max_message_length_agent']:
        print(f"{field}: {type(synapse.features[field]).__name__} = {synapse.features[field]}")
        assert isinstance(synapse.features[field], int), f"{field} should be an integer"
    
    print("\nInteger validation tests passed!")

def test_dict_object_handling():
    """Test the dict object handling fixes"""
    print("\n=== Testing Dict Object Handling Fixes ===")
    
    # Import the reward module
    from conversion_subnet.validator.reward import Validator as RewardValidator
    
    # Create a test validator
    reward_validator = RewardValidator()
    
    # Create test targets
    targets = {
        'conversion_happened': 1,
        'time_to_conversion_seconds': 45.2
    }
    
    # Create test responses as dict objects without response_time attribute
    dict_response1 = {
        'prediction': {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 50.1
        },
        'confidence': 0.8,
        # Missing response_time and miner_uid
    }
    
    dict_response2 = {
        'prediction': {
            'conversion_happened': 0,
            'time_to_conversion_seconds': -1.0
        },
        # Missing confidence, response_time, and miner_uid
    }
    
    # Test reward function with dict responses
    print("\nTesting reward function with dict responses:")
    reward1 = reward_validator.reward(targets, dict_response1)
    print(f"Reward for dict_response1: {reward1}")
    
    reward2 = reward_validator.reward(targets, dict_response2)
    print(f"Reward for dict_response2: {reward2}")
    
    # Test with a proper ConversionSynapse for comparison
    synapse_response = ConversionSynapse(features={})
    synapse_response.prediction = {
        'conversion_happened': 1,
        'time_to_conversion_seconds': 60.0
    }
    synapse_response.confidence = 0.9
    synapse_response.response_time = 0.5
    synapse_response.miner_uid = 1
    
    reward3 = reward_validator.reward(targets, synapse_response)
    print(f"Reward for synapse_response: {reward3}")
    
    print("\nDict object handling tests passed!")

def test_prediction_preprocessing():
    """Test the prediction preprocessing fixes"""
    print("\n=== Testing Prediction Preprocessing Fixes ===")
    
    # Test various prediction formats
    test_cases = [
        # Integer/float mix
        {
            'conversion_happened': 1.0,  # Float that should be an integer
            'time_to_conversion_seconds': 45
        },
        # String values
        {
            'conversion_happened': '1',
            'time_to_conversion_seconds': '45.2'
        },
        # None values
        {
            'conversion_happened': None,
            'time_to_conversion_seconds': None
        },
        # Empty dict
        {}
    ]
    
    print("\nTesting preprocess_prediction function:")
    for i, test_case in enumerate(test_cases):
        processed = preprocess_prediction(test_case)
        print(f"\nTest case {i+1}:")
        print(f"  Input:  {test_case}")
        print(f"  Output: {processed}")
        
        if 'conversion_happened' in processed:
            assert isinstance(processed['conversion_happened'], int), "conversion_happened should be an integer"
        
        if 'time_to_conversion_seconds' in processed:
            assert isinstance(processed['time_to_conversion_seconds'], float), "time_to_conversion_seconds should be a float"
    
    print("\nPrediction preprocessing tests passed!")

def main():
    """Run all tests"""
    print("Starting tests for validator fixes...")
    
    test_integer_validation()
    test_dict_object_handling()
    test_prediction_preprocessing()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 
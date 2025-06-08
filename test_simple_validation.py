#!/usr/bin/env python3
"""
Simple validation test for the external validation system.

This test focuses on the core components without complex HTTP mocking.
"""

import asyncio
import uuid
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conversion_subnet.validator.validation_client import (
    ValidationAPIClient,
    ValidationResult,
    ValidationError
)
from conversion_subnet.validator.forward import generate_ground_truth, validate_prediction
from conversion_subnet.validator.reward import Validator
from conversion_subnet.protocol import ConversionSynapse


def test_validation_result_creation():
    """Test creating ValidationResult objects."""
    print("🧪 Testing ValidationResult creation...")
    
    result = ValidationResult(
        test_pk="test-123",
        labels=True,
        submission_deadline="2025-06-05T20:34:49.944Z",
        response_time=0.5
    )
    
    assert result.test_pk == "test-123"
    assert result.labels is True
    assert result.submission_deadline == "2025-06-05T20:34:49.944Z"
    assert result.response_time == 0.5
    
    print("✅ ValidationResult creation works")


def test_validation_client_creation():
    """Test creating ValidationAPIClient."""
    print("\n🔧 Testing ValidationAPIClient creation...")
    
    client = ValidationAPIClient(
        base_url="https://api.example.com/v1",
        api_key="test-key-123",
        timeout=30.0
    )
    
    assert client.base_url == "https://api.example.com/v1"
    assert client.api_key == "test-key-123"
    assert client.timeout == 30.0
    
    print("✅ ValidationAPIClient creation works")


def test_ground_truth_generation():
    """Test synthetic ground truth generation."""
    print("\n🎯 Testing ground truth generation...")
    
    # Test conversion scenario (Rule 1: has target entity + enough entities)
    features_conversion = {
        'has_target_entity': 1,
        'entities_collected_count': 5,
        'message_ratio': 1.5,
        'conversation_duration_seconds': 120.0
    }
    
    ground_truth = generate_ground_truth(features_conversion)
    print(f"   Conversion scenario: {ground_truth}")
    assert ground_truth['conversion_happened'] == 1
    assert ground_truth['time_to_conversion_seconds'] > 0
    
    # Test conversion scenario (Rule 2: good message ratio + long conversation)
    features_rule2 = {
        'has_target_entity': 0,
        'entities_collected_count': 1,
        'message_ratio': 1.5,  # > 1.2 threshold
        'conversation_duration_seconds': 100.0  # > 90 threshold
    }
    
    ground_truth_rule2 = generate_ground_truth(features_rule2)
    print(f"   Rule 2 conversion: {ground_truth_rule2}")
    assert ground_truth_rule2['conversion_happened'] == 1
    assert ground_truth_rule2['time_to_conversion_seconds'] > 0
    
    # Test no conversion scenario
    features_no_conversion = {
        'has_target_entity': 0,
        'entities_collected_count': 1,
        'message_ratio': 1.0,  # <= 1.2 threshold
        'conversation_duration_seconds': 60.0  # <= 90 threshold
    }
    
    ground_truth_no = generate_ground_truth(features_no_conversion)
    print(f"   No conversion scenario: {ground_truth_no}")
    assert ground_truth_no['conversion_happened'] == 0
    assert ground_truth_no['time_to_conversion_seconds'] == -1.0
    
    print("✅ Ground truth generation works correctly")


def test_prediction_validation():
    """Test prediction validation logic."""
    print("\n✅ Testing prediction validation...")
    
    # Valid predictions
    valid_cases = [
        {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
        {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0},
        {'conversion_happened': '1', 'time_to_conversion_seconds': '60.0'},  # String conversion
        {'conversion_happened': '0', 'time_to_conversion_seconds': '-1.0'},
    ]
    
    for i, prediction in enumerate(valid_cases):
        result = validate_prediction(prediction.copy())
        print(f"   Valid case {i+1}: {result}")
        assert result is True
    
    # Invalid predictions
    invalid_cases = [
        {},  # Empty
        {'conversion_happened': 1},  # Missing time
        {'time_to_conversion_seconds': 60.0},  # Missing conversion
        {'conversion_happened': 2, 'time_to_conversion_seconds': 60.0},  # Invalid conversion value
        {'conversion_happened': 1, 'time_to_conversion_seconds': 0.0},  # Invalid time for conversion=1
        {'conversion_happened': 0, 'time_to_conversion_seconds': 60.0},  # Invalid time for conversion=0
    ]
    
    for i, prediction in enumerate(invalid_cases):
        result = validate_prediction(prediction.copy())
        print(f"   Invalid case {i+1}: {result}")
        assert result is False
    
    print("✅ Prediction validation works correctly")


def test_reward_calculation():
    """Test reward calculation with different scenarios."""
    print("\n💰 Testing reward calculation...")
    
    validator = Validator()
    
    # Ground truth: conversion happened
    ground_truth_conversion = {
        'conversion_happened': 1,
        'time_to_conversion_seconds': 60.0
    }
    
    # Ground truth: no conversion
    ground_truth_no_conversion = {
        'conversion_happened': 0,
        'time_to_conversion_seconds': -1.0
    }
    
    # Test Case 1: Correct prediction (conversion)
    correct_response = ConversionSynapse(features={})
    correct_response.prediction = {
        'conversion_happened': 1,
        'time_to_conversion_seconds': 55.0
    }
    correct_response.confidence = 0.9
    correct_response.response_time = 5.0
    correct_response.miner_uid = 1
    
    correct_reward = validator.reward(ground_truth_conversion, correct_response)
    print(f"   Correct prediction reward: {correct_reward:.4f}")
    
    # Test Case 2: Incorrect prediction (predicted no conversion when there was)
    incorrect_response = ConversionSynapse(features={})
    incorrect_response.prediction = {
        'conversion_happened': 0,
        'time_to_conversion_seconds': -1.0
    }
    incorrect_response.confidence = 0.7
    incorrect_response.response_time = 8.0
    incorrect_response.miner_uid = 2
    
    incorrect_reward = validator.reward(ground_truth_conversion, incorrect_response)
    print(f"   Incorrect prediction reward: {incorrect_reward:.4f}")
    
    # Test Case 3: Correct prediction (no conversion)
    correct_no_conv_response = ConversionSynapse(features={})
    correct_no_conv_response.prediction = {
        'conversion_happened': 0,
        'time_to_conversion_seconds': -1.0
    }
    correct_no_conv_response.confidence = 0.8
    correct_no_conv_response.response_time = 6.0
    correct_no_conv_response.miner_uid = 3
    
    correct_no_conv_reward = validator.reward(ground_truth_no_conversion, correct_no_conv_response)
    print(f"   Correct no-conversion reward: {correct_no_conv_reward:.4f}")
    
    # Verify reward logic
    assert correct_reward > incorrect_reward, "Correct prediction should get higher reward"
    assert 0.0 <= correct_reward <= 1.0, "Rewards should be between 0 and 1"
    assert 0.0 <= incorrect_reward <= 1.0, "Rewards should be between 0 and 1"
    assert 0.0 <= correct_no_conv_reward <= 1.0, "Rewards should be between 0 and 1"
    
    print("✅ Reward calculation works correctly")
    return correct_reward, incorrect_reward, correct_no_conv_reward


def test_external_validation_format():
    """Test the format conversion from external API to ground truth."""
    print("\n🌐 Testing external validation format conversion...")
    
    # Test True labels (conversion happened)
    external_result_true = ValidationResult(
        test_pk="test-123",
        labels=True,
        submission_deadline="2025-06-05T20:34:49.944Z",
        response_time=0.3
    )
    
    ground_truth_true = {
        'conversion_happened': 1 if external_result_true.labels else 0,
        'time_to_conversion_seconds': 60.0 if external_result_true.labels else -1.0
    }
    
    print(f"   External True → Ground truth: {ground_truth_true}")
    assert ground_truth_true['conversion_happened'] == 1
    assert ground_truth_true['time_to_conversion_seconds'] == 60.0
    
    # Test False labels (no conversion)
    external_result_false = ValidationResult(
        test_pk="test-456",
        labels=False,
        submission_deadline="2025-06-05T20:34:49.944Z",
        response_time=0.2
    )
    
    ground_truth_false = {
        'conversion_happened': 1 if external_result_false.labels else 0,
        'time_to_conversion_seconds': 60.0 if external_result_false.labels else -1.0
    }
    
    print(f"   External False → Ground truth: {ground_truth_false}")
    assert ground_truth_false['conversion_happened'] == 0
    assert ground_truth_false['time_to_conversion_seconds'] == -1.0
    
    print("✅ External validation format conversion works")


def test_end_to_end_scenario():
    """Test a complete end-to-end scenario."""
    print("\n🚀 Testing end-to-end scenario...")
    
    # 1. Generate test_pk
    test_pk = str(uuid.uuid4())
    print(f"   Generated test_pk: {test_pk}")
    
    # 2. Simulate external validation result
    external_result = ValidationResult(
        test_pk=test_pk,
        labels=True,  # API says conversion happened
        submission_deadline="2025-06-05T20:34:49.944Z",
        response_time=0.4
    )
    print(f"   External API result: conversion = {external_result.labels}")
    
    # 3. Convert to ground truth
    ground_truth = {
        'conversion_happened': 1 if external_result.labels else 0,
        'time_to_conversion_seconds': 60.0 if external_result.labels else -1.0
    }
    print(f"   Ground truth: {ground_truth}")
    
    # 4. Create miner responses
    miners = [
        {'prediction': {'conversion_happened': 1, 'time_to_conversion_seconds': 55.0}, 'confidence': 0.9},  # Correct
        {'prediction': {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}, 'confidence': 0.7},  # Incorrect
        {'prediction': {'conversion_happened': 1, 'time_to_conversion_seconds': 65.0}, 'confidence': 0.8},  # Correct
    ]
    
    # 5. Calculate rewards
    validator = Validator()
    rewards = []
    
    for i, miner in enumerate(miners):
        response = ConversionSynapse(features={})
        response.prediction = miner['prediction']
        response.confidence = miner['confidence']
        response.response_time = 5.0 + i
        response.miner_uid = i
        
        reward = validator.reward(ground_truth, response)
        rewards.append(reward)
        
        predicted = miner['prediction']['conversion_happened']
        print(f"   Miner {i} predicted {predicted}: reward = {reward:.4f}")
    
    # 6. Verify results
    print(f"\n   📊 Results Summary:")
    print(f"      External API: conversion = {external_result.labels}")
    print(f"      Miner 0 (predicted 1): {rewards[0]:.4f}")
    print(f"      Miner 1 (predicted 0): {rewards[1]:.4f}")
    print(f"      Miner 2 (predicted 1): {rewards[2]:.4f}")
    
    # Miners who predicted correctly (1) should get higher rewards than incorrect (0)
    assert rewards[0] > rewards[1], "Correct prediction should beat incorrect"
    assert rewards[2] > rewards[1], "Correct prediction should beat incorrect"
    
    print("✅ End-to-end scenario completed successfully")
    
    return {
        'test_pk': test_pk,
        'external_result': external_result,
        'ground_truth': ground_truth,
        'rewards': rewards
    }


def main():
    """Run all tests."""
    print("🧪 Simple External Validation Tests")
    print("=" * 50)
    
    try:
        # Test 1: Basic object creation
        test_validation_result_creation()
        test_validation_client_creation()
        
        # Test 2: Core logic
        test_ground_truth_generation()
        test_prediction_validation()
        
        # Test 3: Reward calculation
        test_reward_calculation()
        
        # Test 4: Format conversion
        test_external_validation_format()
        
        # Test 5: End-to-end scenario
        result = test_end_to_end_scenario()
        
        print("\n🎉 All tests passed!")
        print(f"🔗 Successfully validated end-to-end flow with test_pk: {result['test_pk']}")
        print(f"📈 Reward distribution: {[f'{r:.4f}' for r in result['rewards']]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
Integration test script for external validation system.

This script tests the complete flow:
1. Configure validation client
2. Generate test_pk and features
3. Mock miner responses
4. Call external validation (mocked)
5. Calculate rewards
6. Verify results
"""

import asyncio
import uuid
import sys
import os
from unittest.mock import patch, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conversion_subnet.validator.validation_client import (
    ValidationAPIClient,
    ValidationResult,
    configure_default_validation_client,
    get_default_validation_client
)
from conversion_subnet.validator.forward import get_external_validation, generate_ground_truth
from conversion_subnet.validator.reward import Validator
from conversion_subnet.protocol import ConversionSynapse


async def test_validation_client():
    """Test the validation client directly."""
    print("🔧 Testing ValidationAPIClient...")
    
    client = ValidationAPIClient(
        base_url="https://uqhnd2kvgi.execute-api.eu-west-3.amazonaws.com/v1",
        api_key="45b916d2-2ec8-46dc-9035-0b719b33fedf-a655aad5c474b9564d8031991c3817d98619533228d666c12e29b374e691e286",
        timeout=30.0
    )
    
    print(f"✅ Client configured: {client.base_url}")
    return client


async def test_mock_external_validation():
    """Test external validation with mocked responses."""
    print("\n🧪 Testing mocked external validation...")
    
    # Configure default client
    configure_default_validation_client(
        "https://test-api.com",
        "test-key",
        30.0
    )
    
    test_pk = str(uuid.uuid4())
    print(f"📋 Generated test_pk: {test_pk}")
    
    # Mock the validation result
    mock_result = ValidationResult(
        test_pk=test_pk,
        labels=True,
        submission_deadline="2025-06-05T20:34:49.944Z",
        response_time=0.5
    )
    
    # Mock the client call at module level
    with patch('conversion_subnet.validator.validation_client.get_default_validation_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.get_validation_result.return_value = mock_result
        mock_get_client.return_value = mock_client
        
        # Test external validation
        result = await get_external_validation(test_pk)
        
        assert result is not None
        assert result.test_pk == test_pk
        assert result.labels is True
        
        print(f"✅ External validation successful: {result.labels}")
        return result


def test_ground_truth_generation():
    """Test synthetic ground truth generation."""
    print("\n🎯 Testing ground truth generation...")
    
    # Test conversion scenario (Rule 1)
    features_conversion = {
        'has_target_entity': 1,
        'entities_collected_count': 5,
        'message_ratio': 1.5,
        'conversation_duration_seconds': 120.0
    }
    
    ground_truth = generate_ground_truth(features_conversion)
    print(f"✅ Conversion scenario: {ground_truth}")
    assert ground_truth['conversion_happened'] == 1
    assert ground_truth['time_to_conversion_seconds'] > 0
    
    # Test no conversion scenario
    features_no_conversion = {
        'has_target_entity': 0,
        'entities_collected_count': 1,
        'message_ratio': 1.0,
        'conversation_duration_seconds': 60.0
    }
    
    ground_truth_no = generate_ground_truth(features_no_conversion)
    print(f"✅ No conversion scenario: {ground_truth_no}")
    assert ground_truth_no['conversion_happened'] == 0
    assert ground_truth_no['time_to_conversion_seconds'] == -1.0
    
    return ground_truth


def test_reward_calculation():
    """Test reward calculation with external ground truth."""
    print("\n💰 Testing reward calculation...")
    
    validator = Validator()
    
    # External ground truth (from API)
    external_ground_truth = {
        'conversion_happened': 1,
        'time_to_conversion_seconds': 60.0
    }
    
    # Create correct miner response
    correct_response = ConversionSynapse(features={})
    correct_response.prediction = {
        'conversion_happened': 1,
        'time_to_conversion_seconds': 55.0
    }
    correct_response.confidence = 0.9
    correct_response.response_time = 5.0
    correct_response.miner_uid = 1
    
    # Create incorrect miner response
    incorrect_response = ConversionSynapse(features={})
    incorrect_response.prediction = {
        'conversion_happened': 0,
        'time_to_conversion_seconds': -1.0
    }
    incorrect_response.confidence = 0.7
    incorrect_response.response_time = 8.0
    incorrect_response.miner_uid = 2
    
    # Calculate rewards
    correct_reward = validator.reward(external_ground_truth, correct_response)
    incorrect_reward = validator.reward(external_ground_truth, incorrect_response)
    
    print(f"✅ Correct prediction reward: {correct_reward:.4f}")
    print(f"✅ Incorrect prediction reward: {incorrect_reward:.4f}")
    
    assert correct_reward > incorrect_reward
    assert 0.0 <= correct_reward <= 1.0
    assert 0.0 <= incorrect_reward <= 1.0
    
    print(f"✅ Reward calculation working correctly")
    return correct_reward, incorrect_reward


async def test_end_to_end_flow():
    """Test the complete end-to-end flow."""
    print("\n🚀 Testing end-to-end validation flow...")
    
    # 1. Generate test_pk and features
    test_pk = str(uuid.uuid4())
    features = {
        'session_id': 'test-session',
        'test_pk': test_pk,
        'conversation_duration_seconds': 120.0,
        'has_target_entity': 1,
        'entities_collected_count': 5,
        'message_ratio': 1.5
    }
    print(f"📋 Generated test_pk: {test_pk}")
    
    # 2. Create miner responses
    miner_responses = []
    for i in range(3):
        response = ConversionSynapse(features=features)
        response.prediction = {
            'conversion_happened': i % 2,  # Alternate predictions
            'time_to_conversion_seconds': 60.0 if i % 2 else -1.0
        }
        response.confidence = 0.8
        response.response_time = 10.0 + i
        response.miner_uid = i
        miner_responses.append(response)
    
    print(f"🤖 Created {len(miner_responses)} miner responses")
    
    # 3. Mock external validation
    mock_validation_result = ValidationResult(
        test_pk=test_pk,
        labels=True,  # External API says conversion happened
        submission_deadline="2025-06-05T20:34:49.944Z",
        response_time=0.3
    )
    
    configure_default_validation_client("https://test-api.com", "test-key", 30.0)
    
    with patch('conversion_subnet.validator.validation_client.get_default_validation_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.get_validation_result.return_value = mock_validation_result
        mock_get_client.return_value = mock_client
        
        # 4. Get external validation
        validation_result = await get_external_validation(test_pk)
        print(f"🌐 External validation result: conversion={validation_result.labels}")
        
        # 5. Convert to ground truth format
        ground_truth = {
            'conversion_happened': 1 if validation_result.labels else 0,
            'time_to_conversion_seconds': 60.0 if validation_result.labels else -1.0
        }
        print(f"🎯 Ground truth: {ground_truth}")
        
        # 6. Calculate rewards for all miners
        validator = Validator()
        rewards = []
        
        for response in miner_responses:
            reward = validator.reward(ground_truth, response)
            rewards.append(reward)
            print(f"💰 Miner {response.miner_uid} (predicted {response.prediction['conversion_happened']}): reward = {reward:.4f}")
        
        # 7. Verify results
        print(f"\n📊 Summary:")
        print(f"   External API result: conversion = {validation_result.labels}")
        print(f"   Miner 0 predicted: {miner_responses[0].prediction['conversion_happened']} → reward: {rewards[0]:.4f}")
        print(f"   Miner 1 predicted: {miner_responses[1].prediction['conversion_happened']} → reward: {rewards[1]:.4f}")
        print(f"   Miner 2 predicted: {miner_responses[2].prediction['conversion_happened']} → reward: {rewards[2]:.4f}")
        
        # Since external API said conversion=True, miners predicting 1 should get higher rewards
        assert rewards[1] > rewards[0]  # Miner 1 predicted 1, Miner 0 predicted 0
        assert rewards[1] > rewards[2]  # Miner 1 predicted 1, Miner 2 predicted 0
        
        print("✅ End-to-end flow completed successfully!")
        
        return {
            'test_pk': test_pk,
            'validation_result': validation_result,
            'ground_truth': ground_truth,
            'rewards': rewards
        }


async def main():
    """Run all tests."""
    print("🧪 External Validation Integration Tests")
    print("=" * 50)
    
    try:
        # Test 1: Validation client
        await test_validation_client()
        
        # Test 2: Mock external validation
        await test_mock_external_validation()
        
        # Test 3: Ground truth generation
        test_ground_truth_generation()
        
        # Test 4: Reward calculation
        test_reward_calculation()
        
        # Test 5: End-to-end flow
        result = await test_end_to_end_flow()
        
        print("\n🎉 All tests passed!")
        print(f"🔗 Integration successfully validated with test_pk: {result['test_pk']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 
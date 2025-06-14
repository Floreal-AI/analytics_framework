#!/usr/bin/env python3
"""
Detailed Reward Calculation Test
===============================

This test focuses specifically on the reward calculation step,
examining how predictions are scored against ground truth.
"""

import numpy as np
from conversion_subnet.validator.reward import Validator
from conversion_subnet.protocol import ConversionSynapse

def test_reward_calculation_scenarios():
    """Test different reward scenarios in detail."""
    
    validator = Validator()
    
    print("=" * 80)
    print("DETAILED REWARD CALCULATION ANALYSIS")
    print("=" * 80)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Perfect Prediction",
            "ground_truth": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "prediction": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "confidence": 0.95,
            "response_time": 1.5
        },
        {
            "name": "Correct Conversion, Wrong Time",
            "ground_truth": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "prediction": {'conversion_happened': 1, 'time_to_conversion_seconds': 45.5},
            "confidence": 0.87,
            "response_time": 2.5
        },
        {
            "name": "Wrong Conversion Prediction",
            "ground_truth": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "prediction": {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0},
            "confidence": 0.75,
            "response_time": 1.8
        },
        {
            "name": "Correct No-Conversion",
            "ground_truth": {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0},
            "prediction": {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0},
            "confidence": 0.90,
            "response_time": 1.2
        },
        {
            "name": "Wrong No-Conversion Prediction",
            "ground_truth": {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0},
            "prediction": {'conversion_happened': 1, 'time_to_conversion_seconds': 45.0},
            "confidence": 0.85,
            "response_time": 2.0
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'-'*60}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'-'*60}")
        
        # Create synapse for this scenario
        synapse = ConversionSynapse(
            features={"test_pk": f"scenario_{i}"},
            prediction=scenario['prediction'],
            confidence=scenario['confidence'],
            response_time=scenario['response_time'],
            miner_uid=0
        )
        
        print(f"Ground Truth:")
        print(f"  conversion_happened: {scenario['ground_truth']['conversion_happened']}")
        print(f"  time_to_conversion_seconds: {scenario['ground_truth']['time_to_conversion_seconds']}")
        
        print(f"Miner Prediction:")
        print(f"  conversion_happened: {scenario['prediction']['conversion_happened']}")
        print(f"  time_to_conversion_seconds: {scenario['prediction']['time_to_conversion_seconds']}")
        print(f"  confidence: {scenario['confidence']}")
        print(f"  response_time: {scenario['response_time']}")
        
        # Calculate reward
        reward = validator.reward(scenario['ground_truth'], synapse)
        
        print(f"\nReward Calculation:")
        print(f"  Final Reward: {reward:.6f}")
        
        # Analyze the components
        conversion_correct = scenario['prediction']['conversion_happened'] == scenario['ground_truth']['conversion_happened']
        print(f"  Conversion Prediction Correct: {conversion_correct}")
        
        if scenario['ground_truth']['conversion_happened'] == 1:
            time_error = abs(scenario['prediction']['time_to_conversion_seconds'] - scenario['ground_truth']['time_to_conversion_seconds'])
            print(f"  Time Prediction Error: {time_error:.1f} seconds")
        else:
            print(f"  Time Prediction: N/A (no conversion)")
        
        # Performance metrics
        if conversion_correct:
            print(f"  ✅ CORRECT prediction")
        else:
            print(f"  ❌ INCORRECT prediction")

def test_reward_formula_breakdown():
    """Break down the reward formula step by step."""
    print(f"\n{'='*80}")
    print("REWARD FORMULA BREAKDOWN")
    print("=" * 80)
    
    # Let's examine the reward class to understand the formula
    from conversion_subnet.validator.reward import Validator
    import inspect
    
    validator = Validator()
    
    # Get the reward method source
    try:
        source = inspect.getsource(validator.reward)
        print("Reward Method Source Code:")
        print("-" * 40)
        print(source)
    except Exception as e:
        print(f"Could not get source: {e}")
    
    # Test with specific values to see step by step
    print("\nSTEP-BY-STEP CALCULATION EXAMPLE:")
    print("-" * 40)
    
    ground_truth = {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0}
    prediction = {'conversion_happened': 1, 'time_to_conversion_seconds': 45.5}
    confidence = 0.87
    response_time = 2.5
    
    synapse = ConversionSynapse(
        features={"test_pk": "breakdown_test"},  
        prediction=prediction,
        confidence=confidence,
        response_time=response_time,
        miner_uid=0
    )
    
    print(f"Ground Truth: {ground_truth}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence}")
    print(f"Response Time: {response_time}")
    
    reward = validator.reward(ground_truth, synapse)
    print(f"Final Reward: {reward:.6f}")
    
    # Let's manually calculate components based on the reward logic
    # (This assumes we know the formula from the source code)
    conversion_accuracy = 1.0 if prediction['conversion_happened'] == ground_truth['conversion_happened'] else 0.0
    print(f"Conversion Accuracy Component: {conversion_accuracy}")
    
    if ground_truth['conversion_happened'] == 1:
        time_error = abs(prediction['time_to_conversion_seconds'] - ground_truth['time_to_conversion_seconds'])
        print(f"Time Error: {time_error} seconds")
        
        # Assuming some normalization for time accuracy
        max_time_error = 120.0  # example max
        time_accuracy = max(0.0, 1.0 - (time_error / max_time_error))
        print(f"Time Accuracy Component: {time_accuracy:.6f}")
    
    confidence_component = confidence
    print(f"Confidence Component: {confidence_component}")

if __name__ == "__main__":
    test_reward_calculation_scenarios()
    test_reward_formula_breakdown() 
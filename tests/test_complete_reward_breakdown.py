#!/usr/bin/env python3
"""
Complete Reward Calculation Breakdown
===================================

This test shows the complete reward calculation with all internal formulas
and step-by-step breakdowns for each component.
"""

from conversion_subnet.validator.reward import Validator
from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.constants import CLASS_W, BASELINE_MAE, PRED_W, TOTAL_REWARD_W, TIMEOUT_SEC

def detailed_reward_breakdown():
    """Show complete reward calculation breakdown."""
    
    print("=" * 100)
    print("COMPLETE REWARD CALCULATION BREAKDOWN")
    print("=" * 100)
    
    # Show constants first
    print("\n📊 REWARD CONSTANTS:")
    print("-" * 50)
    print(f"CLASS_W (Class Weights): {CLASS_W}")
    print(f"BASELINE_MAE (Baseline Mean Absolute Error): {BASELINE_MAE} seconds")
    print(f"PRED_W (Prediction Weights): {PRED_W}")
    print(f"TOTAL_REWARD_W (Total Reward Weights): {TOTAL_REWARD_W}")
    print(f"TIMEOUT_SEC (Timeout): {TIMEOUT_SEC} seconds")
    
    # Test scenario
    ground_truth = {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0}
    prediction = {'conversion_happened': 1, 'time_to_conversion_seconds': 45.5}
    confidence = 0.87
    response_time = 2.5
    
    synapse = ConversionSynapse(
        features={"test_pk": "detailed_test"},
        prediction=prediction,
        confidence=confidence,
        response_time=response_time,
        miner_uid=0
    )
    
    validator = Validator()
    
    print(f"\n🎯 TEST SCENARIO:")
    print("-" * 50)
    print(f"Ground Truth: {ground_truth}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence}")
    print(f"Response Time: {response_time}s")
    
    print(f"\n🔍 STEP-BY-STEP CALCULATION:")
    print("-" * 50)
    
    # 1. Classification Reward
    print("\n1️⃣ CLASSIFICATION REWARD:")
    classification_correct = prediction['conversion_happened'] == ground_truth['conversion_happened']
    print(f"   Prediction correct: {classification_correct}")
    
    if classification_correct and ground_truth['conversion_happened'] == 1:
        class_reward = 1.0 * CLASS_W['positive']
        print(f"   Base reward: 1.0")
        print(f"   Class weight (positive): {CLASS_W['positive']}")
        print(f"   Classification reward: 1.0 × {CLASS_W['positive']} = {class_reward}")
    elif classification_correct:
        class_reward = 1.0 * CLASS_W['negative']
        print(f"   Base reward: 1.0")
        print(f"   Class weight (negative): {CLASS_W['negative']}")
        print(f"   Classification reward: 1.0 × {CLASS_W['negative']} = {class_reward}")
    else:
        class_reward = 0.0
        print(f"   Classification reward: 0.0 (incorrect)")
    
    # 2. Regression Reward
    print("\n2️⃣ REGRESSION REWARD:")
    if prediction['conversion_happened'] == 1 and ground_truth['conversion_happened'] == 1:
        mae = abs(prediction['time_to_conversion_seconds'] - ground_truth['time_to_conversion_seconds'])
        regression_score = max(1 - mae / BASELINE_MAE, 0)
        print(f"   Both predicted and actual conversion: True")
        print(f"   MAE: |{prediction['time_to_conversion_seconds']} - {ground_truth['time_to_conversion_seconds']}| = {mae}")
        print(f"   Baseline MAE: {BASELINE_MAE}")
        print(f"   Regression score: max(1 - {mae}/{BASELINE_MAE}, 0) = {regression_score}")
    else:
        regression_score = 0.0
        print(f"   No conversion or mismatch: regression_score = 0.0")
    
    # 3. Diversity Reward
    print("\n3️⃣ DIVERSITY REWARD:")
    if confidence is None:
        diversity_reward = 0.0
        print(f"   Confidence is None: diversity_reward = 0.0")
    elif confidence == 0.5:
        diversity_reward = 0.0
        print(f"   Confidence exactly 0.5: diversity_reward = 0.0")
    elif confidence == 0.0 or confidence == 1.0:
        diversity_reward = 0.5
        print(f"   Confidence extreme value: diversity_reward = 0.5")
    else:
        diversity_reward = 0.5 - abs(confidence - 0.5)
        print(f"   Diversity reward: 0.5 - |{confidence} - 0.5| = {diversity_reward}")
    
    # 4. Prediction Reward (Combined)
    print("\n4️⃣ PREDICTION REWARD (COMBINED):")
    pred_reward = (PRED_W["classification"] * class_reward + 
                   PRED_W["regression"] * regression_score + 
                   PRED_W["diversity"] * diversity_reward)
    
    print(f"   Formula: PRED_W[classification] × class_reward + PRED_W[regression] × regression_score + PRED_W[diversity] × diversity_reward")
    print(f"   Calculation: {PRED_W['classification']} × {class_reward} + {PRED_W['regression']} × {regression_score} + {PRED_W['diversity']} × {diversity_reward}")
    print(f"   Prediction reward: {pred_reward}")
    
    # 5. Time Reward
    print("\n5️⃣ TIME REWARD:")
    time_reward = max(1 - response_time / TIMEOUT_SEC, 0)
    print(f"   Formula: max(1 - response_time/timeout, 0)")
    print(f"   Calculation: max(1 - {response_time}/{TIMEOUT_SEC}, 0) = {time_reward}")
    
    # 6. Total Reward
    print("\n6️⃣ TOTAL REWARD:")
    total_reward = (TOTAL_REWARD_W["prediction"] * pred_reward + 
                    TOTAL_REWARD_W["latency"] * time_reward)
    
    print(f"   Formula: TOTAL_REWARD_W[prediction] × pred_reward + TOTAL_REWARD_W[latency] × time_reward")
    print(f"   Calculation: {TOTAL_REWARD_W['prediction']} × {pred_reward} + {TOTAL_REWARD_W['latency']} × {time_reward}")
    print(f"   Total reward: {total_reward}")
    
    # Verify with actual calculation
    print(f"\n✅ VERIFICATION:")
    actual_reward = validator.reward(ground_truth, synapse)
    print(f"   Actual calculated reward: {actual_reward}")
    print(f"   Manual calculation: {total_reward}")
    print(f"   Match: {abs(actual_reward - total_reward) < 1e-6}")

def test_edge_cases():
    """Test edge cases in reward calculation."""
    
    print(f"\n{'='*100}")
    print("EDGE CASES TESTING")
    print("=" * 100)
    
    validator = Validator()
    
    test_cases = [
        {
            "name": "Perfect Prediction (Max Reward)",
            "ground_truth": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "prediction": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "confidence": 1.0,  # Extreme confidence
            "response_time": 0.1  # Very fast
        },
        {
            "name": "Worst Prediction (Min Reward)",
            "ground_truth": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "prediction": {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0},
            "confidence": 0.5,  # No diversity reward
            "response_time": 60.0  # Timeout
        },
        {
            "name": "Conservative Prediction (0.5 confidence)",
            "ground_truth": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "prediction": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "confidence": 0.5,  # Exact middle - no diversity reward
            "response_time": 30.0
        },
        {
            "name": "Time Prediction Way Off",
            "ground_truth": {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0},
            "prediction": {'conversion_happened': 1, 'time_to_conversion_seconds': 120.0},  # 60s error
            "confidence": 0.8,
            "response_time": 5.0
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'-'*80}")
        print(f"EDGE CASE {i}: {case['name']}")
        print("-" * 80)
        
        synapse = ConversionSynapse(
            features={"test_pk": f"edge_case_{i}"},
            prediction=case['prediction'],
            confidence=case['confidence'],
            response_time=case['response_time'],
            miner_uid=0
        )
        
        reward = validator.reward(case['ground_truth'], synapse)
        
        print(f"Ground Truth: {case['ground_truth']}")
        print(f"Prediction: {case['prediction']}")
        print(f"Confidence: {case['confidence']}")
        print(f"Response Time: {case['response_time']}s")
        print(f"Final Reward: {reward:.6f}")
        
        # Quick analysis
        conversion_correct = case['prediction']['conversion_happened'] == case['ground_truth']['conversion_happened']
        print(f"Conversion Correct: {'✅' if conversion_correct else '❌'}")
        
        if case['ground_truth']['conversion_happened'] == 1 and case['prediction']['conversion_happened'] == 1:
            time_error = abs(case['prediction']['time_to_conversion_seconds'] - case['ground_truth']['time_to_conversion_seconds'])
            print(f"Time Error: {time_error:.1f}s")

if __name__ == "__main__":
    detailed_reward_breakdown()
    test_edge_cases()
    
    print(f"\n{'='*100}")
    print("🎉 COMPLETE ANALYSIS FINISHED")
    print("=" * 100) 
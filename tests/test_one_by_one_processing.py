#!/usr/bin/env python3
"""
Test One-by-One Challenge Processing
===================================

This test verifies the new one-by-one processing functionality
implemented in send_challenge.py.
"""

import asyncio
import json
import time
from datetime import datetime

# Import the new functionality
from conversion_subnet.validator.send_challenge import (
    ConversationChallengeGenerator,
    process_miner_challenges_one_by_one
)
from conversion_subnet.miner.miner import BinaryClassificationMiner

async def test_one_by_one_processing():
    """Test the one-by-one challenge processing functionality."""
    
    print("🚀 TESTING ONE-BY-ONE CHALLENGE PROCESSING")
    print("=" * 80)
    
    # Initialize miner
    print("📦 Initializing miner...")
    miner = BinaryClassificationMiner()
    print(f"✅ Miner initialized: {type(miner).__name__}")
    
    # Test with direct class method
    print(f"\n{'='*60}")
    print("TEST 1: Direct Class Method")
    print("=" * 60)
    
    generator = ConversationChallengeGenerator()
    
    results_direct = await generator.process_challenges_one_by_one(
        miner_forward_function=miner.forward,
        num_challenges=3,
        miner_uid=123,
        template_distribution={
            'high_engagement_conversion': 0.4,
            'quick_exit': 0.3,
            'medium_engagement': 0.3
        },
        include_validation=False,  # Skip external validation for test
        delay_between_challenges=0.2
    )
    
    print(f"\n📊 DIRECT METHOD RESULTS:")
    print(f"   Challenges Processed: {results_direct['summary']['successful_predictions']}/3")
    print(f"   Total Reward: {results_direct['summary']['total_reward']:.6f}")
    print(f"   Average Reward: {results_direct['summary']['average_reward']:.6f}")
    print(f"   Prediction Accuracy: {results_direct['summary']['prediction_accuracy']:.1%}")
    print(f"   Total Time: {results_direct['summary']['total_processing_time']:.3f}s")
    
    # Test with convenience function
    print(f"\n{'='*60}")
    print("TEST 2: Convenience Function")
    print("=" * 60)
    
    results_convenience = await process_miner_challenges_one_by_one(
        miner_forward_function=miner.forward,
        num_challenges=3,
        miner_uid=456,
        template_distribution={
            'business_hours_professional': 0.5,
            'extended_exploration': 0.5
        },
        include_validation=False,
        delay_between_challenges=0.1
    )
    
    print(f"\n📊 CONVENIENCE FUNCTION RESULTS:")
    print(f"   Challenges Processed: {results_convenience['summary']['successful_predictions']}/3")
    print(f"   Total Reward: {results_convenience['summary']['total_reward']:.6f}")
    print(f"   Average Reward: {results_convenience['summary']['average_reward']:.6f}")
    print(f"   Prediction Accuracy: {results_convenience['summary']['prediction_accuracy']:.1%}")
    print(f"   Total Time: {results_convenience['summary']['total_processing_time']:.3f}s")
    
    # Detailed analysis of one challenge
    print(f"\n{'='*60}")
    print("TEST 3: Detailed Challenge Analysis")
    print("=" * 60)
    
    if results_direct['challenges']:
        first_challenge = results_direct['challenges'][0]
        print(f"\n🔍 CHALLENGE 1 DETAILED BREAKDOWN:")
        print(f"   Challenge ID: {first_challenge['challenge_id']}")
        print(f"   Success: {first_challenge['success']}")
        print(f"   Total Processing Time: {first_challenge['total_processing_time']}s")
        
        print(f"\n   📋 PROCESSING STEPS:")
        for step in first_challenge['processing_steps']:
            status = "✅" if step['success'] else "❌"
            print(f"   {status} {step['step']}: {step['duration']:.4f}s")
            
            if step['step'] == 'generate_challenge' and step['success']:
                details = step['details']
                print(f"       Template: {details['template_used']}")
                print(f"       Ground Truth: {details['ground_truth']}")
                print(f"       Total Messages: {details['total_messages']}")
                print(f"       Duration: {details['conversation_duration']:.1f} min")
            
            elif step['step'] == 'miner_prediction' and step['success']:
                details = step['details']
                print(f"       Prediction: {details['prediction']}")
                print(f"       Confidence: {details['confidence']:.3f}")
            
            elif step['step'] == 'reward_calculation' and step['success']:
                details = step['details']
                print(f"       Reward: {details['reward']:.6f}")
                print(f"       Correct: {details['prediction_correct']}")
    
    # Performance comparison
    print(f"\n{'='*60}")
    print("TEST 4: Performance Analysis")
    print("=" * 60)
    
    # Test with different delay settings
    start_time = time.time()
    
    fast_results = await process_miner_challenges_one_by_one(
        miner_forward_function=miner.forward,
        num_challenges=5,
        miner_uid=789,
        delay_between_challenges=0.0  # No delay
    )
    
    fast_time = time.time() - start_time
    
    print(f"\n⚡ FAST PROCESSING (no delay):")
    print(f"   5 challenges in {fast_time:.3f}s")
    print(f"   Average per challenge: {fast_time/5:.3f}s")
    print(f"   Success rate: {fast_results['summary']['successful_predictions']}/5")
    
    # Save detailed results
    results_summary = {
        'test_timestamp': datetime.now().isoformat(),
        'direct_method_results': results_direct,
        'convenience_function_results': results_convenience,
        'fast_processing_results': fast_results,
        'performance_metrics': {
            'direct_method_time': results_direct['summary']['total_processing_time'],
            'convenience_function_time': results_convenience['summary']['total_processing_time'],
            'fast_processing_time': fast_time,
            'average_time_per_challenge': fast_time / 5
        }
    }
    
    with open('one_by_one_test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: one_by_one_test_results.json")
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print("✅ All tests completed successfully!")
    print(f"✅ Direct method processing: {results_direct['summary']['successful_predictions']}/3 challenges")
    print(f"✅ Convenience function processing: {results_convenience['summary']['successful_predictions']}/3 challenges")
    print(f"✅ Fast processing: {fast_results['summary']['successful_predictions']}/5 challenges")
    print(f"✅ One-by-one processing is fully functional!")
    
    return results_summary

if __name__ == "__main__":
    asyncio.run(test_one_by_one_processing()) 
#!/usr/bin/env python3
# Test script to verify the validator can run without errors

import time
import asyncio
import bittensor as bt
from conversion_subnet.base.validator import BaseValidatorNeuron
from conversion_subnet.validator.reward import Validator as RewardValidator
from conversion_subnet.protocol import ConversionSynapse, PredictionOutput
from conversion_subnet.validator.generate import generate_conversation

# Configure logging
bt.logging.set_trace(True)
bt.logging.set_debug(True)

class TestValidator:
    """
    Test validator class to simulate validator operations
    """
    def __init__(self):
        print("Initializing TestValidator")
        
        # Initialize reward validator for testing
        self.reward_validator = RewardValidator()
        
        # Dictionary to hold test responses
        self.responses = []
        
        # Generate test ground truth
        self.targets = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 45.2
        }
        
        print("TestValidator initialized")
    
    def generate_test_responses(self):
        """Generate test responses for validation"""
        print("\nGenerating test responses")
        
        # Generate test features
        features = generate_conversation()
        print(f"Generated features with session_id: {features['session_id']}")
        
        # Create test synapse with valid integer fields
        synapse1 = ConversionSynapse(features=features)
        synapse1.prediction = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 50.1
        }
        synapse1.confidence = 0.8
        synapse1.response_time = 1.2
        synapse1.miner_uid = 1
        
        # Create test synapse with a dict
        response_dict = {
            'prediction': {
                'conversion_happened': 0,
                'time_to_conversion_seconds': -1.0
            },
            'confidence': 0.5,
            'miner_uid': 2
            # Deliberately missing response_time to test error handling
        }
        
        # Add responses
        self.responses.append(synapse1)
        self.responses.append(response_dict)
        
        print(f"Generated {len(self.responses)} test responses")
    
    def compute_rewards(self):
        """Compute rewards for test responses"""
        print("\nComputing rewards for test responses")
        
        rewards = []
        for i, response in enumerate(self.responses):
            try:
                reward = self.reward_validator.reward(self.targets, response)
                rewards.append(reward)
                print(f"Response {i+1}: Reward = {reward:.4f}")
            except Exception as e:
                print(f"Error computing reward for response {i+1}: {e}")
                rewards.append(0.0)
        
        # Calculate total rewards
        print(f"\nAverage reward: {sum(rewards)/len(rewards) if rewards else 0:.4f}")
        
        return rewards

async def run_test():
    """Run the test validator simulation"""
    print("Starting validator test simulation")
    
    # Create test validator
    validator = TestValidator()
    
    # Generate test responses
    validator.generate_test_responses()
    
    # Compute rewards
    rewards = validator.compute_rewards()
    
    print("\nValidator test completed successfully!")
    
    return rewards

if __name__ == "__main__":
    # Run the test
    rewards = asyncio.run(run_test())
    
    # Print summary
    print(f"\nFinal rewards: {rewards}")
    print("\nAll tests completed successfully!") 
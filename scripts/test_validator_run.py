#!/usr/bin/env python3
# Test script to run the validator with our fixes

import time
import asyncio
import bittensor as bt
from neurons.validator import Validator

# Configure logging
bt.logging.set_trace(True)
bt.logging.set_debug(True)

class ValidatorRunner:
    """Runs the validator for a limited number of steps for testing"""
    
    def __init__(self, max_steps=5):
        self.max_steps = max_steps
        self.current_step = 0
        
    async def run_validator(self):
        """Run the validator for a limited number of steps"""
        print(f"Starting validator for {self.max_steps} steps...")
        
        # Create validator instance
        config = get_test_config()
        validator = Validator(config)
        
        # Override the run method to just run for max_steps
        original_run = validator.run
        
        async def limited_run():
            self.current_step = 0
            while self.current_step < self.max_steps:
                print(f"\n--- Running step {self.current_step + 1}/{self.max_steps} ---")
                await validator.forward()
                self.current_step += 1
                await asyncio.sleep(1)  # Small delay between steps
            
            print(f"\nCompleted {self.max_steps} steps successfully!")
        
        # Replace run method
        validator.run = limited_run
        
        # Start validator
        with validator:
            await validator.run()
        
        return True

def get_test_config():
    """Get a test configuration for the validator"""
    config = bt.config()
    
    # Subnet configuration
    config.netuid = 2  # Test subnet
    config.subtensor = bt.config()
    config.subtensor.network = "local"  # Use local network
    config.subtensor.chain_endpoint = "ws://127.0.0.1:9944"  # Local endpoint
    
    # Wallet configuration
    config.wallet = bt.config()
    config.wallet.name = "validator"
    config.wallet.hotkey = "default"
    
    # Neuron configuration
    config.neuron = bt.config()
    config.neuron.sample_size = 1  # Query just 1 miner for testing
    config.neuron.moving_average_alpha = 0.05
    config.neuron.no_set_weights = True  # Don't actually set weights on chain
    
    # Logging configuration
    config.logging = bt.config()
    config.logging.debug = True
    config.logging.trace = True
    
    # Return the configuration
    return config

async def main():
    """Run the validator test"""
    runner = ValidatorRunner(max_steps=3)
    await runner.run_validator()

if __name__ == "__main__":
    # Run the test
    asyncio.run(main()) 
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 [Your Name/Organization] (modified for AI Agent Task Performance Subnet)

# [Existing license text remains unchanged]

import time
import os
import bittensor as bt
from conversion_subnet.base.validator import BaseValidatorNeuron
from conversion_subnet.validator.forward import forward
from conversion_subnet.validator.validation_client import configure_default_validation_client

class Validator(BaseValidatorNeuron):
    """
    Validator neuron for the AI Agent Task Performance Subnet. Queries miners with conversation features,
    scores their predictions (conversion_happened, time_to_conversion_seconds), and updates network weights.

    Inherits from BaseValidatorNeuron, which handles Bittensor network operations (wallet, subtensor, dendrite, syncing).
    This class delegates the forward method to the forward.py implementation, which generates or obtains features,
    queries miners, and scores responses using the Incentive Mechanism.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("Initializing validator")
        self.load_state()
        
        # Initialize conversation history storage
        self.conversation_history = {}
        
        # Initialize scores tracker
        self.scores_history = {}
        
        # Configure validation API client
        self._configure_validation_client()

    def _configure_validation_client(self):
        """Configure the validation API client with environment variables."""
        try:
            # Get API configuration from environment
            api_base_url = os.getenv(
                'VALIDATION_API_BASE_URL', 
                'https://uqhnd2kvgi.execute-api.eu-west-3.amazonaws.com/v1'
            )
            api_key = os.getenv(
                'VALIDATION_API_KEY', 
                '45b916d2-2ec8-46dc-9035-0b719b33fedf-a655aad5c474b9564d8031991c3817d98619533228d666c12e29b374e691e286'
            )
            timeout = float(os.getenv('VALIDATION_API_TIMEOUT', '30.0'))
            
            # Configure the default client
            configure_default_validation_client(api_base_url, api_key, timeout)
            bt.logging.info("Validation API client configured successfully")
            
        except Exception as e:
            bt.logging.error(f"Failed to configure validation API client: {e}")
            bt.logging.warning("Validator will fall back to synthetic ground truth")

    async def forward(self):
        """
        The forward function is called by the validator every time step. It delegates to the forward.py implementation,
        which:
        - Generates synthetic conversation features or obtains real-time features
        - Queries miners with ConversionSynapse
        - Scores responses using the Incentive Mechanism
        - Updates miner scores
        """
        try:
            await forward(self)
        except Exception as e:
            bt.logging.error(f"Error in forward function: {e}")

if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... Block: {validator.metagraph.block}")
            time.sleep(60)  # Wait for 1 minute between iterations

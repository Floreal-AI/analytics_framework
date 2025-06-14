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
from conversion_subnet.data_api import VoiceFormAPIClient, VoiceFormAPIConfig

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("DEBUG: .env file loaded successfully")
except ImportError:
    print("DEBUG: python-dotenv not available, relying on system environment variables")
except Exception as e:
    print(f"DEBUG: Error loading .env file: {e}")

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
        
        # Configure data API client
        self._configure_data_client()
        
        # Initialize test data offset tracker
        self.test_data_offset = 0

    def _configure_validation_client(self):
        """Configure the validation API client with environment variables."""
        try:
            # Get API configuration from environment - no defaults, must be in .env file
            api_base_url = os.getenv('VOICEFORM_API_BASE_URL')
            api_key = os.getenv('VOICEFORM_API_KEY')
            timeout = float(os.getenv('VALIDATION_API_TIMEOUT', '30.0'))
            

            # Validate required configuration
            if not api_base_url:
                raise ValueError("VOICEFORM_API_BASE_URL is required")
            if not api_key:
                raise ValueError("VOICEFORM_API_KEY is required")
            if timeout <= 0:
                raise ValueError("VALIDATION_API_TIMEOUT must be positive")
            
            # Configure the default client
            configure_default_validation_client(api_base_url, api_key, timeout)
            bt.logging.info("Validation API client configured successfully")
            
        except Exception as e:
            bt.logging.error(f"Failed to configure validation API client: {e}")
            raise RuntimeError(f"Validator initialization failed: External validation API is required but could not be configured: {e}") from e

    def _configure_data_client(self):
        """Configure the data API client with environment variables."""
        try:
            # Get API configuration from environment - same as validation client
            api_base_url = os.getenv('VOICEFORM_API_BASE_URL')
            api_key = os.getenv('VOICEFORM_API_KEY')
            timeout = float(os.getenv('DATA_API_TIMEOUT', '30.0'))
            
            # Validate required configuration
            if not api_base_url:
                raise ValueError("VOICEFORM_API_BASE_URL is required")
            if not api_key:
                raise ValueError("VOICEFORM_API_KEY is required")
            if timeout <= 0:
                raise ValueError("DATA_API_TIMEOUT must be positive")
            
            # Create configuration and client
            config = VoiceFormAPIConfig(
                api_key=api_key,
                base_url=api_base_url,
                timeout_seconds=int(timeout)
            )
            self.data_client = VoiceFormAPIClient(config)
            bt.logging.info("Data API client configured successfully")
            
        except Exception as e:
            bt.logging.error(f"Failed to configure data API client: {e}")
            raise RuntimeError(f"Validator initialization failed: Data API is required but could not be configured: {e}") from e

    async def forward(self):
        """
        The forward function is called by the validator every time step. It delegates to the forward.py implementation,
        which:
        - Fetches real test data from API instead of generating synthetic features
        - Queries miners with ConversionSynapse containing real test features
        - Scores responses using the Incentive Mechanism with external validation
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

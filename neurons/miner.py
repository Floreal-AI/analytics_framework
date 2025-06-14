import time
import typing
import bittensor as bt

from conversion_subnet.protocol import ConversionSynapse, PredictionOutput
from conversion_subnet.base.miner import BaseMinerNeuron
from conversion_subnet.miner.miner import BinaryClassificationMiner
from conversion_subnet.utils.log import logger
from conversion_subnet.constants import TIMEOUT_SEC

class Miner(BaseMinerNeuron):
    """
    Miner neuron for the AI Agent Task Performance Subnet. Processes conversation features to predict task outcomes
    (conversion_happened, time_to_conversion_seconds). This implementation uses a binary classification model
    to predict conversion outcomes.

    Inherits from BaseMinerNeuron, which handles Bittensor network operations (wallet, subtensor, axon, syncing).
    """

    def __init__(self, config=None):
        # Initialize with base configuration
        super(Miner, self).__init__(config=config)

        # Explicitly create necessary config objects if missing
        if not hasattr(self.config, 'neuron'):
            self.config.neuron = bt.config()
            self.config.neuron.device = 'cpu'
            logger.warning("Created missing neuron config with defaults")

        # Explicitly create miner config if missing
        if not hasattr(self.config, 'miner'):
            self.config.miner = bt.config()
            logger.warning("Created missing miner config with defaults")
        
        # Set device for the miner model, defaulting to CPU
        device = 'cpu'
        # Only use neuron.device if it exists and is not None
        if hasattr(self.config.neuron, 'device') and self.config.neuron.device is not None:
            device = self.config.neuron.device
        
        # Ensure miner device is set
        self.config.miner.device = device
        
        # Initialize the binary classification miner with proper config
        logger.info(f"Initializing classifier with device: {self.config.miner.device}")
        try:
            self.classifier = BinaryClassificationMiner(self.config)
            logger.info("Successfully initialized binary classification miner")
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            # Create a minimal configuration as a fallback
            fallback_config = bt.config()
            fallback_config.miner = bt.config()
            fallback_config.miner.device = 'cpu'
            fallback_config.miner.input_size = 40
            fallback_config.miner.hidden_sizes = [64, 32, 1]
            self.classifier = BinaryClassificationMiner(fallback_config)
            logger.warning("Using fallback classifier configuration")
        
        # List of 40 feature names for reference
        self.features = [
            'conversation_duration_seconds', 'conversation_duration_minutes', 'hour_of_day', 'day_of_week',
            'is_business_hours', 'is_weekend', 'time_to_first_response_seconds', 'avg_response_time_seconds',
            'max_response_time_seconds', 'min_response_time_seconds', 'avg_agent_response_time_seconds',
            'avg_user_response_time_seconds', 'response_time_stddev', 'response_gap_max', 'messages_per_minute',
            'total_messages', 'user_messages_count', 'agent_messages_count', 'message_ratio',
            'avg_message_length_user', 'max_message_length_user', 'min_message_length_user', 'total_chars_from_user',
            'avg_message_length_agent', 'max_message_length_agent', 'min_message_length_agent', 'total_chars_from_agent',
            'question_count_agent', 'questions_per_agent_message', 'question_count_user', 'questions_per_user_message',
            'sequential_user_messages', 'sequential_agent_messages', 'entities_collected_count', 'has_target_entity',
            'avg_entity_confidence', 'min_entity_confidence', 'entity_collection_rate', 'repeated_questions',
            'message_alternation_rate'
        ]

        logger.info("Miner initialized with binary classification model")

    async def forward(self, synapse: ConversionSynapse) -> ConversionSynapse:
        """
        Processes an incoming ConversionSynapse request, predicting task outcomes based on conversation features.

        This implementation uses a binary classification model to predict conversion outcomes.
        - conversion_happened: Binary (0 or 1)
        - time_to_conversion_seconds: Float (positive if conversion_happened=1, -1.0 otherwise)
        - confidence: Float (probability for conversion_happened)

        Args:
            synapse (ConversionSynapse): Synapse object containing 40 conversation features.

        Returns:
            ConversionSynapse: Synapse with predictions (conversion_happened, time_to_conversion_seconds) and confidence.
        """
        try:
            # Use the binary classification model for prediction
            result = self.classifier.forward(synapse)

            # Attach predictions to synapse
            synapse.prediction = {
                'conversion_happened': result['conversion_happened'],
                'time_to_conversion_seconds': result['time_to_conversion_seconds']
            }
            synapse.confidence = result.get('confidence', 0.5)

            logger.debug(f"Processed synapse for hotkey {synapse.dendrite.hotkey}: {synapse.prediction}")

        except Exception as e:
            logger.error(f"Error in forward: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return a valid default prediction instead of invalid -999 values
            synapse.prediction = {
                'conversion_happened': 0,
                'time_to_conversion_seconds': -1.0
            }
            synapse.confidence = 0.0

        return synapse

    async def blacklist(self, synapse: ConversionSynapse) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted.

        Args:
            synapse (ConversionSynapse): Synapse object with request metadata.

        Returns:
            Tuple[bool, str]: (True, reason) if blacklisted, (False, reason) if not.
        """
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            logger.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"
        
        logger.trace(f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized"

    async def priority(self, synapse: ConversionSynapse) -> float:
        """
        Determines the priority of an incoming request based on the caller's stake.

        Args:
            synapse (ConversionSynapse): Synapse object with request metadata.

        Returns:
            float: Priority score (higher means processed sooner).
        """
        try:
            caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            priority = float(self.metagraph.S[caller_uid])  # Stake-based priority
            logger.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
            return priority
        except Exception as e:
            logger.error(f"Error in priority: {e}")
            return 0.0

if __name__ == "__main__":
    with Miner() as miner:
        while True:
            logger.info("Miner running...", time=time.time())
            time.sleep(5)

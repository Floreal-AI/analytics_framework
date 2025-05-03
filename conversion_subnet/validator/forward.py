import time
import bittensor as bt
import numpy as np
from typing import Dict, List

from conversion_subnet.protocol import ConversionSynapse, ConversationFeatures, PredictionOutput
from conversion_subnet.validator.reward import Validator
from conversion_subnet.utils.uids import get_random_uids
from conversion_subnet.validator.generate import generate_conversation
from conversion_subnet.validator.utils import validate_features, log_metrics
from conversion_subnet.utils.log import logger
from conversion_subnet.constants import (
    TIMEOUT_SEC, SAMPLE_SIZE, REQUIRED_FEATURES,
    ENTITY_THRESHOLD, MESSAGE_RATIO_THRESHOLD, MIN_CONVERSATION_DURATION,
    TIME_SCALE_FACTOR, MIN_CONVERSION_TIME
)

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It queries the network with real-time conversation features and scores miner predictions.

    Args:
        self: The validator neuron object containing state (e.g., metagraph, dendrite, config).
    """
    # Select a subset of miners to query
    sample_size = getattr(self.config.neuron, 'sample_size', SAMPLE_SIZE)
    miner_uids = get_random_uids(self, k=sample_size)

    # Generate synthetic conversation features
    conversation = generate_conversation()
    features = validate_features(conversation)
    
    # Store the features for ground truth generation
    self.conversation_history = getattr(self, 'conversation_history', {})
    self.conversation_history[features['session_id']] = features

    # Create ConversionSynapse with features
    synapse = ConversionSynapse(features=features)

    # Query miners and measure response time
    start_time = time.time()
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=True,
        timeout=TIMEOUT_SEC
    )
    end_time = time.time()

    # Update response times in synapses
    for response, uid in zip(responses, miner_uids):
        response.response_time = end_time - start_time
        response.miner_uid = uid

    # Log responses for monitoring
    logger.info(f"Received responses: {[r.prediction for r in responses if r.prediction is not None]}")

    # Generate ground truth based on conversation features
    ground_truth = generate_ground_truth(features)
    
    # Score responses using the Incentive Mechanism
    score_validator = Validator()
    rewards = []
    for response in responses:
        if response.prediction is None or not response.prediction:
            reward = 0.0
        else:
            # Validate prediction format
            if not validate_prediction(response.prediction):
                logger.warning(f"Invalid prediction format from miner {response.miner_uid}: {response.prediction}")
                reward = 0.0
            else:
                reward = score_validator.reward(ground_truth, response)
                log_metrics(response, reward, ground_truth)  # Log detailed metrics
        rewards.append(reward)

    # Convert rewards to numpy array for weight updates
    rewards = np.array(rewards, dtype=np.float32)

    # Log scored responses
    logger.info(f"Scored responses: {rewards}")

    # Update miner scores based on rewards
    self.update_scores(rewards, miner_uids)

def generate_ground_truth(features: ConversationFeatures) -> PredictionOutput:
    """
    Generate ground truth based on conversation features.
    This implements a deterministic rule-based approach that miners can learn.
    
    Args:
        features (ConversationFeatures): Conversation features
        
    Returns:
        PredictionOutput: Ground truth with conversion_happened and time_to_conversion_seconds
    """
    # Determine if conversion happened based on key features
    has_target = features.get('has_target_entity', 0) == 1
    entities_count = features.get('entities_collected_count', 0)
    message_ratio = features.get('message_ratio', 0)
    conversation_duration = features.get('conversation_duration_seconds', 0)
    
    # Rule 1: Has target entity and collected enough entities
    conversion_rule1 = has_target and entities_count >= ENTITY_THRESHOLD
    
    # Rule 2: Good message ratio (agent asks more questions) and conversation is long enough
    conversion_rule2 = message_ratio > MESSAGE_RATIO_THRESHOLD and conversation_duration > MIN_CONVERSATION_DURATION
    
    # Conversion happens if either rule is met
    conversion_happened = 1 if (conversion_rule1 or conversion_rule2) else 0
    
    # Calculate time to conversion if conversion happened
    if conversion_happened == 1:
        # Base time is conversation_duration * scale_factor
        base_time = conversation_duration * TIME_SCALE_FACTOR
        
        # Adjust based on features 
        adjustment = 10 if has_target else 0
        adjustment -= 5 * max(0, entities_count - 3)  # Faster with more entities
        adjustment += 5 * (1.0 - min(1.0, message_ratio / 2.0))  # Faster with better message ratio
        
        time_to_conversion = max(MIN_CONVERSION_TIME, base_time + adjustment)
    else:
        time_to_conversion = -1.0
        
    return {
        'conversion_happened': conversion_happened,
        'time_to_conversion_seconds': time_to_conversion
    }

def validate_prediction(prediction: Dict) -> bool:
    """
    Validate the format of a miner's prediction.
    
    Args:
        prediction (Dict): Miner's prediction
        
    Returns:
        bool: True if prediction is valid, False otherwise
    """
    # Check if required keys exist
    if 'conversion_happened' not in prediction or 'time_to_conversion_seconds' not in prediction:
        return False
        
    # Check if conversion_happened is binary (0 or 1)
    if prediction['conversion_happened'] not in [0, 1]:
        return False
        
    # Check if time_to_conversion_seconds is valid (positive float or -1.0)
    if prediction['conversion_happened'] == 1:
        if not isinstance(prediction['time_to_conversion_seconds'], (int, float)) or prediction['time_to_conversion_seconds'] <= 0:
            return False
    else:
        if prediction['time_to_conversion_seconds'] != -1.0:
            return False
            
    return True

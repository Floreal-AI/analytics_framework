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

def preprocess_features(features: Dict) -> Dict:
    """
    Preprocess features to ensure integer fields are properly converted.
    
    Args:
        features (Dict): Dictionary of conversation features
        
    Returns:
        Dict: Features with integer fields converted to integers
    """
    # Define integer fields
    integer_fields = [
        'hour_of_day', 'day_of_week', 'is_business_hours', 'is_weekend',
        'total_messages', 'user_messages_count', 'agent_messages_count',
        'max_message_length_user', 'min_message_length_user', 'total_chars_from_user',
        'max_message_length_agent', 'min_message_length_agent', 'total_chars_from_agent',
        'question_count_agent', 'question_count_user', 'sequential_user_messages',
        'sequential_agent_messages', 'entities_collected_count', 'has_target_entity',
        'repeated_questions'
    ]
    
    # Convert integer fields
    result = features.copy()
    for field in integer_fields:
        if field in result and result[field] is not None:
            try:
                result[field] = int(result[field])
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert {field} to integer: {result[field]}, error: {e}")
                # If conversion fails, set to a reasonable default
                if field in ['max_message_length_user', 'max_message_length_agent']:
                    result[field] = 100
                elif field in ['min_message_length_user', 'min_message_length_agent']:
                    result[field] = 20
                else:
                    result[field] = 0
    
    return result

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It queries the network with real-time conversation features and scores miner predictions.

    Args:
        self: The validator neuron object containing state (e.g., metagraph, dendrite, config).
    """
    try:
        # Select a subset of miners to query
        sample_size = getattr(self.config.neuron, 'sample_size', SAMPLE_SIZE)
        miner_uids = get_random_uids(self, k=sample_size)

        # Generate synthetic conversation features
        conversation = generate_conversation()
        features = validate_features(conversation)
        
        # Preprocess features to ensure integer fields are properly converted
        features = preprocess_features(features)
        
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
        for i, (response, uid) in enumerate(zip(responses, miner_uids)):
            # Ensure response is a proper ConversionSynapse object
            if isinstance(response, dict):
                # Convert dict to ConversionSynapse
                temp_synapse = ConversionSynapse(features={})
                
                # Copy over any existing attributes
                if 'prediction' in response:
                    temp_synapse.set_prediction(response.get('prediction'))
                
                if 'confidence' in response:
                    temp_synapse.confidence = response.get('confidence')
                else:
                    temp_synapse.confidence = 0.5
                
                responses[i] = temp_synapse
                response = temp_synapse
            
            # Now ensure all necessary attributes are set
            if not hasattr(response, 'response_time') or response.response_time is None:
                response.response_time = end_time - start_time
                
            if not hasattr(response, 'miner_uid') or response.miner_uid is None:
                response.miner_uid = uid
                
            if not hasattr(response, 'confidence') or response.confidence is None:
                response.confidence = 0.5
                
            # Use set_prediction for empty predictions (it handles None values)
            if not hasattr(response, 'prediction') or response.prediction is None:
                response.set_prediction({})
            elif isinstance(response.prediction, dict) and (not response.prediction or
                    'conversion_happened' not in response.prediction or
                    'time_to_conversion_seconds' not in response.prediction):
                # If prediction is an incomplete dictionary, process it
                response.set_prediction(response.prediction)

        # Log responses for monitoring
        logger.info(f"Received responses: {[r.prediction for r in responses if r.prediction is not None]}")

        # Generate ground truth based on conversation features
        ground_truth = generate_ground_truth(features)
        
        # Score responses using the Incentive Mechanism
        score_validator = Validator()
        rewards = []
        for response in responses:
            try:
                if not response.prediction:
                    reward = 0.0
                else:
                    # Use validate_prediction directly as prediction is already preprocessed by set_prediction
                    if not validate_prediction(response.prediction):
                        logger.warning(f"Invalid prediction format from miner {response.miner_uid}: {response.prediction}")
                        reward = 0.0
                    else:
                        reward = score_validator.reward(ground_truth, response)
                        log_metrics(response, reward, ground_truth)  # Log detailed metrics
            except Exception as e:
                logger.error(f"Error scoring response: {e}")
                reward = 0.0
                
            rewards.append(reward)

        # Convert rewards to numpy array for weight updates
        rewards = np.array(rewards, dtype=np.float32)

        # Log scored responses
        logger.info(f"Scored responses: {rewards}")

        # Convert to torch tensor for update_scores
        try:
            import torch
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            self.update_scores(rewards_tensor, miner_uids)
        except Exception as e:
            logger.error(f"Error converting rewards to tensor: {e}. Scores not updated.")
            # Try fallback to numpy array update
            try:
                self.update_scores(rewards, miner_uids)
            except Exception as e2:
                logger.error(f"Error in fallback update: {e2}. Scores not updated at all.")
    except Exception as e:
        logger.error(f"Error in forward function: {e}")
        # Don't re-raise to ensure the validator continues running

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
    
    try:
        # Try to convert conversion_happened to int if it's not already
        conversion_happened = int(prediction['conversion_happened'])
        
        # Check if conversion_happened is binary (0 or 1)
        if conversion_happened not in [0, 1]:
            return False
            
        # Update the prediction with the integer value
        prediction['conversion_happened'] = conversion_happened
            
        # Check if time_to_conversion_seconds is valid (positive float or -1.0)
        time_to_conversion = float(prediction['time_to_conversion_seconds'])
        prediction['time_to_conversion_seconds'] = time_to_conversion
        
        if conversion_happened == 1:
            if time_to_conversion <= 0:
                return False
        else:
            if time_to_conversion != -1.0:
                return False
                
        return True
    except (ValueError, TypeError):
        # Failed to convert types
        return False

def preprocess_prediction(prediction: Dict) -> Dict:
    """
    Preprocess prediction to ensure fields are properly converted to their correct types.
    
    Args:
        prediction (Dict): Dictionary with prediction data
        
    Returns:
        Dict: Prediction with fields converted to correct types
    """
    if not prediction:
        # Return a valid default prediction structure when prediction is empty
        return {
            'conversion_happened': 0,
            'time_to_conversion_seconds': -1.0
        }
        
    result = prediction.copy()
    
    # Ensure required fields exist
    if 'conversion_happened' not in result:
        result['conversion_happened'] = 0
    
    if 'time_to_conversion_seconds' not in result:
        result['time_to_conversion_seconds'] = -1.0
    
    # Handle conversion_happened - should be an integer
    try:
        result['conversion_happened'] = int(result['conversion_happened'])
    except (ValueError, TypeError):
        # Default to 0 if conversion fails
        result['conversion_happened'] = 0
    
    # Handle time_to_conversion_seconds - should be a float
    try:
        result['time_to_conversion_seconds'] = float(result['time_to_conversion_seconds'])
    except (ValueError, TypeError):
        # Default to -1.0 if conversion fails
        result['time_to_conversion_seconds'] = -1.0
    
    return result

"""
Forward pass implementation for validators.

This module handles the core validator logic:
1. Generate conversation features or get real data
2. Query miners for predictions
3. Validate predictions against external API
4. Calculate and assign rewards
"""

import time
import uuid
import numpy as np
from typing import Dict, List, Optional

import bittensor as bt

from conversion_subnet.constants import SAMPLE_SIZE, TIMEOUT_SEC
from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.utils.log import logger
from conversion_subnet.validator.generate import generate_conversation
from conversion_subnet.validator.utils import validate_features, log_metrics
from conversion_subnet.validator.reward import Validator
from conversion_subnet.utils.uids import get_random_uids
from conversion_subnet.validator.validation_client import (
    get_default_validation_client, 
    ValidationError,
    ValidationResult
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

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It queries the network with real-time conversation features and scores miner predictions
    using external validation API.

    Args:
        self: The validator neuron object containing state (e.g., metagraph, dendrite, config).
    """
    try:
        # Select a subset of miners to query
        sample_size = getattr(self.config.neuron, 'sample_size', SAMPLE_SIZE)
        miner_uids = get_random_uids(self, k=sample_size)

        # Generate unique test_pk for this round
        test_pk = str(uuid.uuid4())
        logger.info(f"Starting validation round with test_pk: {test_pk}")

        # Generate synthetic conversation features
        conversation = generate_conversation()
        features = validate_features(conversation)
        
        # Preprocess features to ensure integer fields are properly converted
        features = preprocess_features(features)
        
        # Add test_pk to features for tracking
        features['test_pk'] = test_pk
        
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
        logger.info(f"Received {len(responses)} responses for test_pk: {test_pk}")
        for i, r in enumerate(responses):
            logger.debug(f"Miner {miner_uids[i]} prediction: {r.prediction}")

        # Get external validation result
        validation_result = await get_external_validation(test_pk)
        
        if validation_result is None:
            logger.error(f"Failed to get validation result for test_pk: {test_pk}")
            # Fallback to synthetic ground truth
            ground_truth = generate_ground_truth(features)
            logger.warning("Using synthetic ground truth as fallback")
        else:
            # Convert external validation to ground truth format
            ground_truth = {
                'conversion_happened': 1 if validation_result.labels else 0,
                'time_to_conversion_seconds': 60.0 if validation_result.labels else -1.0  # Default time
            }
            logger.info(f"External validation result: conversion_happened={ground_truth['conversion_happened']}")
        
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
        logger.info(f"Scored responses for test_pk {test_pk}: {rewards}")

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
        logger.error(f"Error in forward pass: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def get_external_validation(test_pk: str) -> Optional[ValidationResult]:
    """
    Get external validation result for a test_pk.
    
    Args:
        test_pk: Test primary key to validate
        
    Returns:
        ValidationResult: The validation result, or None if failed
    """
    try:
        client = get_default_validation_client()
        result = await client.get_validation_result(test_pk)
        logger.info(f"External validation successful for test_pk {test_pk}: labels={result.labels}")
        return result
    except ValidationError as e:
        logger.error(f"Validation API error for test_pk {test_pk}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in external validation for test_pk {test_pk}: {e}")
        return None

def generate_ground_truth(features: Dict) -> Dict:
    """
    Generate ground truth based on conversation features.
    This implements a deterministic rule-based approach that miners can learn.
    
    Args:
        features: Conversation features
        
    Returns:
        Dict: Ground truth with conversion_happened and time_to_conversion_seconds
    """
    # Constants for ground truth rules
    ENTITY_THRESHOLD = 4
    MESSAGE_RATIO_THRESHOLD = 1.2
    MIN_CONVERSATION_DURATION = 90.0
    TIME_SCALE_FACTOR = 0.7
    MIN_CONVERSION_TIME = 30.0
    
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

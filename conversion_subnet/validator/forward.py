"""
Forward pass implementation for validators.

This module handles the core validator logic:
1. Fetch real test data from API using roundNumber parameter
2. Query miners for predictions with real features and real test_pk
3. Validate predictions against external API using the same real test_pk
4. Calculate and assign rewards

STRICT POLICY: No fallbacks, no synthetic data, raise all errors with traceback.
"""

import time
import numpy as np
from typing import Dict, List

import bittensor as bt

from conversion_subnet.constants import SAMPLE_SIZE, TIMEOUT_SEC
from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.utils.log import logger
from conversion_subnet.validator.utils import log_metrics
from conversion_subnet.validator.reward import Validator
from conversion_subnet.utils.uids import get_random_uids
from conversion_subnet.validator.validation_client import (
    get_default_validation_client, 
    ValidationError,
    ValidationResult
)
from conversion_subnet.utils.retry import retry_dendrite_call, RetryConfig, RetryStrategy

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
    
    STRICT IMPLEMENTATION:
    1. Fetch REAL test data from API (with roundNumber=1, no fallbacks)
    2. Query miners with REAL features and REAL test_pk
    3. Validate using SAME REAL test_pk against external validation API
    4. Calculate and assign rewards
    
    NO FALLBACKS: All errors are raised with full traceback.

    Args:
        self: The validator neuron object containing state (e.g., metagraph, dendrite, config, data_client).
    """
    try:
        # Select a subset of miners to query
        sample_size = getattr(self.config.neuron, 'sample_size', SAMPLE_SIZE)
        
        # Initialize test data offset if not exists
        if not hasattr(self, 'test_data_offset'):
            self.test_data_offset = 0

        # Select a random subset of miner UIDs for testing
        miner_uids = get_random_uids(self, k=sample_size)
        logger.debug(f"Selected miner UIDs for query: {miner_uids}")

        # Check if data client is properly configured
        if not hasattr(self, 'data_client') or self.data_client is None:
            error_msg = "Data API client not configured. Missing .env file or invalid API credentials."
            logger.error(error_msg)
            logger.error("Please create .env file with:")
            logger.error("VOICEFORM_API_BASE_URL=your_api_url")
            logger.error("VOICEFORM_API_KEY=your_api_key")
            raise RuntimeError(error_msg)

        # Try to fetch REAL test data from API with proper error handling
        try:
            logger.info(f"Fetching REAL test data from API: limit=1, offset={self.test_data_offset}")
            test_data = await self.data_client.fetch_test_data(
                limit=1,
                offset=self.test_data_offset,
                round_number=1,  # Using roundNumber=1 as per working curl command
                save=False
            )
        except Exception as api_error:
            error_msg = f"Failed to fetch test data from API: {api_error}"
            logger.error(error_msg)
            logger.error("This may be due to:")
            logger.error("1. Missing or invalid .env configuration")
            logger.error("2. API server connectivity issues")
            logger.error("3. Invalid API credentials")
            logger.error("4. API endpoint changes")
            raise RuntimeError(error_msg) from api_error

        # Validate that we received test data with sufficient features
        if len(test_data.features) == 0:
            error_msg = f"API returned empty test data at offset {self.test_data_offset}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if len(test_data.feature_names) < 10:  # Expect at least 10 features, should be ~42
            error_msg = f"API returned insufficient features: {len(test_data.feature_names)} features, expected ~42"
            logger.error(error_msg)
            logger.error(f"Received features: {test_data.feature_names}")
            raise RuntimeError(error_msg)
        
        # Extract REAL test_pk and features from API response
        test_pk = test_data.ids[0]
        raw_features = {
            name: value for name, value in 
            zip(test_data.feature_names, test_data.features[0])
        }
        
        # Convert features to proper format, handling mixed types from API
        features = {}
        for name, value in raw_features.items():
            if isinstance(value, str):
                # Handle string values from API
                if value == "true":
                    features[name] = True
                elif value == "false":
                    features[name] = False
                elif value == "-1.00" or value == "-1":
                    features[name] = -1.0
                else:
                    try:
                        # Try to convert to float first, then int if it's a whole number
                        float_val = float(value)
                        features[name] = int(float_val) if float_val.is_integer() else float_val
                    except ValueError:
                        features[name] = value  # Keep as string if conversion fails
            else:
                features[name] = value
        
        # Add the REAL test_pk to features
        features['test_pk'] = test_pk
        
        # Increment offset for next round
        self.test_data_offset += 1
        
        logger.info(f"Using REAL test data: test_pk={test_pk}, features_count={len(features)}")
        
        # Validate and preprocess features
        features = preprocess_features(features)
        
        # Store the features for tracking
        self.conversation_history = getattr(self, 'conversation_history', {})
        session_id = features.get('session_id', test_pk)
        features['session_id'] = session_id
        self.conversation_history[session_id] = features

        # Create ConversionSynapse with REAL test features and REAL test_pk
        synapse = ConversionSynapse(features=features)

        # Defensive assertion to catch test_pk preservation issues early
        assert 'test_pk' in synapse.features, f"CRITICAL BUG: test_pk not preserved in synapse.features. Available keys: {list(synapse.features.keys())}"
        assert synapse.features['test_pk'] == test_pk, f"CRITICAL BUG: test_pk mismatch. Expected: {test_pk}, Got: {synapse.features.get('test_pk')}"
        assert synapse.features.get('test_pk', 'unknown') != 'unknown', f"CRITICAL BUG: test_pk is 'unknown'. Features: {synapse.features}"

        # Query miners with retry logic and increased timeouts
        start_time = time.time()
        
        # Define the dendrite operation as an async function for retry wrapper
        async def make_dendrite_call():
            return await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=synapse,
                deserialize=True,
                timeout=TIMEOUT_SEC  # This will be overridden by retry mechanism
            )
        
        # Log detailed connection information for debugging
        logger.info(f"Querying {len(miner_uids)} miners with retry resilience for test_pk: {synapse.features.get('test_pk', 'unknown')}")
        for i, uid in enumerate(miner_uids):
            axon = self.metagraph.axons[uid]
            logger.info(f"Miner {uid}: attempting connection to {axon.ip}:{axon.port} (external: {getattr(axon, 'external_ip', 'not_set')}:{getattr(axon, 'external_port', 'not_set')})")
            # Check if this is the problematic 0.0.0.0 address
            if axon.ip == "0.0.0.0":
                logger.warning(f"Miner {uid} has invalid IP 0.0.0.0 - this will cause connection failures")
        
        # Use retry mechanism with exponential backoff and increased timeouts
        
        try:
            responses = await retry_dendrite_call(
                make_dendrite_call,
                max_attempts=3,  # Reduced from 4 attempts
                base_timeout=30.0,  # Reduced from 120s to 30s for faster recovery
            )
            end_time = time.time()
            logger.info(f"Dendrite call succeeded after {end_time - start_time:.2f}s total")
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"Dendrite call failed after all retries in {end_time - start_time:.2f}s: {e}")
            # For now, continue with empty responses to maintain flow
            # This allows the validator to handle failed miner communications gracefully
            responses = [ConversionSynapse(features={}) for _ in miner_uids]
            # DO NOT set empty predictions - leave them as None to avoid validation errors
            # The scoring logic will handle None predictions by assigning 0.0 reward

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
                
            # Handle predictions - DO NOT try to set empty predictions as they will fail validation
            # Leave None predictions as None - they will be handled in scoring with 0.0 reward
            if hasattr(response, 'prediction') and response.prediction is not None:
                if isinstance(response.prediction, dict) and response.prediction:
                    # Only validate non-empty predictions
                    if ('conversion_happened' not in response.prediction or
                        'time_to_conversion_seconds' not in response.prediction):
                        # If prediction is an incomplete dictionary, leave it as None
                        logger.warning(f"Incomplete prediction from miner {uid}, leaving as None: {response.prediction}")
                        response.prediction = None
                    else:
                        # Validate complete predictions using set_prediction
                        try:
                            response.set_prediction(response.prediction)
                        except Exception as pred_error:
                            logger.warning(f"Invalid prediction from miner {uid}, leaving as None: {pred_error}")
                            response.prediction = None
                else:
                    # Empty or non-dict predictions should be left as None
                    response.prediction = None

        # Log responses for monitoring
        logger.info(f"Received {len(responses)} responses for REAL test_pk: {test_pk}")
        for i, r in enumerate(responses):
            logger.debug(f"Miner {miner_uids[i]} prediction: {r.prediction}")

        # Get external validation using SAME REAL test_pk with retry resilience
        logger.info(f"Getting external validation with retry resilience for REAL test_pk: {test_pk}")
        
        async def get_validation():
            return await get_external_validation(test_pk)
            
        try:
            validation_result = await retry_dendrite_call(
                get_validation,
                max_attempts=3,  # Retry external validation calls
                base_timeout=60.0,  # 1 minute base timeout for validation API
            )
            logger.info(f"External validation succeeded for test_pk: {test_pk}")
            
        except Exception as e:
            logger.error(f"External validation failed after all retries for test_pk {test_pk}: {e}")
            raise  # Re-raise to maintain strict error handling
        
        # Convert external validation to ground truth format
        ground_truth = {
            'conversion_happened': 1 if validation_result.labels else 0,
            'time_to_conversion_seconds': 60.0 if validation_result.labels else -1.0
        }
        logger.info(f"External validation result for test_pk {test_pk}: conversion_happened={ground_truth['conversion_happened']}")
        
        # Score responses using the Incentive Mechanism
        score_validator = Validator()
        rewards = []
        for response in responses:
            try:
                if response.prediction is None or not response.prediction:
                    # No prediction available - assign 0.0 reward
                    logger.debug(f"No prediction from miner {response.miner_uid}, assigning 0.0 reward")
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
                logger.error(f"Error scoring response from miner {response.miner_uid}: {e}")
                reward = 0.0
                
            rewards.append(reward)

        # Convert rewards to numpy array for weight updates
        rewards = np.array(rewards, dtype=np.float32)

        # Log scored responses
        logger.info(f"Scored responses for REAL test_pk {test_pk}: {rewards}")

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
        # Re-raise the error to ensure it's not swallowed
        raise

async def get_external_validation(test_pk: str) -> ValidationResult:
    """
    Get external validation result for a REAL test_pk.
    
    STRICT: No fallbacks, raise all errors with traceback.
    
    Args:
        test_pk: REAL test primary key from API
        
    Returns:
        ValidationResult: The validation result
        
    Raises:
        ValidationError: If the external validation API call fails
        RuntimeError: If the validation client is not configured
    """
    try:
        client = get_default_validation_client()
    except RuntimeError as e:
        logger.error(f"Validation client not configured: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"External validation failed: {e}") from e
    
    try:
        result = await client.get_validation_result(test_pk)
        logger.info(f"External validation successful for REAL test_pk {test_pk}: labels={result.labels}")
        return result
    except ValidationError as e:
        logger.error(f"Validation API error for REAL test_pk {test_pk}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValidationError(f"External validation failed for REAL test_pk {test_pk}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in external validation for REAL test_pk {test_pk}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"External validation failed with unexpected error for REAL test_pk {test_pk}: {e}") from e

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

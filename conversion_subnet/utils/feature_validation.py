"""
Feature validation utilities for the conversion_subnet package.

This module provides functions to validate conversation features before processing
them in miners and validators, ensuring all required fields are present and have
the correct data types.
"""

from typing import Dict, List, Any, Optional, Set, Tuple

from conversion_subnet.protocol import ConversationFeatures
from conversion_subnet.constants import REQUIRED_FEATURES
from conversion_subnet.utils.log import logger

def validate_features(features: Dict[str, Any]) -> ConversationFeatures:
    """
    Validate that the given features dictionary contains all required features
    and that they have the correct data types.
    
    Args:
        features (Dict[str, Any]): Dictionary of conversation features
        
    Returns:
        ConversationFeatures: Validated features
        
    Raises:
        ValueError: If required features are missing or have invalid types
    """
    # Check for required features
    missing_features = [f for f in REQUIRED_FEATURES if f not in features]
    if missing_features:
        error_msg = f"Missing required features: {missing_features}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Type validation for key fields
    type_errors = []
    
    # Check numeric fields
    for numeric_field in ['conversation_duration_seconds', 'entities_collected_count', 'message_ratio']:
        if numeric_field in features and not isinstance(features[numeric_field], (int, float)):
            type_errors.append(f"{numeric_field} must be numeric, got {type(features[numeric_field])}")
    
    # Check binary fields
    for binary_field in ['has_target_entity']:
        if binary_field in features and features[binary_field] not in [0, 1]:
            type_errors.append(f"{binary_field} must be 0 or 1, got {features[binary_field]}")
    
    # Check string fields
    for string_field in ['session_id']:
        if string_field in features and not isinstance(features[string_field], str):
            type_errors.append(f"{string_field} must be a string, got {type(features[string_field])}")
    
    if type_errors:
        error_msg = f"Feature type validation errors: {type_errors}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # All validations passed, return as ConversationFeatures type
    return features

def find_missing_or_invalid_features(
    features: Dict[str, Any], required: Optional[List[str]] = None
) -> Tuple[Set[str], List[str]]:
    """
    Find missing or invalid features in the given dictionary.
    
    Args:
        features (Dict[str, Any]): Dictionary of conversation features
        required (Optional[List[str]]): List of required features, defaults to REQUIRED_FEATURES
        
    Returns:
        Tuple[Set[str], List[str]]: 
            - Set of missing features
            - List of validation error messages
    """
    required = required or REQUIRED_FEATURES
    missing = set(required) - set(features.keys())
    validation_errors = []
    
    # Type validation
    for field in set(features.keys()).intersection(required):
        if field == 'session_id' and not isinstance(features[field], str):
            validation_errors.append(f"{field} must be string")
        elif field in ['conversation_duration_seconds', 'message_ratio'] and not isinstance(features[field], (int, float)):
            validation_errors.append(f"{field} must be numeric")
        elif field == 'has_target_entity' and features[field] not in [0, 1]:
            validation_errors.append(f"{field} must be 0 or 1")
        elif field == 'entities_collected_count' and not isinstance(features[field], int):
            validation_errors.append(f"{field} must be integer")
            
    return missing, validation_errors

def get_numeric_features(features: ConversationFeatures) -> List[float]:
    """
    Extract all numeric features from the conversation features as a list,
    which can be used for model input.
    
    Args:
        features (ConversationFeatures): Dictionary of conversation features
        
    Returns:
        List[float]: List of numeric feature values in a consistent order
    """
    # Define the order of features for consistency
    feature_order = [
        'conversation_duration_seconds', 'conversation_duration_minutes', 'hour_of_day', 
        'day_of_week', 'is_business_hours', 'is_weekend', 'time_to_first_response_seconds', 
        'avg_response_time_seconds', 'max_response_time_seconds', 'min_response_time_seconds', 
        'avg_agent_response_time_seconds', 'avg_user_response_time_seconds', 'response_time_stddev', 
        'response_gap_max', 'messages_per_minute', 'total_messages', 'user_messages_count', 
        'agent_messages_count', 'message_ratio', 'avg_message_length_user', 'max_message_length_user', 
        'min_message_length_user', 'total_chars_from_user', 'avg_message_length_agent', 
        'max_message_length_agent', 'min_message_length_agent', 'total_chars_from_agent', 
        'question_count_agent', 'questions_per_agent_message', 'question_count_user', 
        'questions_per_user_message', 'sequential_user_messages', 'sequential_agent_messages', 
        'entities_collected_count', 'has_target_entity', 'avg_entity_confidence', 
        'min_entity_confidence', 'entity_collection_rate', 'repeated_questions', 
        'message_alternation_rate'
    ]
    
    # Extract and convert features to float
    result = []
    for feature in feature_order:
        if feature in features:
            try:
                result.append(float(features[feature]))
            except (ValueError, TypeError):
                # Default to 0.0 for non-numeric values or missing features
                logger.warning(f"Feature {feature} could not be converted to float")
                result.append(0.0)
        else:
            # Default to 0.0 for missing features
            result.append(0.0)
    
    return result 
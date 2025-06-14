"""
Feature validation utilities for the conversion_subnet package.

This module provides functions to validate conversation features before processing
them in miners and validators, ensuring all required fields are present and have
the correct data types.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
import re
import uuid

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

def is_uuid_or_id_column(key: str, value: Any) -> bool:
    """
    Detect if a column is a UUID or ID column that should be excluded from features.
    
    Args:
        key (str): Column name
        value (Any): Column value
        
    Returns:
        bool: True if this is an ID/UUID column
    """
    # Check if key indicates it's an ID - be more specific
    id_keywords = ['id', 'uuid', 'pk', 'key', 'session']
    key_lower = key.lower()
    
    # Check for exact matches or standalone ID keywords
    if any(keyword == key_lower for keyword in id_keywords):
        return True
    
    # Check for ID keywords that are standalone or at the end
    if any(key_lower.endswith('_' + keyword) or key_lower.startswith(keyword + '_') for keyword in id_keywords):
        return True
    
    # Specific patterns that are definitely IDs
    if key_lower in ['test_pk', 'primary_key', 'session_id']:
        return True
    
    # Don't exclude features that contain common statistical terms even if they have ID-like keywords
    statistical_terms = ['avg', 'min', 'max', 'mean', 'std', 'count', 'sum', 'ratio', 'rate', 'confidence']
    if any(term in key_lower for term in statistical_terms):
        return False
    
    # Check if value looks like a UUID (only if it's a string)
    if isinstance(value, str):
        # Try to parse as UUID
        try:
            uuid.UUID(value)
            return True
        except (ValueError, TypeError):
            pass
        
        # Check if it's alphanumeric with dashes/underscores (common ID format)
        # But only if it's reasonably long and looks like an ID
        if re.match(r'^[a-zA-Z0-9\-_]+$', value) and len(value) > 10:
            # Additional check: if it contains only numbers, it might be a numeric feature
            if value.replace('.', '').replace('-', '').isdigit():
                return False
            return True
    
    return False

def extract_numeric_features_only(features: ConversationFeatures) -> Dict[str, float]:
    """
    Extract only numeric features, excluding all ID/UUID columns.
    
    Args:
        features (ConversationFeatures): Input features dictionary
        
    Returns:
        Dict[str, float]: Dictionary with only numeric features
        
    Raises:
        AssertionError: If feature validation fails
    """
    assert isinstance(features, dict), f"Features must be a dictionary, got {type(features)}"
    assert len(features) > 0, "Features dictionary cannot be empty"
    
    numeric_features = {}
    excluded_columns = []
    
    logger.debug(f"Processing {len(features)} input features: {list(features.keys())}")
    
    for key, value in features.items():
        logger.debug(f"Evaluating feature: {key}={value} (type: {type(value)})")
        
        # Skip ID/UUID columns
        if is_uuid_or_id_column(key, value):
            excluded_columns.append(f"{key}={value}")
            logger.debug(f"Excluding ID column: {key}={value}")
            continue
            
        # Skip non-numeric values
        if isinstance(value, str) and not is_uuid_or_id_column(key, value):
            # This shouldn't happen - raise error instead of silently skipping
            logger.error(f"Non-UUID string value found in features: {key}={value}")
            raise ValueError(f"Non-UUID string value found in features: {key}={value}")
        
        # Convert to numeric
        if value is None:
            numeric_features[key] = 0.0
            logger.debug(f"Converted None to 0.0 for {key}")
        else:
            try:
                float_value = float(value)
                numeric_features[key] = float_value
                logger.debug(f"Added numeric feature: {key}={float_value}")
            except (ValueError, TypeError) as e:
                logger.error(f"Cannot convert feature {key}={value} to float: {e}")
                raise ValueError(f"Cannot convert feature {key}={value} to float: {e}")
    
    logger.info(f"Extracted {len(numeric_features)} numeric features, excluded {len(excluded_columns)} ID columns")
    logger.debug(f"Excluded columns: {excluded_columns}")
    logger.debug(f"Numeric features: {sorted(numeric_features.keys())}")
    
    # Assert we have the expected number of features
    expected_feature_count = 42
    actual_feature_count = len(numeric_features)
    
    if actual_feature_count != expected_feature_count:
        logger.error(f"Feature count mismatch after ID exclusion:")
        logger.error(f"  Expected: {expected_feature_count}")
        logger.error(f"  Actual: {actual_feature_count}")
        logger.error(f"  Input features: {list(features.keys())}")
        logger.error(f"  Numeric features: {list(numeric_features.keys())}")
        logger.error(f"  Excluded columns: {excluded_columns}")
        
        # Show exactly which features are missing
        expected_features = [
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
            'message_alternation_rate', 'time_to_conversion_seconds', 'time_to_conversion_minutes'
        ]
        
        missing = set(expected_features) - set(numeric_features.keys())
        extra = set(numeric_features.keys()) - set(expected_features)
        
        if missing:
            logger.error(f"  Missing expected features: {sorted(missing)}")
        if extra:
            logger.error(f"  Extra unexpected features: {sorted(extra)}")
        
        # This is the real issue - don't hide it
        raise AssertionError(f"Feature count mismatch: expected {expected_feature_count}, got {actual_feature_count}. Missing: {missing}, Extra: {extra}")
    
    return numeric_features

def get_standardized_features_for_model(features: ConversationFeatures) -> np.ndarray:
    """
    Extract standardized numeric features for model prediction with strict validation.
    NO FALLBACKS - raises errors if data doesn't match expectations.
    
    Args:
        features (ConversationFeatures): Dictionary of conversation features
        
    Returns:
        np.ndarray: Standardized feature array with shape (1, 42) for model input
        
    Raises:
        AssertionError: If features don't match expected structure
        ValueError: If feature extraction fails
    """
    assert isinstance(features, dict), f"Features must be a dictionary, got {type(features)}"
    assert len(features) > 0, "Features dictionary cannot be empty"
    
    # Extract only numeric features, excluding IDs
    numeric_features = extract_numeric_features_only(features)
    
    # Define the exact feature order used during training (from model_report.json)
    expected_feature_order = [
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
        'message_alternation_rate', 'time_to_conversion_seconds', 'time_to_conversion_minutes'
    ]
    
    # Validate we have the expected number of features in our template
    assert len(expected_feature_order) == 42, f"Expected feature template has {len(expected_feature_order)} features, must be 42"
    
    # Extract features in the exact order used during training
    result = []
    missing_features = []
    
    for feature in expected_feature_order:
        if feature in numeric_features:
            result.append(numeric_features[feature])
        else:
            missing_features.append(feature)
            result.append(0.0)  # Use 0.0 for missing features
    
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features (using 0.0): {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
    
    # Convert to numpy array with proper shape
    features_array = np.array(result, dtype=np.float32).reshape(1, -1)
    
    # Final validation with strict assertion
    assert features_array.shape == (1, 42), f"Feature array must have shape (1, 42), got {features_array.shape}"
    assert not np.any(np.isnan(features_array)), "Feature array contains NaN values"
    assert np.all(np.isfinite(features_array)), "Feature array contains infinite values"
    
    logger.debug(f"Standardized features: {features_array.shape} - first 5 values: {result[:5]}")
    
    return features_array 
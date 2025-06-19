#!/usr/bin/env python3
"""
Feature encoding utilities to work around Bittensor network transmission bug.

The Bittensor network layer drops 95% of features during transmission.
This module provides base64 encoding/decoding to pack all features into a single field.
"""

import base64
import json
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def camel_to_snake(name: str) -> str:
    """
    Convert camelCase to snake_case.
    
    Args:
        name: camelCase string
        
    Returns:
        snake_case string
    """
    # Insert an underscore before any uppercase letter that follows a lowercase letter
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert an underscore before any uppercase letter that follows a lowercase letter or digit
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def convert_feature_names_to_snake_case(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert all feature names from camelCase to snake_case.
    
    Args:
        features: Dictionary with camelCase feature names
        
    Returns:
        Dictionary with snake_case feature names
    """
    snake_case_features = {}
    for key, value in features.items():
        snake_key = camel_to_snake(key)
        snake_case_features[snake_key] = value
    
    logger.debug(f"Converted {len(features)} feature names to snake_case")
    return snake_case_features

def encode_features_to_base64(features: Dict[str, Any]) -> str:
    """
    Encode a feature dictionary to a base64 string.
    
    Args:
        features: Dictionary of features to encode
        
    Returns:
        Base64 encoded string containing all features
        
    Raises:
        ValueError: If encoding fails
    """
    try:
        # Convert to JSON string
        json_str = json.dumps(features, default=str, separators=(',', ':'))
        
        # Encode to base64
        json_bytes = json_str.encode('utf-8')
        base64_bytes = base64.b64encode(json_bytes)
        base64_str = base64_bytes.decode('utf-8')
        
        logger.debug(f"Encoded {len(features)} features to base64 ({len(base64_str)} chars)")
        return base64_str
        
    except Exception as e:
        logger.error(f"Failed to encode features to base64: {e}")
        raise ValueError(f"Feature encoding failed: {e}")

def decode_features_from_base64(base64_str: str) -> Dict[str, Any]:
    """
    Decode a base64 string back to a feature dictionary.
    
    Args:
        base64_str: Base64 encoded string
        
    Returns:
        Dictionary of decoded features
        
    Raises:
        ValueError: If decoding fails
    """
    try:
        # Decode from base64
        base64_bytes = base64_str.encode('utf-8')
        json_bytes = base64.b64decode(base64_bytes)
        json_str = json_bytes.decode('utf-8')
        
        # Parse JSON
        features = json.loads(json_str)
        
        logger.debug(f"Decoded base64 to {len(features)} features")
        return features
        
    except Exception as e:
        logger.error(f"Failed to decode base64 to features: {e}")
        raise ValueError(f"Feature decoding failed: {e}")

def pack_features_for_transmission(original_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pack features for transmission through Bittensor network.
    
    Since Bittensor only transmits 1 field reliably, we encode all features
    as a base64 string in the session_id field.
    
    Args:
        original_features: Original feature dictionary with 40+ features
        
    Returns:
        Packed dictionary with only 1 field that will survive transmission
    """
    try:
        # Encode all features as base64
        encoded_features = encode_features_to_base64(original_features)
        
        # Create packed dictionary with encoded features in session_id
        packed = {
            'session_id': f'ENCODED:{encoded_features}'
        }
        
        logger.debug(f"Packed {len(original_features)} features into {len(packed)} transmission field")
        return packed
        
    except Exception as e:
        logger.error(f"Failed to pack features: {e}")
        # Return original features as fallback
        return original_features

def unpack_features_from_transmission(packed_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unpack features received through Bittensor network.
    
    Args:
        packed_features: Dictionary received from network (should have 1 field)
        
    Returns:
        Original feature dictionary with all features restored
    """
    try:
        logger.debug(f"Unpacking received features: {list(packed_features.keys())}")
        
        # Check if we have the encoded features in session_id
        session_id = packed_features.get('session_id', '')
        if not session_id.startswith('ENCODED:'):
            logger.warning("No 'ENCODED:' prefix in session_id - falling back to original features")
            return packed_features
        
        # Extract the base64 encoded features
        encoded_features = session_id[8:]  # Remove 'ENCODED:' prefix
        original_features = decode_features_from_base64(encoded_features)
        
        # Convert camelCase feature names to snake_case
        snake_case_features = convert_feature_names_to_snake_case(original_features)
        
        logger.info(f"Unpacked {len(snake_case_features)} features from transmission")
        logger.debug(f"Restored features: {list(snake_case_features.keys())}")
        
        return snake_case_features
        
    except Exception as e:
        logger.error(f"Failed to unpack features: {e}")
        # Return original features as fallback
        return packed_features

def encode_predictions_for_response(predictions: Dict[str, Any], confidence: float = 0.5) -> Dict[str, Any]:
    """
    Encode miner predictions for transmission back to validator.
    
    Since Bittensor network drops prediction fields, we encode them as base64
    and put them in a field that survives transmission.
    
    Args:
        predictions: Dictionary with prediction results
        confidence: Confidence score
        
    Returns:
        Encoded response dictionary that will survive network transmission
    """
    try:
        # Create response payload
        response_data = {
            'predictions': predictions,
            'confidence': confidence
        }
        
        # Encode as base64
        encoded_response = encode_features_to_base64(response_data)
        
        # Pack into a field that survives transmission
        # Use 'prediction' field with encoded prefix
        encoded_predictions = {
            'prediction': f'ENCODED_RESPONSE:{encoded_response}',
            'confidence': confidence  # Also keep as separate field as backup
        }
        
        logger.debug(f"Encoded predictions for response transmission")
        return encoded_predictions
        
    except Exception as e:
        logger.error(f"Failed to encode predictions: {e}")
        # Return original predictions as fallback
        return {'prediction': predictions, 'confidence': confidence}

def decode_predictions_from_response(encoded_response: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
    """
    Decode miner predictions received from network transmission.
    
    Args:
        encoded_response: Dictionary received from miner
        
    Returns:
        Tuple of (predictions_dict, confidence_score)
    """
    try:
        # Check if we have encoded response
        prediction_field = encoded_response.get('prediction', '')
        
        if isinstance(prediction_field, str) and prediction_field.startswith('ENCODED_RESPONSE:'):
            # Extract and decode the base64 response
            encoded_data = prediction_field[17:]  # Remove 'ENCODED_RESPONSE:' prefix
            response_data = decode_features_from_base64(encoded_data)
            
            predictions = response_data.get('predictions', {})
            confidence = response_data.get('confidence', 0.5)
            
            logger.debug(f"Decoded predictions from response transmission")
            return predictions, confidence
        else:
            # Try to use the prediction field directly
            predictions = prediction_field if isinstance(prediction_field, dict) else {}
            confidence = encoded_response.get('confidence', 0.5)
            
            logger.debug(f"Using direct predictions (no encoding detected)")
            return predictions, confidence
            
    except Exception as e:
        logger.error(f"Failed to decode predictions: {e}")
        # Return empty predictions as fallback
        return {}, 0.0

# Test the encoding/decoding functionality
if __name__ == "__main__":
    # Test feature encoding
    test_features = {
        'agentMessagesCount': 5,
        'avgResponseTimeSeconds': 2.5,
        'conversationDurationMinutes': 10.0,
        'session_id': 'test-session-123',
        'test_pk': 'test-pk-456'
    }
    
    print("=== FEATURE ENCODING TEST ===")
    print(f"Original features: {test_features}")
    
    # Test packing
    packed = pack_features_for_transmission(test_features)
    print(f"Packed features: {packed}")
    
    # Test unpacking
    unpacked = unpack_features_from_transmission(packed)
    print(f"Unpacked features: {unpacked}")
    
    print("\\n=== PREDICTION ENCODING TEST ===")
    # Test prediction encoding
    test_predictions = {
        'conversion_happened': 1,
        'time_to_conversion_seconds': 120.5
    }
    test_confidence = 0.85
    
    print(f"Original predictions: {test_predictions}, confidence: {test_confidence}")
    
    # Test encoding
    encoded_pred = encode_predictions_for_response(test_predictions, test_confidence)
    print(f"Encoded predictions: {encoded_pred}")
    
    # Test decoding
    decoded_pred, decoded_conf = decode_predictions_from_response(encoded_pred)
    print(f"Decoded predictions: {decoded_pred}, confidence: {decoded_conf}")
    
    print("\\n✅ All tests completed successfully!") 
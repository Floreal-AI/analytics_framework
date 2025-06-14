import uuid
from typing import Dict
import bittensor as bt
from conversion_subnet.protocol import ConversionSynapse

def validate_features(features: Dict) -> Dict:
    """
    Validate and correct conversation features to ensure logical consistency.

    Args:
        features (Dict): Dictionary of 40 conversation features

    Returns:
        Dict: Validated and corrected features
    """
    validated = features.copy()

    # Ensure integer fields are properly converted
    # Handle both naming conventions (snake_case and camelCase)
    integer_field_mappings = {
        'hour_of_day': 'hourOfDay',
        'day_of_week': 'dayOfWeek', 
        'is_business_hours': 'isBusinessHours',
        'is_weekend': 'isWeekend',
        'total_messages': 'totalMessages',
        'user_messages_count': 'userMessagesCount',
        'agent_messages_count': 'agentMessagesCount',
        'max_message_length_user': 'maxMessageLengthUser',
        'min_message_length_user': 'minMessageLengthUser',
        'avg_message_length_user': 'avgMessageLengthUser',
        'total_chars_from_user': 'totalCharsFromUser',
        'max_message_length_agent': 'maxMessageLengthAgent',
        'min_message_length_agent': 'minMessageLengthAgent',
        'avg_message_length_agent': 'avgMessageLengthAgent',
        'total_chars_from_agent': 'totalCharsFromAgent',
        'question_count_agent': 'questionCountAgent',
        'question_count_user': 'questionCountUser',
        'sequential_user_messages': 'sequentialUserMessages',
        'sequential_agent_messages': 'sequentialAgentMessages',
        'entities_collected_count': 'entitiesCollectedCount',
        'has_target_entity': 'hasTargetEntity',
        'repeated_questions': 'repeatedQuestions'
    }
    
        # Convert integer fields using both naming conventions
    for snake_case, camel_case in integer_field_mappings.items():
        field_key = snake_case if snake_case in validated else camel_case
        if field_key in validated and validated[field_key] is not None:
            try:
                validated[field_key] = int(validated[field_key])
                # Also ensure we have the snake_case version for consistency
                if field_key == camel_case:
                    validated[snake_case] = validated[field_key]
            except (ValueError, TypeError) as e:
                bt.logging.warning(f"Failed to convert {field_key} to integer: {validated[field_key]}, error: {e}")
                # Provide a reasonable default
                validated[field_key] = 0

    # Ensure conversation_duration_minutes is consistent
    # Handle both naming conventions
    duration_seconds_key = 'conversation_duration_seconds' if 'conversation_duration_seconds' in validated else 'conversationDurationSeconds'
    if duration_seconds_key in validated:
        validated['conversation_duration_minutes'] = round(validated[duration_seconds_key] / 60, 4)
        # Also ensure we have the standard key name
        validated['conversation_duration_seconds'] = validated[duration_seconds_key]

    # Ensure total_messages = user_messages_count + agent_messages_count
    # Handle both naming conventions for message_ratio
    message_ratio_key = 'message_ratio' if 'message_ratio' in validated else 'messageRatio'
    user_msgs = int(validated['total_messages'] / (1 + 1/validated[message_ratio_key]))
    validated['user_messages_count'] = user_msgs
    validated['agent_messages_count'] = validated['total_messages'] - user_msgs

    # Ensure total_chars_from_user and total_chars_from_agent
    # Handle both naming conventions for average message length fields
    avg_user_length_key = 'avg_message_length_user' if 'avg_message_length_user' in validated else 'avgMessageLengthUser'
    avg_agent_length_key = 'avg_message_length_agent' if 'avg_message_length_agent' in validated else 'avgMessageLengthAgent'
    
    validated['total_chars_from_user'] = int(user_msgs * validated[avg_user_length_key])
    validated['total_chars_from_agent'] = int(validated['agent_messages_count'] * validated[avg_agent_length_key])

    # Ensure is_business_hours and is_weekend
    validated['is_business_hours'] = 1 if 9 <= validated['hour_of_day'] <= 17 else 0
    validated['is_weekend'] = 1 if validated['day_of_week'] in [0, 6] else 0

    # Ensure questions_per_agent_message and questions_per_user_message
    validated['questions_per_agent_message'] = round(
        validated['question_count_agent'] / validated['agent_messages_count'], 4
    ) if validated['agent_messages_count'] > 0 else 0.0
    validated['questions_per_user_message'] = round(
        validated['question_count_user'] / validated['user_messages_count'], 4
    ) if validated['user_messages_count'] > 0 else 0.0

    # Ensure messages_per_minute
    validated['messages_per_minute'] = round(
        validated['total_messages'] / validated['conversation_duration_minutes'], 4
    ) if validated['conversation_duration_minutes'] > 0 else 0.0

    # Validate message length relationships - handle both naming conventions
    min_user_key = 'min_message_length_user' if 'min_message_length_user' in validated else 'minMessageLengthUser'
    max_user_key = 'max_message_length_user' if 'max_message_length_user' in validated else 'maxMessageLengthUser'
    min_agent_key = 'min_message_length_agent' if 'min_message_length_agent' in validated else 'minMessageLengthAgent'
    max_agent_key = 'max_message_length_agent' if 'max_message_length_agent' in validated else 'maxMessageLengthAgent'
    
    if validated[min_user_key] > validated[avg_user_length_key] or \
       validated[avg_user_length_key] > validated[max_user_key]:
        validated[min_user_key] = min(validated[min_user_key], validated[avg_user_length_key])
        validated[max_user_key] = max(validated[max_user_key], validated[avg_user_length_key])
    if validated[min_agent_key] > validated[avg_agent_length_key] or \
       validated[avg_agent_length_key] > validated[max_agent_key]:
        validated[min_agent_key] = min(validated[min_agent_key], validated[avg_agent_length_key])
        validated[max_agent_key] = max(validated[max_agent_key], validated[avg_agent_length_key])

    # Ensure session_id is a string UUID
    if 'session_id' not in validated or not validated['session_id']:
        validated['session_id'] = str(uuid.uuid4())

    return validated

def log_metrics(response: ConversionSynapse, reward: float, ground_truth: Dict):
    """
    Log detailed metrics for a miner's response for debugging and monitoring.

    Args:
        response (ConversionSynapse): Miner's response
        reward (float): Computed reward
        ground_truth (Dict): Ground truth data
    """
    predicted = response.prediction or {}
    bt.logging.debug(
        f"Miner UID: {response.miner_uid}, "
        f"Predicted: {predicted}, "
        f"True: {ground_truth}, "
        f"Confidence: {response.confidence:.3f}, "
        f"Response Time: {response.response_time:.2f}s, "
        f"Reward: {reward:.4f}"
    )

def preprocess_prediction(prediction: Dict) -> Dict:
    """
    Preprocess prediction data to ensure correct data types.
    NO FALLBACKS - raises errors if validation fails.
    
    Args:
        prediction (Dict): Dictionary with miner predictions
        
    Returns:
        Dict: Dictionary with validated data types
        
    Raises:
        ValueError: If prediction validation fails
        AssertionError: If prediction format is wrong
    """
    if not prediction:
        raise ValueError("Prediction cannot be empty - miners must provide valid predictions")
        
    assert isinstance(prediction, dict), f"Prediction must be dict, got {type(prediction)}"
    
    result = prediction.copy()
    
    # Check required fields
    if 'conversion_happened' not in result:
        raise ValueError("Prediction must contain 'conversion_happened' field")
    if 'time_to_conversion_seconds' not in result:
        raise ValueError("Prediction must contain 'time_to_conversion_seconds' field")
    
    # Validate conversion_happened - NO FALLBACKS
    if result['conversion_happened'] is None:
        raise ValueError("conversion_happened cannot be None")
    
    try:
        conversion_happened = int(result['conversion_happened'])
        if conversion_happened not in [0, 1]:
            raise ValueError(f"conversion_happened must be 0 or 1, got {conversion_happened}")
        result['conversion_happened'] = conversion_happened
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid conversion_happened value '{result['conversion_happened']}': {e}") from e
    
    # Validate time_to_conversion_seconds - NO FALLBACKS
    if result['time_to_conversion_seconds'] is None:
        raise ValueError("time_to_conversion_seconds cannot be None")
    
    try:
        time_to_conversion = float(result['time_to_conversion_seconds'])
        # Allow -1.0 or positive values only
        if time_to_conversion != -1.0 and time_to_conversion <= 0:
            raise ValueError(f"time_to_conversion_seconds must be -1.0 or positive, got {time_to_conversion}")
        result['time_to_conversion_seconds'] = time_to_conversion
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid time_to_conversion_seconds value '{result['time_to_conversion_seconds']}': {e}") from e
    
    # Validate logical consistency
    if result['conversion_happened'] == 1 and result['time_to_conversion_seconds'] == -1.0:
        raise ValueError("If conversion_happened=1, time_to_conversion_seconds must be positive, not -1.0")
    if result['conversion_happened'] == 0 and result['time_to_conversion_seconds'] != -1.0:
        raise ValueError("If conversion_happened=0, time_to_conversion_seconds must be -1.0")
            
    return result

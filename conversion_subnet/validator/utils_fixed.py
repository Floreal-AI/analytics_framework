import uuid
from typing import Dict

def validate_features(features: Dict) -> Dict:
    """
    Validate and correct conversation features to ensure logical consistency.

    Args:
        features (Dict): Dictionary of conversation features

    Returns:
        Dict: Validated and corrected features
    """
    validated = features.copy()

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
        'total_chars_from_user': 'totalCharsFromUser',
        'max_message_length_agent': 'maxMessageLengthAgent',
        'min_message_length_agent': 'minMessageLengthAgent',
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
            except (ValueError, TypeError):
                # Provide a reasonable default
                validated[field_key] = 0

    # Handle conversation duration fields
    duration_seconds_key = 'conversation_duration_seconds' if 'conversation_duration_seconds' in validated else 'conversationDurationSeconds'
    if duration_seconds_key in validated:
        validated['conversation_duration_minutes'] = round(validated[duration_seconds_key] / 60, 4)
        # Also ensure we have the standard key name
        validated['conversation_duration_seconds'] = validated[duration_seconds_key]

    # Handle message ratio and counts
    message_ratio_key = 'message_ratio' if 'message_ratio' in validated else 'messageRatio'
    total_messages_key = 'total_messages' if 'total_messages' in validated else 'totalMessages'
    
    if message_ratio_key in validated and total_messages_key in validated:
        user_msgs = int(validated[total_messages_key] / (1 + 1/validated[message_ratio_key]))
        validated['user_messages_count'] = user_msgs
        validated['userMessagesCount'] = user_msgs
        validated['agent_messages_count'] = validated[total_messages_key] - user_msgs
        validated['agentMessagesCount'] = validated[total_messages_key] - user_msgs

    # Handle average message length fields
    avg_user_key = 'avg_message_length_user' if 'avg_message_length_user' in validated else 'avgMessageLengthUser'
    avg_agent_key = 'avg_message_length_agent' if 'avg_message_length_agent' in validated else 'avgMessageLengthAgent'
    
    if avg_user_key in validated and 'user_messages_count' in validated:
        validated['total_chars_from_user'] = int(validated['user_messages_count'] * validated[avg_user_key])
        validated['totalCharsFromUser'] = validated['total_chars_from_user']
    
    if avg_agent_key in validated and 'agent_messages_count' in validated:
        validated['total_chars_from_agent'] = int(validated['agent_messages_count'] * validated[avg_agent_key])
        validated['totalCharsFromAgent'] = validated['total_chars_from_agent']

    # Handle time-based fields
    hour_key = 'hour_of_day' if 'hour_of_day' in validated else 'hourOfDay'
    day_key = 'day_of_week' if 'day_of_week' in validated else 'dayOfWeek'
    
    if hour_key in validated:
        validated['is_business_hours'] = 1 if 9 <= validated[hour_key] <= 17 else 0
        validated['isBusinessHours'] = validated['is_business_hours']
    
    if day_key in validated:
        validated['is_weekend'] = 1 if validated[day_key] in [0, 6] else 0
        validated['isWeekend'] = validated['is_weekend']

    # Ensure session_id is a string UUID
    if 'session_id' not in validated or not validated['session_id']:
        validated['session_id'] = str(uuid.uuid4())

    return validated

def log_metrics(response, reward: float, ground_truth: Dict):
    """
    Log detailed metrics for a miner's response for debugging and monitoring.

    Args:
        response: Miner's response
        reward (float): Computed reward
        ground_truth (Dict): Ground truth data
    """
    predicted = getattr(response, 'prediction', {}) or {}
    print(
        f"Miner UID: {getattr(response, 'miner_uid', 'unknown')}, "
        f"Predicted: {predicted}, "
        f"True: {ground_truth}, "
        f"Confidence: {getattr(response, 'confidence', 0.0):.3f}, "
        f"Response Time: {getattr(response, 'response_time', 0.0):.2f}s, "
        f"Reward: {reward:.4f}"
    )

def preprocess_prediction(prediction: Dict) -> Dict:
    """
    Preprocess prediction data to ensure correct data types.
    
    Args:
        prediction (Dict): Dictionary with miner predictions
        
    Returns:
        Dict: Dictionary with corrected data types
    """
    if not prediction:
        return {}
        
    result = prediction.copy()
    
    # Ensure conversion_happened is an integer
    if 'conversion_happened' in result and result['conversion_happened'] is not None:
        try:
            result['conversion_happened'] = int(result['conversion_happened'])
        except (ValueError, TypeError):
            result['conversion_happened'] = 0
    elif 'conversion_happened' in result and result['conversion_happened'] is None:
        result['conversion_happened'] = 0
            
    # Ensure time_to_conversion_seconds is a float
    if 'time_to_conversion_seconds' in result and result['time_to_conversion_seconds'] is not None:
        try:
            result['time_to_conversion_seconds'] = float(result['time_to_conversion_seconds'])
        except (ValueError, TypeError):
            result['time_to_conversion_seconds'] = -1.0
    elif 'time_to_conversion_seconds' in result and result['time_to_conversion_seconds'] is None:
        result['time_to_conversion_seconds'] = -1.0
            
    return result 
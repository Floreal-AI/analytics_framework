
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

    # Ensure conversation_duration_minutes is consistent
    validated['conversation_duration_minutes'] = round(validated['conversation_duration_seconds'] / 60, 4)

    # Ensure total_messages = user_messages_count + agent_messages_count
    user_msgs = int(validated['total_messages'] / (1 + 1/validated['message_ratio']))
    validated['user_messages_count'] = user_msgs
    validated['agent_messages_count'] = validated['total_messages'] - user_msgs

    # Ensure total_chars_from_user and total_chars_from_agent
    validated['total_chars_from_user'] = int(user_msgs * validated['avg_message_length_user'])
    validated['total_chars_from_agent'] = int(validated['agent_messages_count'] * validated['avg_message_length_agent'])

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

    # Validate message length relationships
    if validated['min_message_length_user'] > validated['avg_message_length_user'] or \
       validated['avg_message_length_user'] > validated['max_message_length_user']:
        validated['min_message_length_user'] = min(validated['min_message_length_user'], validated['avg_message_length_user'])
        validated['max_message_length_user'] = max(validated['max_message_length_user'], validated['avg_message_length_user'])
    if validated['min_message_length_agent'] > validated['avg_message_length_agent'] or \
       validated['avg_message_length_agent'] > validated['max_message_length_agent']:
        validated['min_message_length_agent'] = min(validated['min_message_length_agent'], validated['avg_message_length_agent'])
        validated['max_message_length_agent'] = max(validated['max_message_length_agent'], validated['avg_message_length_agent'])

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

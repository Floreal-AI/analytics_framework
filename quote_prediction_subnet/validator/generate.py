
import uuid
import random
import numpy as np
from faker import Faker
from typing import Dict

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
Faker.seed(seed)
fake = Faker()

def generate_conversation() -> Dict:
    """
    Generate a single synthetic conversation entry with 40 features for the AI Agent Task Performance Subnet.
    Features match the dataset format for training or testing, excluding target variables.
    Designed for validators to create one row at a time, simulating real-time conversation data.

    Returns:
        Dict: Conversation data with 40 features (no targets).
    """
    # Feature distributions based on example dataset
    features = {
        'conversation_duration_seconds': np.clip(np.random.normal(100.0, 20.0), 60.0, 150.0),
        'hour_of_day': random.randint(8, 19),
        'day_of_week': random.randint(0, 6),
        'time_to_first_response_seconds': np.clip(np.random.normal(10.0, 3.0), 5.0, 20.0),
        'avg_response_time_seconds': np.clip(np.random.normal(15.0, 2.0), 10.0, 20.0),
        'max_response_time_seconds': np.clip(np.random.normal(18.0, 2.5), 15.0, 25.0),
        'min_response_time_seconds': np.clip(np.random.normal(10.0, 2.0), 5.0, 15.0),
        'avg_agent_response_time_seconds': np.clip(np.random.normal(15.0, 2.0), 10.0, 20.0),
        'avg_user_response_time_seconds': np.clip(np.random.normal(15.0, 2.0), 10.0, 20.0),
        'response_time_stddev': np.clip(np.random.normal(3.0, 1.0), 0.0, 5.0),
        'response_gap_max': np.clip(np.random.normal(18.0, 2.5), 15.0, 25.0),
        'total_messages': int(np.clip(np.random.normal(8.0, 2.0), 4, 20)),
        'message_ratio': np.clip(np.random.normal(1.3, 0.3), 1.0, 2.0),
        'avg_message_length_user': np.clip(np.random.normal(40.0, 5.0), 20.0, 60.0),
        'max_message_length_user': np.clip(np.random.normal(50.0, 5.0), 30.0, 70.0),
        'min_message_length_user': np.clip(np.random.normal(30.0, 5.0), 15.0, 45.0),
        'avg_message_length_agent': np.clip(np.random.normal(75.0, 10.0), 50.0, 100.0),
        'max_message_length_agent': np.clip(np.random.normal(100.0, 15.0), 70.0, 130.0),
        'min_message_length_agent': np.clip(np.random.normal(50.0, 10.0), 30.0, 80.0),
        'question_count_agent': int(np.clip(np.random.normal(3.0, 1.0), 1, 5)),
        'question_count_user': int(np.clip(np.random.normal(1.0, 0.5), 0, 3)),
        'sequential_user_messages': int(np.clip(np.random.normal(2.0, 1.0), 1, 4)),
        'sequential_agent_messages': 1,  # Typically 1
        'entities_collected_count': int(np.clip(np.random.normal(4.0, 1.5), 1, 6)),
        'has_target_entity': random.choice([0, 1]),
        'avg_entity_confidence': np.clip(np.random.normal(0.9, 0.02), 0.8, 0.95),
        'min_entity_confidence': np.clip(np.random.normal(0.85, 0.03), 0.8, 0.9),
        'entity_collection_rate': 0,  # Typically 0
        'repeated_questions': random.choice([0, 1]),
        'message_alternation_rate': np.clip(np.random.normal(0.9, 0.1), 0.7, 1.0)
    }

    # Ensure logical consistency
    features['conversation_duration_minutes'] = features['conversation_duration_seconds'] / 60
    user_msgs = int(features['total_messages'] / (1 + 1/features['message_ratio']))
    features['user_messages_count'] = user_msgs
    features['agent_messages_count'] = features['total_messages'] - user_msgs
    features['total_chars_from_user'] = int(user_msgs * features['avg_message_length_user'])
    features['total_chars_from_agent'] = int(features['agent_messages_count'] * features['avg_message_length_agent'])
    features['is_business_hours'] = 1 if 9 <= features['hour_of_day'] <= 17 else 0
    features['is_weekend'] = 1 if features['day_of_week'] in [0, 6] else 0
    features['questions_per_agent_message'] = features['question_count_agent'] / features['agent_messages_count'] if features['agent_messages_count'] > 0 else 0
    features['questions_per_user_message'] = features['question_count_user'] / features['user_messages_count'] if features['user_messages_count'] > 0 else 0
    features['messages_per_minute'] = features['total_messages'] / features['conversation_duration_minutes'] if features['conversation_duration_minutes'] > 0 else 0

    # Combine features with session_id
    conversation = {
        'session_id': str(uuid.uuid4()),
        **features
    }

    return conversation

if __name__ == "__main__":
    # Generate and print a single conversation entry for testing
    single_conversation = generate_conversation()
    print("Single Conversation Example:", single_conversation)


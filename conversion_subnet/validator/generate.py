import uuid
import random
import numpy as np
import pandas as pd
from faker import Faker
from typing import Dict
from conversion_subnet.validator.utils import validate_features

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
Faker.seed(seed)
fake = Faker()

def generate_conversation() -> Dict:
    """
    Generate a synthetic conversation with realistic features.
    
    This is used to simulate real-time conversations for testing and development.
    All 40 features are generated with realistic distributions.
    
    Returns:
        Dict: Dictionary of 40 conversation features
    """
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    
    # Feature distributions based on example dataset (15 conversations)
    features = {
        'session_id': session_id,
        'conversation_duration_seconds': round(float(np.clip(np.random.normal(100.0, 20.0), 60.0, 150.0)), 2),
        'hour_of_day': int(random.randint(8, 19)),  # Working hours: 8 AM to 7 PM
        'day_of_week': int(random.randint(0, 6)),   # 0 (Sunday) to 6 (Saturday)
        'time_to_first_response_seconds': round(float(np.clip(np.random.normal(10.0, 3.0), 5.0, 20.0)), 2),
        'avg_response_time_seconds': round(float(np.clip(np.random.normal(15.0, 2.0), 10.0, 20.0)), 2),
        'max_response_time_seconds': round(float(np.clip(np.random.normal(18.0, 2.5), 15.0, 25.0)), 2),
        'min_response_time_seconds': round(float(np.clip(np.random.normal(10.0, 2.0), 5.0, 15.0)), 2),
        'avg_agent_response_time_seconds': round(float(np.clip(np.random.normal(15.0, 2.0), 10.0, 20.0)), 2),
        'avg_user_response_time_seconds': round(float(np.clip(np.random.normal(15.0, 2.0), 10.0, 20.0)), 2),
        'response_time_stddev': round(float(np.clip(np.random.normal(3.0, 1.0), 0.0, 5.0)), 2),
        'response_gap_max': round(float(np.clip(np.random.normal(18.0, 2.5), 15.0, 25.0)), 2),
        'total_messages': int(np.clip(np.random.normal(8.0, 2.0), 4, 20)),
        'message_ratio': round(float(np.clip(np.random.normal(1.3, 0.3), 1.0, 2.0)), 2),
        'avg_message_length_user': round(float(np.clip(np.random.normal(40.0, 5.0), 20.0, 60.0)), 2),
        'max_message_length_user': int(np.clip(np.random.normal(50.0, 5.0), 30.0, 70.0)),
        'min_message_length_user': int(np.clip(np.random.normal(30.0, 5.0), 15.0, 45.0)),
        'avg_message_length_agent': round(float(np.clip(np.random.normal(75.0, 10.0), 50.0, 100.0)), 2),
        'max_message_length_agent': int(np.clip(np.random.normal(100.0, 15.0), 70.0, 130.0)),
        'min_message_length_agent': int(np.clip(np.random.normal(50.0, 10.0), 30.0, 80.0)),
        'question_count_agent': int(np.clip(np.random.normal(3.0, 1.0), 1, 5)),
        'question_count_user': int(np.clip(np.random.normal(1.0, 0.5), 0, 3)),
        'sequential_user_messages': int(np.clip(np.random.normal(2.0, 1.0), 1, 4)),
        'sequential_agent_messages': int(1),  # Typically 1, as agent responses are non-sequential
        'entities_collected_count': int(np.clip(np.random.normal(4.0, 1.5), 1, 6)),
        'has_target_entity': int(random.choice([0, 1])),
        'avg_entity_confidence': round(float(np.clip(np.random.normal(0.9, 0.02), 0.8, 0.95)), 3),
        'min_entity_confidence': round(float(np.clip(np.random.normal(0.85, 0.03), 0.8, 0.9)), 3),
        'entity_collection_rate': 0.0,  # Typically 0 in example dataset
        'repeated_questions': int(random.choice([0, 1])),
        'message_alternation_rate': round(float(np.clip(np.random.normal(0.9, 0.1), 0.7, 1.0)), 3)
    }

    # Preprocess features to ensure correct integer types
    features = preprocess_features(features)

    # Validate features using utils to ensure logical consistency
    validated_features = validate_features(features)

    return validated_features

def preprocess_features(features: Dict) -> Dict:
    """
    Preprocess features to ensure correct data types, especially for integer fields.
    
    Args:
        features (Dict): Dictionary of conversation features
        
    Returns:
        Dict: Dictionary with corrected data types
    """
    # List of integer fields that must be integers, not floats
    integer_fields = [
        'hour_of_day', 'day_of_week', 'is_business_hours', 'is_weekend',
        'total_messages', 'user_messages_count', 'agent_messages_count',
        'max_message_length_user', 'min_message_length_user', 'total_chars_from_user',
        'max_message_length_agent', 'min_message_length_agent', 'total_chars_from_agent',
        'question_count_agent', 'question_count_user', 'sequential_user_messages',
        'sequential_agent_messages', 'entities_collected_count', 'has_target_entity',
        'repeated_questions'
    ]
    
    # Convert any float values to integers for integer fields
    for field in integer_fields:
        if field in features and features[field] is not None:
            try:
                features[field] = int(features[field])
            except (ValueError, TypeError):
                # If conversion fails, use a default value
                features[field] = 0
    
    return features

def generate_conversations(num_conversations: int, path: str) -> None:
    """
    Generate multiple synthetic conversation entries and save to a CSV file for training purposes.

    Args:
        num_conversations (int): Number of conversations to generate
        path (str): Path to save the CSV file

    Returns:
        None
    """
    if num_conversations <= 0:
        raise ValueError("Number of conversations must be positive")
    conversations = [generate_conversation() for _ in range(num_conversations)]
    df = pd.DataFrame(conversations)
    df.to_csv(path, index=False)
    print(f"Saved {num_conversations} synthetic conversations to {path}")

if __name__ == "__main__":
    # Generate and print a single conversation entry for testing
    single_conversation = generate_conversation()
    print("Single Conversation Example:", single_conversation)


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
    Generate a single synthetic conversation entry with 40 features for the AI Agent Task Performance Subnet.
    Features match the dataset format for training or testing, excluding target variables (conversion_happened,
    time_to_conversion_seconds, time_to_conversion_minutes). Designed for validators to create one row at a time,
    simulating real-time conversation data.

    Features include:
    - session_id: Unique UUID
    - Conversation metrics: duration, messages per minute, total messages, user/agent message counts, message ratio
    - Timing metrics: response times (first, average, max, min, agent, user), standard deviation, max gap
    - Message lengths: average, max, min for user and agent, total characters
    - Question metrics: count and rate for agent and user, repeated questions
    - Entity metrics: count, target entity presence, confidence (average, min), collection rate
    - Interaction patterns: sequential messages, alternation rate
    - Context: hour of day, day of week, business hours, weekend

    Returns:
        Dict: Conversation data with 40 features (no targets).
    """
    # Feature distributions based on example dataset (15 conversations)
    features = {
        'conversation_duration_seconds': round(float(np.clip(np.random.normal(100.0, 20.0), 60.0, 150.0)), 2),
        'hour_of_day': random.randint(8, 19),  # Working hours: 8 AM to 7 PM
        'day_of_week': random.randint(0, 6),   # 0 (Sunday) to 6 (Saturday)
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
        'max_message_length_user': round(float(np.clip(np.random.normal(50.0, 5.0), 30.0, 70.0)), 2),
        'min_message_length_user': round(float(np.clip(np.random.normal(30.0, 5.0), 15.0, 45.0)), 2),
        'avg_message_length_agent': round(float(np.clip(np.random.normal(75.0, 10.0), 50.0, 100.0)), 2),
        'max_message_length_agent': round(float(np.clip(np.random.normal(100.0, 15.0), 70.0, 130.0)), 2),
        'min_message_length_agent': round(float(np.clip(np.random.normal(50.0, 10.0), 30.0, 80.0)), 2),
        'question_count_agent': int(np.clip(np.random.normal(3.0, 1.0), 1, 5)),
        'question_count_user': int(np.clip(np.random.normal(1.0, 0.5), 0, 3)),
        'sequential_user_messages': int(np.clip(np.random.normal(2.0, 1.0), 1, 4)),
        'sequential_agent_messages': 1,  # Typically 1, as agent responses are non-sequential
        'entities_collected_count': int(np.clip(np.random.normal(4.0, 1.5), 1, 6)),
        'has_target_entity': random.choice([0, 1]),
        'avg_entity_confidence': round(float(np.clip(np.random.normal(0.9, 0.02), 0.8, 0.95)), 3),
        'min_entity_confidence': round(float(np.clip(np.random.normal(0.85, 0.03), 0.8, 0.9)), 3),
        'entity_collection_rate': 0,  # Typically 0 in example dataset
        'repeated_questions': random.choice([0, 1]),
        'message_alternation_rate': round(float(np.clip(np.random.normal(0.9, 0.1), 0.7, 1.0)), 3)
    }

    # Validate features using utils to ensure logical consistency
    validated_features = validate_features(features)

    # Combine features with session_id
    conversation = {
        'session_id': str(uuid.uuid4()),
        **validated_features
    }

    return conversation

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

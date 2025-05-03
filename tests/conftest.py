"""
Pytest configuration file with fixtures for testing the conversion_subnet package.
"""

import os
import pytest
import torch
import numpy as np
from pathlib import Path

# Import our custom pytest plugins
pytest_plugins = ["tests.pytest_plugins"]

from conversion_subnet.protocol import ConversionSynapse, ConversationFeatures, PredictionOutput
from conversion_subnet.utils.configuration import ConversionSubnetConfig, MinerConfig
from conversion_subnet.miner.miner import BinaryClassificationMiner
from conversion_subnet.utils.log import logger
from tests.test_utils import ensure_test_results_dir, save_test_logs, save_test_results

# Constants for testing
TEST_DEVICE = "cpu"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment for the entire test session."""
    # Ensure test results directory exists
    test_results_dir = ensure_test_results_dir()
    
    # Log the test session start
    logger.info("Starting test session", test_dir=str(test_results_dir))
    
    yield
    
    # Log the test session end
    logger.info("Test session completed")


@pytest.fixture
def sample_features() -> ConversationFeatures:
    """Fixture providing sample conversation features for testing."""
    return {
        'session_id': 'test-session-123',
        'conversation_duration_seconds': 120.5,
        'conversation_duration_minutes': 2.0,
        'hour_of_day': 14,
        'day_of_week': 2,
        'is_business_hours': 1,
        'is_weekend': 0,
        'time_to_first_response_seconds': 3.5,
        'avg_response_time_seconds': 8.2,
        'max_response_time_seconds': 15.0,
        'min_response_time_seconds': 2.1,
        'avg_agent_response_time_seconds': 7.5,
        'avg_user_response_time_seconds': 9.0,
        'response_time_stddev': 3.2,
        'response_gap_max': 15.0,
        'messages_per_minute': 6.3,
        'total_messages': 12,
        'user_messages_count': 5,
        'agent_messages_count': 7,
        'message_ratio': 1.4,
        'avg_message_length_user': 42.3,
        'max_message_length_user': 78,
        'min_message_length_user': 12,
        'total_chars_from_user': 211,
        'avg_message_length_agent': 55.6,
        'max_message_length_agent': 102,
        'min_message_length_agent': 18,
        'total_chars_from_agent': 389,
        'question_count_agent': 4,
        'questions_per_agent_message': 0.57,
        'question_count_user': 2,
        'questions_per_user_message': 0.4,
        'sequential_user_messages': 0,
        'sequential_agent_messages': 1,
        'entities_collected_count': 5,
        'has_target_entity': 1,
        'avg_entity_confidence': 0.87,
        'min_entity_confidence': 0.65,
        'entity_collection_rate': 1.25,
        'repeated_questions': 1,
        'message_alternation_rate': 0.91
    }


@pytest.fixture
def sample_prediction() -> PredictionOutput:
    """Fixture providing a sample prediction output for testing."""
    return {
        'conversion_happened': 1,
        'time_to_conversion_seconds': 85.5
    }


@pytest.fixture
def sample_synapse(sample_features) -> ConversionSynapse:
    """Fixture providing a sample synapse for testing."""
    return ConversionSynapse(features=sample_features)


@pytest.fixture
def test_config() -> ConversionSubnetConfig:
    """Fixture providing a test configuration."""
    config = ConversionSubnetConfig()
    config.miner.device = TEST_DEVICE
    return config


@pytest.fixture
def mock_model() -> torch.nn.Module:
    """Fixture providing a simple mock model for testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(40, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
        torch.nn.Sigmoid()
    )
    return model


@pytest.fixture
def mock_miner(test_config, mock_model) -> BinaryClassificationMiner:
    """Fixture providing a miner with a mock model for testing."""
    miner = BinaryClassificationMiner(test_config)
    # Replace the model with our mock model
    miner.model = mock_model
    return miner


@pytest.fixture
def sample_dataset():
    """Fixture providing a small sample dataset for testing."""
    # Create a simple numpy dataset with 10 samples, 40 features each
    X = np.random.rand(10, 40)
    y = np.random.randint(0, 2, 10)
    return X, y


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Create a temporary directory for model checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir 
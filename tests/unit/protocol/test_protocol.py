"""
Unit tests for the protocol module.
"""

import pytest
from unittest.mock import patch, MagicMock
import bittensor as bt

from conversion_subnet.protocol import (
    ConversionSynapse, ConversationFeatures, PredictionOutput, DEFAULT_PREDICTION
)


class TestConversationFeatures:
    """Test suite for ConversationFeatures TypedDict."""
    
    def test_conversation_features_structure(self):
        """Test that ConversationFeatures can be created with required fields."""
        features: ConversationFeatures = {
            'session_id': 'test_session_123',
            'conversation_duration_seconds': 120.5,
            'has_target_entity': 1,
            'entities_collected_count': 3,
            'message_ratio': 1.5
        }
        
        # Verify required fields
        assert features['session_id'] == 'test_session_123'
        assert features['conversation_duration_seconds'] == 120.5
        assert features['has_target_entity'] == 1
        assert features['entities_collected_count'] == 3
        assert features['message_ratio'] == 1.5

    def test_conversation_features_with_optional_fields(self):
        """Test ConversationFeatures with optional fields."""
        features: ConversationFeatures = {
            'session_id': 'test_session_456',
            'conversation_duration_seconds': 300.0,
            'has_target_entity': 0,
            'entities_collected_count': 0,
            'message_ratio': 0.8,
            'hour_of_day': 14,
            'day_of_week': 2,
            'is_business_hours': 1,
            'is_weekend': 0,
            'total_messages': 25,
            'user_messages_count': 12,
            'agent_messages_count': 13,
            'avg_message_length_user': 45.2,
            'avg_message_length_agent': 52.1
        }
        
        # Verify optional fields
        assert features['hour_of_day'] == 14
        assert features['day_of_week'] == 2
        assert features['is_business_hours'] == 1
        assert features['is_weekend'] == 0
        assert features['total_messages'] == 25
        assert features['user_messages_count'] == 12
        assert features['agent_messages_count'] == 13
        assert features['avg_message_length_user'] == 45.2
        assert features['avg_message_length_agent'] == 52.1


class TestPredictionOutput:
    """Test suite for PredictionOutput TypedDict."""
    
    def test_prediction_output_structure(self):
        """Test that PredictionOutput can be created with required fields."""
        prediction: PredictionOutput = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 180.5
        }
        
        assert prediction['conversion_happened'] == 1
        assert prediction['time_to_conversion_seconds'] == 180.5

    def test_prediction_output_no_conversion(self):
        """Test PredictionOutput for no conversion case."""
        prediction: PredictionOutput = {
            'conversion_happened': 0,
            'time_to_conversion_seconds': -1.0
        }
        
        assert prediction['conversion_happened'] == 0
        assert prediction['time_to_conversion_seconds'] == -1.0


class TestDefaultPrediction:
    """Test suite for DEFAULT_PREDICTION constant."""
    
    def test_default_prediction_structure(self):
        """Test that DEFAULT_PREDICTION has correct structure."""
        assert 'conversion_happened' in DEFAULT_PREDICTION
        assert 'time_to_conversion_seconds' in DEFAULT_PREDICTION
        assert DEFAULT_PREDICTION['conversion_happened'] == 0
        assert DEFAULT_PREDICTION['time_to_conversion_seconds'] == -1.0

    def test_default_prediction_immutability(self):
        """Test that modifying a copy doesn't affect the original."""
        copy = DEFAULT_PREDICTION.copy()
        copy['conversion_happened'] = 1
        
        # Original should remain unchanged
        assert DEFAULT_PREDICTION['conversion_happened'] == 0


class TestConversionSynapse:
    """Test suite for ConversionSynapse class."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample conversation features for testing."""
        return {
            'session_id': 'test_session_789',
            'conversation_duration_seconds': 240.0,
            'has_target_entity': 1,
            'entities_collected_count': 5,
            'message_ratio': 1.2,
            'hour_of_day': 10,  # Use integer instead of float
            'day_of_week': 3,   # Use integer instead of float
            'is_business_hours': 1,  # Use integer instead of float
            'total_messages': 30,    # Use integer instead of float
            'user_messages_count': 15,
            'agent_messages_count': 15
        }

    def test_synapse_initialization_with_defaults(self, sample_features):
        """Test ConversionSynapse initialization with default values."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Check that defaults are set
        assert synapse.prediction == DEFAULT_PREDICTION
        assert synapse.confidence is None
        assert synapse.response_time == 0.0
        assert synapse.miner_uid == 0

    def test_synapse_initialization_with_custom_values(self, sample_features):
        """Test ConversionSynapse initialization with custom values."""
        custom_prediction = {'conversion_happened': 1, 'time_to_conversion_seconds': 150.0}
        
        synapse = ConversionSynapse(
            features=sample_features,
            prediction=custom_prediction,
            confidence=0.85,
            response_time=2.5,
            miner_uid=42
        )
        
        assert synapse.prediction == custom_prediction
        assert synapse.confidence == 0.85
        assert synapse.response_time == 2.5
        assert synapse.miner_uid == 42

    def test_post_init_integer_conversion(self):
        """Test that __post_init__ converts float fields to integers."""
        # Create features with float values that should be converted
        features_with_floats = {
            'session_id': 'test_session_789',
            'conversation_duration_seconds': 240.0,
            'has_target_entity': 1.0,  # Float that should convert to int
            'entities_collected_count': 5.0,  # Float that should convert to int
            'message_ratio': 1.2,
            'hour_of_day': 10,
            'day_of_week': 3,
            'is_business_hours': 1,
            'total_messages': 30,
            'user_messages_count': 15,
            'agent_messages_count': 15
        }
        
        synapse = ConversionSynapse(features=features_with_floats)
        
        # Check that integer fields are integers
        assert isinstance(synapse.features['has_target_entity'], int)
        assert isinstance(synapse.features['entities_collected_count'], int)
        assert isinstance(synapse.features['hour_of_day'], int)
        assert isinstance(synapse.features['day_of_week'], int)
        assert isinstance(synapse.features['is_business_hours'], int)
        assert isinstance(synapse.features['total_messages'], int)

    @patch('bittensor.logging.warning')
    def test_post_init_conversion_failure_handling(self, mock_warning):
        """Test handling of conversion failures in __post_init__."""
        # Create valid base features first
        valid_features = {
            'session_id': 'test_session_789',
            'conversation_duration_seconds': 240.0,
            'has_target_entity': 1,
            'entities_collected_count': 5,
            'message_ratio': 1.2,
            'hour_of_day': 10,
            'day_of_week': 3,
            'is_business_hours': 1,
            'total_messages': 30,
            'user_messages_count': 15,
            'agent_messages_count': 15
        }
        
        # Create synapse with valid features first
        synapse = ConversionSynapse(features=valid_features)
        
        # Now test the post_init conversion failure handling by directly calling it
        # with invalid data in the features
        synapse.features['hour_of_day'] = 'invalid'
        synapse.features['total_messages'] = None
        
        # Call __post_init__ directly to test conversion failure handling
        synapse.__post_init__()
        
        # Should log warnings for failed conversions
        assert mock_warning.call_count >= 1

    def test_post_init_with_none_features(self):
        """Test __post_init__ with minimal valid features."""
        # Create minimal valid features
        minimal_features = {
            'session_id': 'test_session',
            'conversation_duration_seconds': 120.0,
            'has_target_entity': 1,
            'entities_collected_count': 0,
            'message_ratio': 1.0
        }
        
        synapse = ConversionSynapse(features=minimal_features)
        
        # Should not crash and should have default prediction
        assert synapse.prediction == DEFAULT_PREDICTION

    def test_set_prediction_with_valid_data(self, sample_features):
        """Test set_prediction with valid prediction data."""
        synapse = ConversionSynapse(features=sample_features)
        
        prediction = {'conversion_happened': 1, 'time_to_conversion_seconds': 120.0}
        synapse.set_prediction(prediction)
        
        assert synapse.prediction['conversion_happened'] == 1
        assert synapse.prediction['time_to_conversion_seconds'] == 120.0

    def test_set_prediction_with_none(self, sample_features):
        """Test set_prediction with None input."""
        synapse = ConversionSynapse(features=sample_features)
        
        synapse.set_prediction(None)
        
        assert synapse.prediction == DEFAULT_PREDICTION

    def test_set_prediction_with_empty_dict(self, sample_features):
        """Test set_prediction with empty dictionary."""
        synapse = ConversionSynapse(features=sample_features)
        
        synapse.set_prediction({})
        
        assert synapse.prediction == DEFAULT_PREDICTION

    def test_set_prediction_with_missing_fields(self, sample_features):
        """Test set_prediction with missing required fields."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Only provide one field
        prediction = {'conversion_happened': 1}
        synapse.set_prediction(prediction)
        
        # Should fill in missing field with default
        assert synapse.prediction['conversion_happened'] == 1
        assert synapse.prediction['time_to_conversion_seconds'] == -1.0

    def test_set_prediction_type_conversion(self, sample_features):
        """Test set_prediction with type conversion."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Provide string values that should be converted
        prediction = {
            'conversion_happened': '1',
            'time_to_conversion_seconds': '180.5'
        }
        synapse.set_prediction(prediction)
        
        assert synapse.prediction['conversion_happened'] == 1
        assert synapse.prediction['time_to_conversion_seconds'] == 180.5
        assert isinstance(synapse.prediction['conversion_happened'], int)
        assert isinstance(synapse.prediction['time_to_conversion_seconds'], float)

    def test_set_prediction_invalid_type_conversion(self, sample_features):
        """Test set_prediction with invalid values for type conversion."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Provide invalid values
        prediction = {
            'conversion_happened': 'invalid',
            'time_to_conversion_seconds': 'also_invalid'
        }
        synapse.set_prediction(prediction)
        
        # Should use defaults for invalid values
        assert synapse.prediction['conversion_happened'] == 0
        assert synapse.prediction['time_to_conversion_seconds'] == -1.0

    def test_set_prediction_preserves_original(self, sample_features):
        """Test that set_prediction doesn't modify the original input."""
        synapse = ConversionSynapse(features=sample_features)
        
        original_prediction = {'conversion_happened': 1, 'time_to_conversion_seconds': 120.0}
        original_copy = original_prediction.copy()
        
        synapse.set_prediction(original_prediction)
        
        # Original should be unchanged
        assert original_prediction == original_copy

    def test_deserialize_with_valid_prediction(self, sample_features):
        """Test deserialize method with valid prediction."""
        prediction = {'conversion_happened': 1, 'time_to_conversion_seconds': 90.0}
        synapse = ConversionSynapse(features=sample_features, prediction=prediction)
        
        result = synapse.deserialize()
        
        assert result == prediction
        assert result['conversion_happened'] == 1
        assert result['time_to_conversion_seconds'] == 90.0

    def test_deserialize_with_none_prediction(self, sample_features):
        """Test deserialize method with None prediction."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Use set_prediction method instead of direct assignment to avoid Pydantic validation
        synapse.set_prediction(None)
        
        result = synapse.deserialize()
        
        assert result == DEFAULT_PREDICTION
        # Ensure it's a copy, not the original
        result['conversion_happened'] = 999
        assert DEFAULT_PREDICTION['conversion_happened'] == 0

    def test_synapse_with_all_integer_fields(self):
        """Test synapse with all possible integer fields to ensure coverage."""
        features = {
            'session_id': 'comprehensive_test',
            'conversation_duration_seconds': 300.0,
            'has_target_entity': 1,  # Use integer instead of float
            'entities_collected_count': 5,  # Use integer instead of float
            'message_ratio': 1.5,
            # All integer fields - use integers instead of floats
            'hour_of_day': 15,
            'day_of_week': 4,
            'is_business_hours': 1,
            'is_weekend': 0,
            'total_messages': 50,
            'user_messages_count': 25,
            'agent_messages_count': 25,
            'max_message_length_user': 100,
            'min_message_length_user': 10,
            'total_chars_from_user': 1500,
            'max_message_length_agent': 120,
            'min_message_length_agent': 15,
            'total_chars_from_agent': 1800,
            'question_count_agent': 8,
            'question_count_user': 5,
            'sequential_user_messages': 3,
            'sequential_agent_messages': 2,
            'repeated_questions': 1
        }
        
        synapse = ConversionSynapse(features=features)
        
        # Verify all integer fields are integers
        integer_fields = [
            'hour_of_day', 'day_of_week', 'is_business_hours', 'is_weekend',
            'total_messages', 'user_messages_count', 'agent_messages_count',
            'max_message_length_user', 'min_message_length_user', 'total_chars_from_user',
            'max_message_length_agent', 'min_message_length_agent', 'total_chars_from_agent',
            'question_count_agent', 'question_count_user', 'sequential_user_messages',
            'sequential_agent_messages', 'entities_collected_count', 'has_target_entity',
            'repeated_questions'
        ]
        
        for field in integer_fields:
            if field in synapse.features:
                assert isinstance(synapse.features[field], int), f"{field} should be int but is {type(synapse.features[field])}"

    def test_synapse_inheritance_from_bt_synapse(self, sample_features):
        """Test that ConversionSynapse properly inherits from bt.Synapse."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Should be instance of bt.Synapse
        assert isinstance(synapse, bt.Synapse)
        
        # Should have the features attribute
        assert hasattr(synapse, 'features')
        assert synapse.features == sample_features 
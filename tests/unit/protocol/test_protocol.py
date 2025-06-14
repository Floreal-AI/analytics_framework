"""
Unit tests for the protocol module.
"""

import pytest
from unittest.mock import patch, MagicMock
import bittensor as bt

from conversion_subnet.protocol import (
    ConversionSynapse, ConversationFeatures, PredictionOutput
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

    @pytest.fixture
    def valid_prediction(self):
        """Create a valid prediction for testing."""
        return {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 60.0
        }

    def test_synapse_initialization_without_prediction(self, sample_features):
        """Test ConversionSynapse initialization without prediction (should be None)."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Check that prediction is None by default
        assert synapse.prediction is None
        assert synapse.confidence is None
        assert synapse.response_time == 0.0
        assert synapse.miner_uid == 0

    def test_synapse_initialization_with_valid_prediction(self, sample_features, valid_prediction):
        """Test ConversionSynapse initialization with valid prediction."""
        synapse = ConversionSynapse(
            features=sample_features,
            prediction=valid_prediction,
            confidence=0.85,
            response_time=2.5,
            miner_uid=42
        )
        
        assert synapse.prediction == valid_prediction
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

    def test_post_init_conversion_failure_raises_error(self):
        """Test that conversion failures in __post_init__ raise errors."""
        # Create features with invalid data that can't be converted
        features_with_invalid_data = {
            'session_id': 'test_session_789',
            'conversation_duration_seconds': 240.0,
            'has_target_entity': 1,
            'entities_collected_count': 5,
            'message_ratio': 1.2,
            'hour_of_day': "invalid_string",  # This should cause a ValidationError from Pydantic
            'day_of_week': 3,
            'is_business_hours': 1,
            'total_messages': 30,
            'user_messages_count': 15,
            'agent_messages_count': 15
        }
        
        # Should raise ValidationError from Pydantic since it validates before __post_init__
        from pydantic_core import ValidationError
        with pytest.raises(ValidationError):
            ConversionSynapse(features=features_with_invalid_data)

    def test_post_init_with_none_features(self):
        """Test __post_init__ with None features raises ValidationError."""
        # Pydantic will validate that features is a dict, so None will raise ValidationError
        from pydantic_core import ValidationError
        with pytest.raises(ValidationError):
            ConversionSynapse(features=None)

    def test_set_prediction_with_valid_data(self, sample_features):
        """Test set_prediction with valid prediction data."""
        synapse = ConversionSynapse(features=sample_features)
        
        valid_prediction = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 120.5
        }
        
        synapse.set_prediction(valid_prediction)
        
        assert synapse.prediction['conversion_happened'] == 1
        assert synapse.prediction['time_to_conversion_seconds'] == 120.5

    def test_set_prediction_with_none_raises_error(self, sample_features):
        """Test set_prediction with None raises ValueError."""
        synapse = ConversionSynapse(features=sample_features)
        
        with pytest.raises(ValueError, match="Prediction cannot be None"):
            synapse.set_prediction(None)

    def test_set_prediction_with_empty_dict_raises_error(self, sample_features):
        """Test set_prediction with empty dict raises ValueError."""
        synapse = ConversionSynapse(features=sample_features)
        
        with pytest.raises(ValueError, match="Prediction cannot be empty"):
            synapse.set_prediction({})

    def test_set_prediction_with_missing_fields_raises_error(self, sample_features):
        """Test set_prediction with missing required fields raises ValueError."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Missing conversion_happened
        with pytest.raises(ValueError, match="Prediction must contain 'conversion_happened' field"):
            synapse.set_prediction({'time_to_conversion_seconds': 120.5})
        
        # Missing time_to_conversion_seconds
        with pytest.raises(ValueError, match="Prediction must contain 'time_to_conversion_seconds' field"):
            synapse.set_prediction({'conversion_happened': 1})

    def test_set_prediction_type_conversion(self, sample_features):
        """Test set_prediction handles type conversion correctly."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Test with string values that can be converted
        prediction_with_strings = {
            'conversion_happened': '1',
            'time_to_conversion_seconds': '120.5'
        }
        
        synapse.set_prediction(prediction_with_strings)
        
        assert synapse.prediction['conversion_happened'] == 1
        assert isinstance(synapse.prediction['conversion_happened'], int)
        assert synapse.prediction['time_to_conversion_seconds'] == 120.5
        assert isinstance(synapse.prediction['time_to_conversion_seconds'], float)

    def test_set_prediction_invalid_type_conversion_raises_error(self, sample_features):
        """Test set_prediction with invalid type conversion raises ValueError."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Invalid conversion_happened value
        with pytest.raises(ValueError, match="Invalid conversion_happened value"):
            synapse.set_prediction({
                'conversion_happened': 'invalid_string',
                'time_to_conversion_seconds': 120.5
            })
        
        # Invalid time_to_conversion_seconds value
        with pytest.raises(ValueError, match="Invalid time_to_conversion_seconds value"):
            synapse.set_prediction({
                'conversion_happened': 1,
                'time_to_conversion_seconds': 'invalid_string'
            })

    def test_set_prediction_logical_validation(self, sample_features):
        """Test set_prediction validates logical consistency."""
        synapse = ConversionSynapse(features=sample_features)
        
        # Invalid: conversion_happened=1 but time_to_conversion_seconds=-1.0
        with pytest.raises(ValueError, match="If conversion_happened=1, time_to_conversion_seconds must be positive"):
            synapse.set_prediction({
                'conversion_happened': 1,
                'time_to_conversion_seconds': -1.0
            })
        
        # Invalid: conversion_happened=0 but time_to_conversion_seconds positive
        with pytest.raises(ValueError, match="If conversion_happened=0, time_to_conversion_seconds must be -1.0"):
            synapse.set_prediction({
                'conversion_happened': 0,
                'time_to_conversion_seconds': 60.0
            })

    def test_set_prediction_preserves_original(self, sample_features):
        """Test set_prediction doesn't modify original prediction dict."""
        synapse = ConversionSynapse(features=sample_features)
        
        original_prediction = {
            'conversion_happened': '1',
            'time_to_conversion_seconds': '120.5'
        }
        original_copy = original_prediction.copy()
        
        synapse.set_prediction(original_prediction)
        
        # Original should be unchanged
        assert original_prediction == original_copy

    def test_deserialize_with_valid_prediction(self, sample_features, valid_prediction):
        """Test deserialize with valid prediction."""
        synapse = ConversionSynapse(features=sample_features, prediction=valid_prediction)
        
        result = synapse.deserialize()
        
        assert result == valid_prediction

    def test_deserialize_with_none_prediction_raises_error(self, sample_features):
        """Test deserialize with None prediction raises ValueError."""
        synapse = ConversionSynapse(features=sample_features)
        
        with pytest.raises(ValueError, match="No prediction available"):
            synapse.deserialize()

    def test_synapse_with_all_integer_fields(self):
        """Test synapse initialization with all properly typed integer fields."""
        features = {
            'session_id': 'test_session_comprehensive',
            'conversation_duration_seconds': 180.0,
            'has_target_entity': 1,
            'entities_collected_count': 4,
            'message_ratio': 1.3,
            'hour_of_day': 15,
            'day_of_week': 4,
            'is_business_hours': 1,
            'is_weekend': 0,
            'total_messages': 20,
            'user_messages_count': 8,
            'agent_messages_count': 12,
            'max_message_length_user': 150,
            'min_message_length_user': 10,
            'total_chars_from_user': 800,
            'max_message_length_agent': 200,
            'min_message_length_agent': 20,
            'total_chars_from_agent': 1200,
            'question_count_agent': 5,
            'question_count_user': 3,
            'sequential_user_messages': 2,
            'sequential_agent_messages': 1,
            'entities_collected_count': 4,
            'repeated_questions': 0
        }
        
        valid_prediction = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 150.0
        }
        
        synapse = ConversionSynapse(features=features, prediction=valid_prediction)
        
        # Verify all integer fields are correctly typed
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
                assert isinstance(synapse.features[field], int), f"Field {field} should be int but is {type(synapse.features[field])}"

    def test_synapse_inheritance_from_bt_synapse(self, sample_features, valid_prediction):
        """Test that ConversionSynapse properly inherits from bt.Synapse."""
        synapse = ConversionSynapse(features=sample_features, prediction=valid_prediction)
        
        # Should be an instance of bt.Synapse
        assert isinstance(synapse, bt.Synapse)
        
        # Should have bt.Synapse attributes/methods
        assert hasattr(synapse, 'deserialize')
        assert callable(synapse.deserialize) 
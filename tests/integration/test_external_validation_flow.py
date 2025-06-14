"""
Integration tests for the complete external validation flow.

Tests the end-to-end process:
1. Generate challenge with test_pk
2. Get miner predictions  
3. Call external validation API
4. Calculate rewards based on external ground truth
"""

import pytest
import asyncio
import uuid
from unittest.mock import patch, AsyncMock, MagicMock

from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.validator.validation_client import (
    ValidationAPIClient, 
    ValidationResult, 
    ValidationError,
    configure_default_validation_client
)
from conversion_subnet.validator.forward import (
    forward, 
    get_external_validation,
    validate_prediction
)
from conversion_subnet.validator.reward import Validator


@pytest.mark.asyncio
class TestExternalValidationFlow:
    """Integration tests for external validation workflow."""

    @pytest.fixture
    def mock_validator_neuron(self):
        """Create a mock validator neuron for testing."""
        validator = MagicMock()
        validator.config = MagicMock()
        validator.config.neuron = MagicMock()
        validator.config.neuron.sample_size = 3
        validator.metagraph = MagicMock()
        validator.metagraph.n = 5
        validator.metagraph.axons = [MagicMock() for _ in range(5)]
        validator.dendrite = AsyncMock()
        validator.conversation_history = {}
        validator.update_scores = MagicMock()
        return validator

    @pytest.fixture
    def sample_miner_responses(self):
        """Create sample miner responses."""
        responses = []
        for i in range(3):
            response = ConversionSynapse(features={})
            response.prediction = {
                'conversion_happened': i % 2,  # Alternate between 0 and 1
                'time_to_conversion_seconds': 60.0 if i % 2 else -1.0
            }
            response.confidence = 0.8
            response.response_time = 10.0 + i
            response.miner_uid = i
            responses.append(response)
        return responses

    def setup_method(self):
        """Setup for each test."""
        # Configure a test validation client
        configure_default_validation_client(
            "https://test-api.com/v1",
            "test-api-key",
            30.0
        )

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset global client state
        import conversion_subnet.validator.validation_client
        conversion_subnet.validator.validation_client._default_client = None

    def test_reward_calculation_with_external_ground_truth(self):
        """Test reward calculation using external ground truth."""
        validator = Validator()
        
        # External ground truth: conversion happened
        external_ground_truth = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 60.0
        }
        
        # Test correct prediction
        correct_response = ConversionSynapse(features={})
        correct_response.prediction = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 55.0
        }
        correct_response.confidence = 0.9
        correct_response.response_time = 5.0
        correct_response.miner_uid = 1
        
        correct_reward = validator.reward(external_ground_truth, correct_response)
        
        # Test incorrect prediction  
        incorrect_response = ConversionSynapse(features={})
        incorrect_response.prediction = {
            'conversion_happened': 0,
            'time_to_conversion_seconds': -1.0
        }
        incorrect_response.confidence = 0.7
        incorrect_response.response_time = 8.0
        incorrect_response.miner_uid = 2
        
        incorrect_reward = validator.reward(external_ground_truth, incorrect_response)
        
        # Correct prediction should get higher reward
        assert correct_reward > incorrect_reward
        assert 0.0 <= correct_reward <= 1.0
        assert 0.0 <= incorrect_reward <= 1.0

    async def test_get_external_validation_success(self):
        """Test successful external validation."""
        test_pk = "test-pk-123"
        mock_result = ValidationResult(
            test_pk=test_pk,
            labels=True,
            submission_deadline="2025-06-05T20:34:49.944Z",
            response_time=0.2
        )

        # Configure a client and mock its method
        import conversion_subnet.validator.validation_client
        configure_default_validation_client("https://test-api.com/v1", "test-key", 30.0)

        with patch.object(
            conversion_subnet.validator.validation_client._default_client, 
            'get_validation_result'
        ) as mock_get_result:
            mock_get_result.return_value = mock_result
            
            result = await get_external_validation(test_pk)
            
            assert result == mock_result
            mock_get_result.assert_called_once_with(test_pk)

    async def test_get_external_validation_api_error(self):
        """Test get_external_validation when API returns error."""
        test_pk = "test-pk-456"

        # Reset the default client to ensure our mock is used
        import conversion_subnet.validator.validation_client
        conversion_subnet.validator.validation_client._default_client = None

        with pytest.raises(RuntimeError, match="External validation failed"):
            await get_external_validation(test_pk)

    async def test_get_external_validation_validation_error(self):
        """Test get_external_validation when API returns ValidationError."""
        test_pk = "test-pk-789"

        # Configure a client but make it fail with ValidationError
        import conversion_subnet.validator.validation_client
        configure_default_validation_client("https://test-api.com/v1", "test-key", 30.0)

        with patch.object(
            conversion_subnet.validator.validation_client._default_client, 
            'get_validation_result'
        ) as mock_get_result:
            mock_get_result.side_effect = ValidationError("API Error")
            
            with pytest.raises(ValidationError, match="External validation failed for test_pk"):
                await get_external_validation(test_pk)
            
            mock_get_result.assert_called_once_with(test_pk)

    async def test_get_external_validation_client_not_configured(self):
        """Test get_external_validation when client is not configured."""
        # Reset client configuration
        import conversion_subnet.validator.validation_client
        conversion_subnet.validator.validation_client._default_client = None
        
        test_pk = "test-pk-789"
        
        with pytest.raises(RuntimeError, match="External validation failed"):
            await get_external_validation(test_pk) 
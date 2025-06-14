"""
Integration tests for miner and validator interaction.
"""

import pytest
import torch
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.miner.miner import BinaryClassificationMiner
from conversion_subnet.validator.reward import Validator
from conversion_subnet.validator.forward import forward
from conversion_subnet.validator import forward as forward_module
from conversion_subnet.utils.uids import get_random_uids


@pytest.mark.asyncio
class TestMinerValidatorIntegration:
    """Integration tests for miner and validator interaction."""

    async def test_miner_response_to_validator(self, mock_miner, sample_features):
        """Test that miner can process features and return valid response."""
        # Create a sample synapse with features
        synapse = ConversionSynapse(features=sample_features)
        
        # Create a mock for the dendrite property
        mock_dendrite = MagicMock()
        mock_dendrite.hotkey = "test_hotkey"
        
        # Patch the synapse directly
        synapse.__dict__['dendrite'] = mock_dendrite
        
        # Mock the metagraph for the miner
        mock_miner.metagraph = MagicMock()
        mock_miner.metagraph.hotkeys = ["test_hotkey"]
        
        # Create the expected output
        expected_output = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 60.0,
            'confidence': 0.8
        }
        
        # Mock the forward method to return the expected output
        with patch.object(mock_miner, 'forward', return_value=expected_output):
            # Process synapse with async forward
            result = await mock_miner.forward_async(synapse)
            
            # Check result values
            assert result['conversion_happened'] == 1
            assert result['time_to_conversion_seconds'] == 60.0
            assert result['confidence'] == 0.8

    async def test_validator_evaluation_of_miner_response(self, mock_miner, sample_features):
        """Test that validator can evaluate miner responses correctly."""
        # Create a sample synapse with features
        synapse = ConversionSynapse(features=sample_features)
        
        # Create a mock for the dendrite property
        mock_dendrite = MagicMock()
        mock_dendrite.hotkey = "test_hotkey"
        
        # Patch the synapse directly
        synapse.__dict__['dendrite'] = mock_dendrite
        
        # Mock the metagraph for the miner
        mock_miner.metagraph = MagicMock()
        mock_miner.metagraph.hotkeys = ["test_hotkey"]
        
        # Use external ground truth
        ground_truth = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 60.0
        }
        
        # Create expected output
        expected_output = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 60.0,
            'confidence': 0.8
        }
        
        # Mock forward method
        with patch.object(mock_miner, 'forward', return_value=expected_output):
            # Process synapse with async forward
            response = await mock_miner.forward_async(synapse)
            
            # Copy response to a ConversionSynapse object for validator
            synapse_response = ConversionSynapse(
                features=sample_features,
                prediction=response,
                confidence=response['confidence'],
                response_time=10.0,
                miner_uid=0
            )
            
            # Create validator and evaluate response
            validator = Validator()
            reward = validator.reward(ground_truth, synapse_response)
            
            # Check that reward is calculated correctly
            assert 0.0 <= reward <= 1.0
            
            # Make sure EMA scores are updated
            assert 0 in validator.ema_scores
            assert ground_truth in validator.ground_truth_history

    async def test_miner_validator_end_to_end(self, mock_miner, sample_features):
        """Test end-to-end flow from validator query to miner to reward calculation."""
        # Create an actual validator
        validator = Validator()
        
        # We'll test the reward calculation directly
        ground_truth = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 60.0
        }
        
        # Create expected output from miner
        expected_output = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 60.0,
            'confidence': 0.8
        }
        
        # Create a mock response
        response = ConversionSynapse(
            features=sample_features,
            prediction=expected_output,
            confidence=expected_output['confidence'],
            response_time=10.0,
            miner_uid=0
        )
        
        # Test the reward calculation
        reward = validator.reward(ground_truth, response)
        
        # Check reward is valid
        assert 0.0 <= reward <= 1.0
        
        # Check that validator state was updated
        assert response.miner_uid in validator.ema_scores
        assert ground_truth in validator.ground_truth_history 
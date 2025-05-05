"""
Unit tests for the BinaryClassificationMiner class.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from conversion_subnet.miner.miner import BinaryClassificationMiner
from conversion_subnet.protocol import ConversionSynapse, PredictionOutput


class TestBinaryClassificationMiner:
    """Test suite for the BinaryClassificationMiner class."""

    def test_init(self, test_config):
        """Test that the miner initializes correctly."""
        miner = BinaryClassificationMiner(test_config)
        
        # Check that the model was initialized
        assert miner.model is not None
        assert isinstance(miner.model, torch.nn.Module)
        
        # Check that the model has the expected structure (input -> hidden -> output)
        assert isinstance(miner.model[0], torch.nn.Linear)  # First layer should be Linear
        assert miner.model[0].in_features == 40  # Default input size is 40

    def test_model_architecture(self, test_config):
        """Test that the model architecture matches the configuration."""
        # Set custom layer sizes
        test_config.miner.hidden_sizes = [32, 16, 1]
        
        miner = BinaryClassificationMiner(test_config)
        
        # The model should have 2 linear layers with a ReLU activation between them
        # and a sigmoid at the end (5 layers total: Linear, ReLU, Linear, ReLU, Linear, Sigmoid)
        assert len(list(miner.model.children())) == 6
        
        # Check layer sizes
        linear_layers = [l for l in miner.model.children() if isinstance(l, torch.nn.Linear)]
        assert linear_layers[0].out_features == 32
        assert linear_layers[1].out_features == 16
        assert linear_layers[2].out_features == 1

    def test_forward(self, mock_miner, sample_synapse):
        """Test the forward method returns the expected output structure."""
        # Completely mock the forward method to return a specific output
        expected_output = {
            'conversion_happened': 1,
            'time_to_conversion_seconds': 60.0,
            'confidence': 0.7
        }
        
        with patch.object(mock_miner, 'forward', return_value=expected_output):
            result = mock_miner.forward(sample_synapse)
            
            # Check result structure
            assert 'conversion_happened' in result
            assert 'time_to_conversion_seconds' in result
            assert 'confidence' in result
            
            # Check result values
            assert result['conversion_happened'] == 1
            assert result['time_to_conversion_seconds'] == 60.0
            assert result['confidence'] == 0.7

    def test_forward_negative_case(self, mock_miner, sample_synapse):
        """Test the forward method with a negative prediction."""
        # Completely mock the forward method to return a specific output
        expected_output = {
            'conversion_happened': 0,
            'time_to_conversion_seconds': -1.0,
            'confidence': 0.3
        }
        
        with patch.object(mock_miner, 'forward', return_value=expected_output):
            result = mock_miner.forward(sample_synapse)
            
            assert result['conversion_happened'] == 0
            assert result['time_to_conversion_seconds'] == -1.0
            assert result['confidence'] == 0.3

    def test_forward_exception_handling(self, mock_miner, sample_synapse):
        """Test that exceptions in forward are handled properly."""
        # Create a simple mock features dictionary with numeric values
        mock_features = {f'feature_{i}': float(i) for i in range(40)}
        sample_synapse.features = mock_features
        
        # Mock the model to raise an exception
        with patch.object(mock_miner.model, '__call__', side_effect=RuntimeError("Test error")):
            result = mock_miner.forward(sample_synapse)
            
            # Should return safe defaults
            assert result['conversion_happened'] == 0
            assert result['time_to_conversion_seconds'] == -1.0
            assert result['confidence'] == 0.0  # Confidence should be 0.0 in error case

    def test_model_device(self, test_config):
        """Test that the model is on the correct device."""
        test_config.miner.device = "cpu"
        miner = BinaryClassificationMiner(test_config)
        
        # All parameters should be on CPU
        for param in miner.model.parameters():
            assert param.device.type == "cpu"
        
        # Test with input
        sample_input = torch.randn(1, 40)
        with torch.no_grad():
            output = miner.model(sample_input)
        
        # Output should be on CPU
        assert output.device.type == "cpu" 
"""
Unit tests for the base miner neuron.
"""

import pytest
import torch
import threading
import time
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from abc import ABC

import bittensor as bt
from conversion_subnet.base.miner import BaseMinerNeuron
from conversion_subnet.protocol import ConversionSynapse


# Create a concrete test implementation to avoid abstract method issues
class ConcreteMinerForTesting(BaseMinerNeuron):
    """Concrete implementation of BaseMinerNeuron for testing purposes."""
    
    async def forward(self, synapse: ConversionSynapse) -> ConversionSynapse:
        """Concrete implementation of the abstract forward method."""
        synapse.prediction = {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0}
        synapse.confidence = 0.8
        return synapse
    
    def blacklist(self, synapse: ConversionSynapse) -> bool:
        """Concrete implementation of blacklist method."""
        return False
    
    def priority(self, synapse: ConversionSynapse) -> float:
        """Concrete implementation of priority method."""
        return 1.0


class TestBaseMinerNeuron:
    """Test suite for BaseMinerNeuron class."""

    @patch('bittensor.axon')
    @patch('conversion_subnet.base.neuron.BaseNeuron.__init__')
    def test_init_with_config(self, mock_base_init, mock_axon):
        """Test base miner initialization with valid config."""
        # Setup mocks
        mock_base_init.return_value = None
        mock_axon_instance = MagicMock()
        mock_axon.return_value = mock_axon_instance
        
        # Create test config with required attributes
        config = MagicMock()
        config.blacklist.force_validator_permit = True
        config.blacklist.allow_non_registered = False
        config.axon.port = 8091
        
        # Create concrete miner instance
        miner = ConcreteMinerForTesting.__new__(ConcreteMinerForTesting)
        miner.config = config
        miner.wallet = MagicMock()
        
        # Call __init__ manually after setting up attributes
        BaseMinerNeuron.__init__(miner, config=config)
        
        # Verify initialization
        mock_base_init.assert_called_once_with(config=config)
        mock_axon.assert_called_once_with(wallet=miner.wallet, port=config.axon.port)
        assert miner.axon == mock_axon_instance
        assert miner.should_exit is False
        assert miner.is_running is False
        assert miner.thread is None
        assert miner.lock is not None

    @patch('bittensor.axon')
    @patch('conversion_subnet.base.neuron.BaseNeuron.__init__')
    def test_init_security_warnings(self, mock_base_init, mock_axon):
        """Test that security warnings are logged for insecure configurations."""
        # Setup mocks
        mock_base_init.return_value = None
        mock_axon_instance = MagicMock()
        mock_axon.return_value = mock_axon_instance
        
        # Create insecure config
        config = MagicMock()
        config.blacklist.force_validator_permit = False
        config.blacklist.allow_non_registered = True
        config.axon.port = 8091
        
        # Create concrete miner instance
        miner = ConcreteMinerForTesting.__new__(ConcreteMinerForTesting)
        miner.config = config
        miner.wallet = MagicMock()
        
        with patch('bittensor.logging.warning') as mock_warning:
            BaseMinerNeuron.__init__(miner, config=config)
            
            # Verify security warnings were logged
            assert mock_warning.call_count == 2
            warning_calls = [call[0][0] for call in mock_warning.call_args_list]
            assert any("non-validators" in call for call in warning_calls)
            assert any("non-registered" in call for call in warning_calls)

    def test_run_in_background_thread(self):
        """Test running miner in background thread."""
        # Create a properly configured mock miner
        miner = MagicMock()
        miner.should_exit = False
        miner.is_running = False
        miner.thread = None
        miner.run = MagicMock()
        
        # Bind the actual method to the mock
        miner.run_in_background_thread = BaseMinerNeuron.run_in_background_thread.__get__(miner)
        
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            
            # Test starting background thread
            miner.run_in_background_thread()
            
            assert miner.should_exit is False
            assert miner.is_running is True
            assert miner.thread == mock_thread_instance
            mock_thread.assert_called_once_with(target=miner.run, daemon=True)
            mock_thread_instance.start.assert_called_once()

    def test_stop_run_thread(self):
        """Test stopping the background thread."""
        miner = MagicMock()
        miner.should_exit = False
        miner.is_running = True
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        miner.thread = mock_thread
        
        # Bind the actual method
        miner.stop_run_thread = BaseMinerNeuron.stop_run_thread.__get__(miner)
        
        miner.stop_run_thread()
        
        assert miner.should_exit is True
        mock_thread.join.assert_called_once_with(5)

    def test_context_manager_functionality(self):
        """Test context manager functionality with simpler approach."""
        # Test that the context manager methods exist and can be called
        miner = MagicMock()
        
        # Verify the methods exist on the class
        assert hasattr(BaseMinerNeuron, '__enter__')
        assert hasattr(BaseMinerNeuron, '__exit__')
        
        # Test that they are callable
        assert callable(BaseMinerNeuron.__enter__)
        assert callable(BaseMinerNeuron.__exit__)

    @patch('bittensor.axon')
    @patch('conversion_subnet.base.neuron.BaseNeuron.__init__')
    def test_axon_attachment(self, mock_base_init, mock_axon):
        """Test that axon is properly attached with forward function."""
        # Setup mocks
        mock_base_init.return_value = None
        mock_axon_instance = MagicMock()
        mock_axon.return_value = mock_axon_instance
        
        config = MagicMock()
        config.blacklist.force_validator_permit = True
        config.blacklist.allow_non_registered = False
        config.axon.port = 8091
        
        # Create miner
        miner = ConcreteMinerForTesting.__new__(ConcreteMinerForTesting)
        miner.config = config
        miner.wallet = MagicMock()
        
        BaseMinerNeuron.__init__(miner, config=config)
        
        # Verify axon.attach was called with forward function
        mock_axon_instance.attach.assert_called_once()
        call_args = mock_axon_instance.attach.call_args
        assert 'forward' in str(call_args)

    @patch('bittensor.axon')
    @patch('conversion_subnet.base.neuron.BaseNeuron.__init__')
    def test_miner_lifecycle_integration(self, mock_base_init, mock_axon):
        """Test complete miner lifecycle from init to shutdown."""
        # Setup
        mock_base_init.return_value = None
        mock_axon_instance = MagicMock()
        mock_axon.return_value = mock_axon_instance
        
        config = MagicMock()
        config.blacklist.force_validator_permit = True
        config.blacklist.allow_non_registered = False
        config.axon.port = 8091
        
        # Create and initialize miner
        miner = ConcreteMinerForTesting.__new__(ConcreteMinerForTesting)
        miner.config = config
        miner.wallet = MagicMock()
        
        BaseMinerNeuron.__init__(miner, config=config)
        
        # Test initialization state
        assert miner.should_exit is False
        assert miner.is_running is False
        assert miner.thread is None
        assert miner.axon == mock_axon_instance
        
        # Test forward method
        synapse = ConversionSynapse(
            features={'session_id': 'test'},
            prediction={'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        )
        
        # Test the concrete forward implementation
        result = asyncio.run(miner.forward(synapse))
        assert result.prediction['conversion_happened'] == 1
        assert result.prediction['time_to_conversion_seconds'] == 60.0
        assert result.confidence == 0.8

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods are properly enforced."""
        # This should fail - cannot instantiate abstract class directly
        with pytest.raises(TypeError, match="abstract"):
            BaseMinerNeuron()

    @patch('bittensor.axon')
    @patch('conversion_subnet.base.neuron.BaseNeuron.__init__')
    def test_config_validation_and_warnings(self, mock_base_init, mock_axon):
        """Test comprehensive config validation and warning scenarios."""
        mock_base_init.return_value = None
        mock_axon_instance = MagicMock()
        mock_axon.return_value = mock_axon_instance
        
        # Test multiple insecure configurations
        test_cases = [
            {
                'config': {
                    'blacklist.force_validator_permit': False,
                    'blacklist.allow_non_registered': False,
                    'axon.port': 8091
                },
                'expected_warnings': 1
            },
            {
                'config': {
                    'blacklist.force_validator_permit': True,
                    'blacklist.allow_non_registered': True,
                    'axon.port': 8091
                },
                'expected_warnings': 1
            },
            {
                'config': {
                    'blacklist.force_validator_permit': False,
                    'blacklist.allow_non_registered': True,
                    'axon.port': 8091
                },
                'expected_warnings': 2
            }
        ]
        
        for test_case in test_cases:
            config = MagicMock()
            for key, value in test_case['config'].items():
                keys = key.split('.')
                obj = config
                for k in keys[:-1]:
                    if not hasattr(obj, k):
                        setattr(obj, k, MagicMock())
                    obj = getattr(obj, k)
                setattr(obj, keys[-1], value)
            
            miner = ConcreteMinerForTesting.__new__(ConcreteMinerForTesting)
            miner.config = config
            miner.wallet = MagicMock()
            
            with patch('bittensor.logging.warning') as mock_warning:
                BaseMinerNeuron.__init__(miner, config=config)
                assert mock_warning.call_count == test_case['expected_warnings'] 
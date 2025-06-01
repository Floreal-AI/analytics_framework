"""
Unit tests for the UIDs utility module.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
import bittensor as bt

from conversion_subnet.utils.uids import check_uid_availability, get_random_uids


class TestCheckUidAvailability:
    """Test suite for check_uid_availability function."""
    
    @pytest.fixture
    def mock_metagraph(self):
        """Create a mock metagraph for testing."""
        metagraph = MagicMock()
        metagraph.axons = []
        metagraph.validator_permit = []
        metagraph.S = []
        return metagraph

    def test_uid_not_serving(self, mock_metagraph):
        """Test that non-serving UIDs are not available."""
        # Setup mock axon that is not serving
        mock_axon = MagicMock()
        mock_axon.is_serving = False
        mock_metagraph.axons = [mock_axon]
        mock_metagraph.validator_permit = [False]
        mock_metagraph.S = [100]
        
        result = check_uid_availability(mock_metagraph, 0, 1024)
        assert result is False

    def test_uid_serving_no_validator_permit(self, mock_metagraph):
        """Test that serving UID without validator permit is available."""
        # Setup mock axon that is serving
        mock_axon = MagicMock()
        mock_axon.is_serving = True
        mock_metagraph.axons = [mock_axon]
        mock_metagraph.validator_permit = [False]
        mock_metagraph.S = [100]
        
        result = check_uid_availability(mock_metagraph, 0, 1024)
        assert result is True

    def test_uid_serving_with_validator_permit_low_stake(self, mock_metagraph):
        """Test that serving UID with validator permit and low stake is available."""
        mock_axon = MagicMock()
        mock_axon.is_serving = True
        mock_metagraph.axons = [mock_axon]
        mock_metagraph.validator_permit = [True]
        mock_metagraph.S = [500]  # Below limit
        
        result = check_uid_availability(mock_metagraph, 0, 1024)
        assert result is True

    def test_uid_serving_with_validator_permit_high_stake(self, mock_metagraph):
        """Test that serving UID with validator permit and high stake is not available."""
        mock_axon = MagicMock()
        mock_axon.is_serving = True
        mock_metagraph.axons = [mock_axon]
        mock_metagraph.validator_permit = [True]
        mock_metagraph.S = [2000]  # Above limit
        
        result = check_uid_availability(mock_metagraph, 0, 1024)
        assert result is False

    def test_uid_serving_with_validator_permit_exact_limit(self, mock_metagraph):
        """Test UID with stake exactly at the limit."""
        mock_axon = MagicMock()
        mock_axon.is_serving = True
        mock_metagraph.axons = [mock_axon]
        mock_metagraph.validator_permit = [True]
        mock_metagraph.S = [1024]  # Exactly at limit
        
        result = check_uid_availability(mock_metagraph, 0, 1024)
        assert result is True  # Should be available (not greater than limit)


class TestGetRandomUids:
    """Test suite for get_random_uids function."""
    
    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator object for testing."""
        validator = MagicMock()
        validator.metagraph = MagicMock()
        validator.config = MagicMock()
        validator.config.neuron = MagicMock()
        validator.config.neuron.vpermit_tao_limit = 1024
        return validator

    def setup_metagraph_with_uids(self, validator, uid_configs):
        """Helper to setup metagraph with specific UID configurations."""
        n_uids = len(uid_configs)
        validator.metagraph.n = torch.tensor(n_uids)
        
        # Setup axons
        axons = []
        validator_permits = []
        stakes = []
        
        for config in uid_configs:
            axon = MagicMock()
            axon.is_serving = config.get('is_serving', True)
            axons.append(axon)
            validator_permits.append(config.get('validator_permit', False))
            stakes.append(config.get('stake', 100))
        
        validator.metagraph.axons = axons
        validator.metagraph.validator_permit = validator_permits
        validator.metagraph.S = stakes

    @patch('conversion_subnet.utils.uids.random.sample')
    def test_get_random_uids_basic(self, mock_sample, mock_validator):
        """Test basic functionality of get_random_uids."""
        # Setup 3 available UIDs
        uid_configs = [
            {'is_serving': True, 'validator_permit': False, 'stake': 100},
            {'is_serving': True, 'validator_permit': False, 'stake': 200},
            {'is_serving': True, 'validator_permit': False, 'stake': 300}
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        # Mock random.sample to return specific UIDs
        mock_sample.return_value = [0, 2]
        
        result = get_random_uids(mock_validator, k=2)
        
        # Verify result
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long
        assert len(result) == 2
        assert list(result) == [0, 2]
        
        # Verify random.sample was called correctly
        mock_sample.assert_called_once_with([0, 1, 2], 2)

    @patch('conversion_subnet.utils.uids.random.sample')
    def test_get_random_uids_with_exclusions(self, mock_sample, mock_validator):
        """Test get_random_uids with excluded UIDs."""
        uid_configs = [
            {'is_serving': True, 'validator_permit': False, 'stake': 100},
            {'is_serving': True, 'validator_permit': False, 'stake': 200},
            {'is_serving': True, 'validator_permit': False, 'stake': 300},
            {'is_serving': True, 'validator_permit': False, 'stake': 400}
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        mock_sample.return_value = [1, 3]
        
        result = get_random_uids(mock_validator, k=2, exclude=[0, 2])
        
        # Should only sample from non-excluded UIDs [1, 3]
        mock_sample.assert_called_once_with([1, 3], 2)
        assert list(result) == [1, 3]

    @patch('conversion_subnet.utils.uids.random.sample')
    @patch('bittensor.logging.warning')
    def test_get_random_uids_no_candidates_fallback(self, mock_warning, mock_sample, mock_validator):
        """Test fallback when no candidate UIDs are available after exclusions."""
        uid_configs = [
            {'is_serving': True, 'validator_permit': False, 'stake': 100},
            {'is_serving': True, 'validator_permit': False, 'stake': 200}
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        # Exclude all UIDs
        mock_sample.return_value = [0, 1]
        
        result = get_random_uids(mock_validator, k=2, exclude=[0, 1])
        
        # Should fall back to all available UIDs
        mock_warning.assert_called_with("No available candidate UIDs found. Using all available UIDs.")
        mock_sample.assert_called_once_with([0, 1], 2)

    @patch('conversion_subnet.utils.uids.random.sample')
    @patch('bittensor.logging.warning')
    def test_get_random_uids_insufficient_candidates(self, mock_warning, mock_sample, mock_validator):
        """Test when requested k is larger than available candidates."""
        uid_configs = [
            {'is_serving': True, 'validator_permit': False, 'stake': 100},
            {'is_serving': True, 'validator_permit': False, 'stake': 200}
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        mock_sample.return_value = [0, 1]
        
        result = get_random_uids(mock_validator, k=5)  # Request more than available
        
        # Should adjust k and warn
        mock_warning.assert_called_with("Requested 5 UIDs but only 2 are available.")
        mock_sample.assert_called_once_with([0, 1], 2)
        assert len(result) == 2

    @patch('bittensor.logging.warning')
    def test_get_random_uids_no_available_uids(self, mock_warning, mock_validator):
        """Test when no UIDs are available at all."""
        uid_configs = [
            {'is_serving': False, 'validator_permit': False, 'stake': 100},
            {'is_serving': False, 'validator_permit': False, 'stake': 200}
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        result = get_random_uids(mock_validator, k=2)
        
        # Should return empty tensor and warn
        mock_warning.assert_called_with("No UIDs available at all. Returning empty tensor.")
        assert isinstance(result, torch.Tensor)
        assert len(result) == 0
        assert result.dtype == torch.long

    @patch('conversion_subnet.utils.uids.random.sample')
    def test_get_random_uids_mixed_availability(self, mock_sample, mock_validator):
        """Test with mixed UID availability scenarios."""
        uid_configs = [
            {'is_serving': True, 'validator_permit': False, 'stake': 100},      # Available
            {'is_serving': False, 'validator_permit': False, 'stake': 200},     # Not serving
            {'is_serving': True, 'validator_permit': True, 'stake': 2000},      # High stake validator
            {'is_serving': True, 'validator_permit': True, 'stake': 500},       # Low stake validator
            {'is_serving': True, 'validator_permit': False, 'stake': 300}       # Available
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        mock_sample.return_value = [0, 3]
        
        result = get_random_uids(mock_validator, k=2)
        
        # Should only sample from available UIDs: [0, 3, 4]
        # (UID 1 not serving, UID 2 has high stake)
        mock_sample.assert_called_once_with([0, 3, 4], 2)

    @patch('conversion_subnet.utils.uids.random.sample')
    def test_get_random_uids_zero_k(self, mock_sample, mock_validator):
        """Test requesting zero UIDs."""
        uid_configs = [
            {'is_serving': True, 'validator_permit': False, 'stake': 100}
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        result = get_random_uids(mock_validator, k=0)
        
        # Should return empty tensor without calling random.sample
        assert isinstance(result, torch.Tensor)
        assert len(result) == 0
        mock_sample.assert_not_called()

    @patch('conversion_subnet.utils.uids.random.sample')
    def test_get_random_uids_single_uid(self, mock_sample, mock_validator):
        """Test requesting single UID."""
        uid_configs = [
            {'is_serving': True, 'validator_permit': False, 'stake': 100},
            {'is_serving': True, 'validator_permit': False, 'stake': 200}
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        mock_sample.return_value = [1]
        
        result = get_random_uids(mock_validator, k=1)
        
        assert len(result) == 1
        assert result[0] == 1
        mock_sample.assert_called_once_with([0, 1], 1)

    def test_get_random_uids_empty_metagraph(self, mock_validator):
        """Test with empty metagraph (no UIDs)."""
        mock_validator.metagraph.n = torch.tensor(0)
        mock_validator.metagraph.axons = []
        mock_validator.metagraph.validator_permit = []
        mock_validator.metagraph.S = []
        
        with patch('bittensor.logging.warning') as mock_warning:
            result = get_random_uids(mock_validator, k=2)
            
            mock_warning.assert_called_with("No UIDs available at all. Returning empty tensor.")
            assert len(result) == 0

    @patch('conversion_subnet.utils.uids.random.sample')
    def test_get_random_uids_exclude_none(self, mock_sample, mock_validator):
        """Test with exclude=None (should be treated as no exclusions)."""
        uid_configs = [
            {'is_serving': True, 'validator_permit': False, 'stake': 100},
            {'is_serving': True, 'validator_permit': False, 'stake': 200}
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        mock_sample.return_value = [0, 1]
        
        result = get_random_uids(mock_validator, k=2, exclude=None)
        
        # Should include all available UIDs
        mock_sample.assert_called_once_with([0, 1], 2)

    @patch('conversion_subnet.utils.uids.random.sample')
    def test_get_random_uids_exclude_empty_list(self, mock_sample, mock_validator):
        """Test with empty exclusion list."""
        uid_configs = [
            {'is_serving': True, 'validator_permit': False, 'stake': 100},
            {'is_serving': True, 'validator_permit': False, 'stake': 200}
        ]
        self.setup_metagraph_with_uids(mock_validator, uid_configs)
        
        mock_sample.return_value = [0, 1]
        
        result = get_random_uids(mock_validator, k=2, exclude=[])
        
        # Should include all available UIDs
        mock_sample.assert_called_once_with([0, 1], 2) 
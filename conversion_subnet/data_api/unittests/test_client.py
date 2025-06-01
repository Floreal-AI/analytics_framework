"""
Test suite for simplified data API client.

Following TDD principles:
- Test first
- Many assertions to catch issues early
- No fallback testing
- Clear error validation
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import patch, AsyncMock
from aiohttp import ClientError
import os

from conversion_subnet.data_api.core import (
    VoiceFormAPIClient, 
    VoiceFormAPIConfig,
    TrainingData, 
    TestData, 
    APIError, 
    DataValidationError
)


class TestVoiceFormAPIConfig:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        config = VoiceFormAPIConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            timeout_seconds=30
        )
        
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com"
        assert config.timeout_seconds == 30
    
    def test_config_validation_assertions(self):
        """Test configuration validation with assertions."""
        # Empty API key should fail
        with pytest.raises(AssertionError, match="API key cannot be empty"):
            VoiceFormAPIConfig(
                api_key="",
                base_url="https://api.example.com"
            )
        
        # Empty base URL should fail
        with pytest.raises(AssertionError, match="Base URL cannot be empty"):
            VoiceFormAPIConfig(
                api_key="test-key",
                base_url=""
            )
        
        # Invalid timeout should fail
        with pytest.raises(AssertionError, match="Timeout must be positive"):
            VoiceFormAPIConfig(
                api_key="test-key",
                base_url="https://api.example.com",
                timeout_seconds=0
            )
        
        # Invalid URL format should fail
        with pytest.raises(AssertionError, match="Invalid URL format"):
            VoiceFormAPIConfig(
                api_key="test-key",
                base_url="not-a-url"
            )
    
    def test_from_env_success(self):
        """Test successful creation from environment variables."""
        with patch.dict(os.environ, {
            'VOICEFORM_API_KEY': 'test-key',
            'VOICEFORM_API_BASE_URL': 'https://api.example.com',
            'DATA_API_TIMEOUT': '45'
        }):
            config = VoiceFormAPIConfig.from_env()
            
            assert config.api_key == 'test-key'
            assert config.base_url == 'https://api.example.com'
            assert config.timeout_seconds == 45
    
    def test_from_env_missing_key(self):
        """Test environment creation with missing API key."""
        with patch('conversion_subnet.data_api.core.config.DOTENV_AVAILABLE', False), \
             patch.dict(os.environ, {
                 'VOICEFORM_API_BASE_URL': 'https://api.example.com'
             }, clear=True):
            with pytest.raises(ValueError, match="VOICEFORM_API_KEY environment variable is required"):
                VoiceFormAPIConfig.from_env()
    
    def test_from_env_missing_url(self):
        """Test environment creation with missing URL."""
        with patch('conversion_subnet.data_api.core.config.DOTENV_AVAILABLE', False), \
             patch.dict(os.environ, {
                 'VOICEFORM_API_KEY': 'test-key'
             }, clear=True):
            with pytest.raises(ValueError, match="VOICEFORM_API_BASE_URL environment variable is required"):
                VoiceFormAPIConfig.from_env()
    
    def test_from_env_with_dotenv_path(self):
        """Test environment creation with custom dotenv path."""
        # Mock dotenv functionality
        with patch('conversion_subnet.data_api.core.config.DOTENV_AVAILABLE', True), \
             patch('conversion_subnet.data_api.core.config.load_dotenv') as mock_load_dotenv, \
             patch('conversion_subnet.data_api.core.config.Path') as mock_path, \
             patch.dict(os.environ, {
                 'VOICEFORM_API_KEY': 'dotenv-key',
                 'VOICEFORM_API_BASE_URL': 'https://dotenv.example.com',
                 'DATA_API_TIMEOUT': '60'
             }):
            
            # Mock path existence
            mock_path.return_value.exists.return_value = True
            
            config = VoiceFormAPIConfig.from_env('custom.env')
            
            # Verify dotenv was called with custom path
            mock_load_dotenv.assert_called_once_with('custom.env')
            
            # Verify config values
            assert config.api_key == 'dotenv-key'
            assert config.base_url == 'https://dotenv.example.com'
            assert config.timeout_seconds == 60
    
    def test_from_env_dotenv_file_not_found(self):
        """Test error when custom dotenv file doesn't exist."""
        with patch('conversion_subnet.data_api.core.config.DOTENV_AVAILABLE', True), \
             patch('conversion_subnet.data_api.core.config.Path') as mock_path:
            
            # Mock path doesn't exist
            mock_path.return_value.exists.return_value = False
            
            with pytest.raises(ValueError, match=".env file not found: custom.env"):
                VoiceFormAPIConfig.from_env('custom.env')
    
    def test_from_env_dotenv_not_available(self):
        """Test error when dotenv is requested but not installed."""
        with patch('conversion_subnet.data_api.core.config.DOTENV_AVAILABLE', False):
            with pytest.raises(ImportError, match="python-dotenv is required"):
                VoiceFormAPIConfig.from_env('custom.env')


class TestVoiceFormAPIClient:
    """Test the simplified client functionality."""
    
    @pytest.fixture
    def config(self):
        """Test configuration fixture."""
        return VoiceFormAPIConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            timeout_seconds=30
        )
    
    @pytest.fixture
    def client(self, config):
        """Test client fixture."""
        return VoiceFormAPIClient(config)
    
    @pytest.fixture
    def mock_training_response(self):
        """Mock training data response."""
        return {
            'train_features': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            'train_targets': [0, 1],
            'features_list': ['feature1', 'feature2', 'feature3'],
            'round_number': 1,
            'refresh_date': '2023-12-01T10:30:00Z'
        }
    
    @pytest.fixture
    def mock_test_response(self):
        """Mock test data response."""
        return {
            'train_features': [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            'features_list': ['feature1', 'feature2', 'feature3'],
            'submissionDeadline': '2024-01-15T23:59:59Z'
        }
    
    def test_client_initialization(self, config):
        """Test client initialization."""
        client = VoiceFormAPIClient(config)
        
        assert client.config == config
        assert client._session is None
    
    def test_from_env_creation(self):
        """Test client creation from environment."""
        with patch.dict(os.environ, {
            'VOICEFORM_API_KEY': 'env-key',
            'VOICEFORM_API_BASE_URL': 'https://env.example.com'
        }):
            client = VoiceFormAPIClient.from_env()
            
            assert client.config.api_key == 'env-key'
            assert client.config.base_url == 'https://env.example.com'
    
    def test_get_headers(self, client):
        """Test HTTP headers generation."""
        headers = client._get_headers()
        
        assert headers['x-api-key'] == 'test-key'
        assert headers['accept'] == 'application/json'
        assert headers['user-agent'] == 'simple-data-client/1.0'
    
    @pytest.mark.asyncio
    async def test_fetch_training_data_success(self, client, mock_training_response):
        """Test successful training data fetch."""
        with patch.object(client, '_make_request', return_value=mock_training_response) as mock_request:
            result = await client.fetch_training_data(limit=50, offset=10, round_number=2, save=False)
            
            # Verify request parameters
            mock_request.assert_called_once_with('/train-data', {
                'limit': 50,
                'offset': 10,
                'roundNumber': 2
            })
            
            # Verify result structure and types
            assert isinstance(result, TrainingData)
            assert isinstance(result.features, np.ndarray)
            assert isinstance(result.targets, np.ndarray)
            assert isinstance(result.feature_names, list)
            assert isinstance(result.round_number, int)
            assert isinstance(result.updated_at, str)
            
            # Verify data content
            assert result.features.shape == (2, 3)
            assert result.targets.shape == (2,)
            assert result.feature_names == ['feature1', 'feature2', 'feature3']
            assert result.round_number == 1
            assert result.updated_at == '2023-12-01T10:30:00Z'
            
            # Verify data values
            np.testing.assert_array_equal(result.features, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            np.testing.assert_array_equal(result.targets, [0, 1])
    
    @pytest.mark.asyncio
    async def test_fetch_test_data_success(self, client, mock_test_response):
        """Test successful test data fetch."""
        with patch.object(client, '_make_request', return_value=mock_test_response) as mock_request:
            result = await client.fetch_test_data(limit=25, offset=5, save=False)
            
            # Verify request parameters
            mock_request.assert_called_once_with('/test-data', {
                'limit': 25,
                'offset': 5
            })
            
            # Verify result structure and types
            assert isinstance(result, TestData)
            assert isinstance(result.features, np.ndarray)
            assert isinstance(result.feature_names, list)
            assert isinstance(result.submission_deadline, str)
            
            # Verify data content
            assert result.features.shape == (2, 3)
            assert result.feature_names == ['feature1', 'feature2', 'feature3']
            assert result.submission_deadline == '2024-01-15T23:59:59Z'
            
            # Verify data values
            np.testing.assert_array_equal(result.features, [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    
    @pytest.mark.asyncio
    async def test_parameter_validation_assertions(self, client):
        """Test parameter validation with assertions."""
        # Invalid limit
        with pytest.raises(AssertionError, match="Limit must be positive"):
            await client.fetch_training_data(limit=0, save=False)
        
        # Invalid offset
        with pytest.raises(AssertionError, match="Offset must be non-negative"):
            await client.fetch_training_data(offset=-1, save=False)
        
        # Invalid round number
        with pytest.raises(AssertionError, match="Round number must be positive"):
            await client.fetch_training_data(round_number=0, save=False)
        
        # Test data parameter validation
        with pytest.raises(AssertionError, match="Limit must be positive"):
            await client.fetch_test_data(limit=-5, save=False)
        
        with pytest.raises(AssertionError, match="Offset must be non-negative"):
            await client.fetch_test_data(offset=-10, save=False)
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, client):
        """Test API error handling without fallbacks."""
        # Test HTTP error
        with patch.object(client, '_make_request', side_effect=APIError("HTTP 500", status_code=500)):
            with pytest.raises(APIError) as exc_info:
                await client.fetch_training_data(save=False)
            
            assert exc_info.value.status_code == 500
            assert "HTTP 500" in str(exc_info.value)
        
        # Test network error
        with patch.object(client, '_make_request', side_effect=APIError("Network error")):
            with pytest.raises(APIError) as exc_info:
                await client.fetch_test_data(save=False)
            
            assert "Network error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, client):
        """Test data validation error handling."""
        # Missing required fields
        invalid_response = {'train_features': [[1, 2, 3]]}
        
        with patch.object(client, '_make_request', return_value=invalid_response):
            with pytest.raises(DataValidationError) as exc_info:
                await client.fetch_training_data(save=False)
            
            assert "Missing required fields" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_fetch_both_success(self, client, mock_training_response, mock_test_response):
        """Test concurrent fetch of both datasets."""
        with patch.object(client, 'fetch_training_data', return_value=TrainingData(
            features=np.array([[1.0, 2.0], [3.0, 4.0]]),
            targets=np.array([0, 1]),
            feature_names=['f1', 'f2'],
            round_number=1,
            updated_at='2023-12-01T10:30:00Z'
        )) as mock_train, \
        patch.object(client, 'fetch_test_data', return_value=TestData(
            features=np.array([[5.0, 6.0], [7.0, 8.0]]),
            feature_names=['f1', 'f2'],
            submission_deadline='2024-01-15T23:59:59Z'
        )) as mock_test:
            
            train_data, test_data = await client.fetch_both(
                train_limit=50,
                train_offset=10,
                round_number=2,
                test_limit=25,
                test_offset=5,
                save_training=False,
                save_test=False
            )
            
            # Verify both methods were called with correct parameters
            mock_train.assert_called_once_with(50, 10, 2, False)
            mock_test.assert_called_once_with(25, 5, False)
            
            # Verify results
            assert isinstance(train_data, TrainingData)
            assert isinstance(test_data, TestData)
            assert train_data.feature_names == test_data.feature_names
    
    @pytest.mark.asyncio
    async def test_fetch_both_feature_mismatch(self, client):
        """Test fetch_both with mismatched feature names."""
        train_data = TrainingData(
            features=np.array([[1.0, 2.0]]),
            targets=np.array([0]),
            feature_names=['f1', 'f2'],
            round_number=1,
            updated_at='2023-12-01T10:30:00Z'
        )
        
        test_data = TestData(
            features=np.array([[1.0, 2.0]]),
            feature_names=['f1', 'f3'],  # Different feature names
            submission_deadline='2024-01-15T23:59:59Z'
        )
        
        with patch.object(client, 'fetch_training_data', return_value=train_data), \
             patch.object(client, 'fetch_test_data', return_value=test_data):
            
            with pytest.raises(DataValidationError, match="different feature names"):
                await client.fetch_both(save_training=False, save_test=False)
    
    @pytest.mark.asyncio
    async def test_session_management(self, client):
        """Test HTTP session creation and cleanup."""
        # Initially no session
        assert client._session is None
        
        # Session created on first request
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            session = await client._get_session()
            
            assert session == mock_session
            assert client._session == mock_session
            mock_session_class.assert_called_once()
        
        # Test session cleanup
        await client.close()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_training_data_csv_saving(self, client, mock_training_response, tmp_path):
        """Test CSV saving functionality for training data."""
        import os
        
        # Change to temporary directory for testing
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch.object(client, '_make_request', return_value=mock_training_response):
                # Test with save=True (default)
                result = await client.fetch_training_data(limit=10, save=True)
                
                # Check that CSV file was created
                csv_path = tmp_path / "data" / "train_data.csv"
                assert csv_path.exists()
                
                # Check CSV content
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                # Verify structure
                assert df.shape == (2, 4)  # 2 rows, 3 features + 1 target
                assert list(df.columns) == ['feature1', 'feature2', 'feature3', 'target']
                assert list(df['target']) == [0, 1]
                
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_test_data_csv_saving(self, client, mock_test_response, tmp_path):
        """Test CSV saving functionality for test data."""
        import os
        
        # Change to temporary directory for testing
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch.object(client, '_make_request', return_value=mock_test_response):
                # Test with save=True (not default for test data)
                result = await client.fetch_test_data(limit=10, save=True)
                
                # Check that CSV file was created
                csv_path = tmp_path / "data" / "test_data.csv"
                assert csv_path.exists()
                
                # Check CSV content
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                # Verify structure
                assert df.shape == (2, 3)  # 2 rows, 3 features (no target)
                assert list(df.columns) == ['feature1', 'feature2', 'feature3']
                
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_csv_saving_disabled_by_default(self, client, mock_test_response, tmp_path):
        """Test that CSV saving is disabled by default for test data."""
        import os
        
        # Change to temporary directory for testing
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch.object(client, '_make_request', return_value=mock_test_response):
                # Test with default save=False for test data
                result = await client.fetch_test_data(limit=10)
                
                # Check that CSV file was NOT created
                csv_path = tmp_path / "data" / "test_data.csv"
                assert not csv_path.exists()
                
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 
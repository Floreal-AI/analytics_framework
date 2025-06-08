"""
Test module for VoiceFormAPIClient - Following TDD principles.

These tests define the expected behavior of the API client before implementation.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

# Import the specific exceptions
from conversion_subnet.data_api.core.models import DataValidationError, APIError


def create_test_client(api_key="test-key", base_url="https://api.example.com", timeout=30):
    """Helper function to create test client instances."""
    from conversion_subnet.data_api import VoiceFormAPIClient, VoiceFormAPIConfig
    config = VoiceFormAPIConfig(api_key=api_key, base_url=base_url, timeout_seconds=timeout)
    return VoiceFormAPIClient(config)


class TestVoiceFormAPIClientBasic:
    """Test basic functionality and configuration of VoiceFormAPIClient."""
    
    def test_client_requires_api_key(self):
        """Test that client raises error without API key."""
        # This is testing that we don't allow empty/None API keys
        with pytest.raises(AssertionError, match="API key cannot be empty"):
            from conversion_subnet.data_api import VoiceFormAPIClient, VoiceFormAPIConfig
            config = VoiceFormAPIConfig(api_key="", base_url="https://example.com")
            VoiceFormAPIClient(config)
            
        with pytest.raises(AssertionError, match="API key cannot be empty"):
            from conversion_subnet.data_api import VoiceFormAPIClient, VoiceFormAPIConfig
            config = VoiceFormAPIConfig(api_key="", base_url="https://example.com")
            VoiceFormAPIClient(config)
    
    def test_client_requires_base_url(self):
        """Test that client requires valid base URL."""
        with pytest.raises(AssertionError, match="Base URL cannot be empty"):
            from conversion_subnet.data_api import VoiceFormAPIClient, VoiceFormAPIConfig
            config = VoiceFormAPIConfig(api_key="test-key", base_url="")
            VoiceFormAPIClient(config)
            
    def test_client_validates_url_format(self):
        """Test that client validates URL format."""
        with pytest.raises(AssertionError, match="Invalid URL format"):
            from conversion_subnet.data_api import VoiceFormAPIClient, VoiceFormAPIConfig
            config = VoiceFormAPIConfig(api_key="test-key", base_url="not-a-url")
            VoiceFormAPIClient(config)
    
    def test_client_initialization_success(self):
        """Test successful client initialization."""
        from conversion_subnet.data_api import VoiceFormAPIClient, VoiceFormAPIConfig
        config = VoiceFormAPIConfig(
            api_key="test-key", 
            base_url="https://api.example.com"
        )
        client = VoiceFormAPIClient(config)
        assert client.config.api_key == "test-key"
        assert client.config.base_url == "https://api.example.com"
        assert client.config.timeout_seconds > 0  # Should have reasonable default timeout
        

class TestTrainDataFetching:
    """Test training data fetching functionality."""
    
    @pytest.fixture
    def mock_train_response(self):
        """Mock response data for training endpoint."""
        return {
            'train_features': np.random.rand(100, 5).tolist(),  # 100 samples, 5 features
            'train_targets': np.random.randint(0, 2, 100).tolist(),  # binary targets
            'features_list': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            'ids': list(range(100)),  # IDs for each sample
            'round_number': 42,
            'refresh_date': datetime.now().isoformat()
        }
    
    @pytest.mark.asyncio
    async def test_fetch_train_data_success(self, mock_train_response):
        """Test successful training data fetch."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup mock response
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = mock_train_response
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = create_test_client()
            result = await client.fetch_training_data()
            
            # Assert response structure - result is a TrainingData object
            assert hasattr(result, 'features')
            assert hasattr(result, 'targets')
            assert hasattr(result, 'feature_names')
            assert hasattr(result, 'ids')
            assert hasattr(result, 'round_number')
            assert hasattr(result, 'updated_at')
            
            # Assert data types and shapes
            assert isinstance(result.features, np.ndarray)
            assert isinstance(result.targets, np.ndarray)
            assert isinstance(result.feature_names, list)
            assert isinstance(result.ids, np.ndarray)
            assert isinstance(result.round_number, int)
            assert isinstance(result.updated_at, str)
            
            # Assert shapes match
            assert result.features.shape[0] == result.targets.shape[0]
            assert len(result.feature_names) == result.features.shape[1]
            assert len(result.ids) == result.features.shape[0]
            
            # Assert API call was made correctly
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert 'x-api-key' in call_args[1]['headers']
            assert call_args[1]['headers']['x-api-key'] == 'test-key'
    
    @pytest.mark.asyncio
    async def test_fetch_train_data_http_error(self):
        """Test handling of HTTP errors when fetching training data."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup mock error response
            mock_resp = AsyncMock()
            mock_resp.status = 404
            mock_resp.text.return_value = "Not Found"
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = create_test_client()
            
            # Should raise specific exception, not swallow it
            with pytest.raises(APIError) as exc_info:
                await client.fetch_training_data()
            
            assert "404" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_fetch_train_data_invalid_response_structure(self):
        """Test handling of invalid response structure."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup mock response with invalid structure
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {'invalid': 'structure'}
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = create_test_client()
            
            # Should raise validation error, not return invalid data
            with pytest.raises(DataValidationError) as exc_info:
                await client.fetch_training_data()
            
            assert "required field" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()


class TestTestDataFetching:
    """Test test data fetching functionality."""
    
    @pytest.fixture
    def mock_test_response(self):
        """Mock response data for test endpoint."""
        return {
            'test_features': np.random.rand(50, 5).tolist(),  # 50 test samples, 5 features
            'features_list': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            'ids': list(range(50)),  # IDs for each sample
            'submissionDeadline': '2024-01-15T23:59:59Z'
        }
    
    @pytest.mark.asyncio
    async def test_fetch_test_data_success(self, mock_test_response):
        """Test successful test data fetch."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = mock_test_response
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = create_test_client()
            result = await client.fetch_test_data()
            
            # Assert response structure - result is a TestData object
            assert hasattr(result, 'features')
            assert hasattr(result, 'feature_names')
            assert hasattr(result, 'ids')
            assert hasattr(result, 'submission_deadline')
            
            # Assert data types
            assert isinstance(result.features, np.ndarray)
            assert isinstance(result.feature_names, list)
            assert isinstance(result.ids, np.ndarray)
            assert isinstance(result.submission_deadline, str)
            
            # Assert feature list matches array dimensions
            assert len(result.feature_names) == result.features.shape[1]
            assert len(result.ids) == result.features.shape[0]
    
    @pytest.mark.asyncio
    async def test_fetch_test_data_invalid_submission_deadline(self):
        """Test handling of invalid submissionDeadline values."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Test empty submissionDeadline
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {
                'test_features': [[1, 2, 3], [4, 5, 6]],
                'features_list': ['feature_1', 'feature_2', 'feature_3'],
                'ids': [1, 2],
                'submissionDeadline': ''  # Invalid: empty string
            }
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = create_test_client()
            
            # Should raise validation error for empty deadline
            with pytest.raises(DataValidationError, match="submissionDeadline must be non-empty string"):
                await client.fetch_test_data()
                
            # Test missing submissionDeadline field
            mock_resp.json.return_value = {
                'test_features': [[1, 2, 3], [4, 5, 6]],
                'features_list': ['feature_1', 'feature_2', 'feature_3'],
                'ids': [1, 2]
                # Missing submissionDeadline field
            }
            
            with pytest.raises(DataValidationError, match="Missing required fields.*submissionDeadline"):
                await client.fetch_test_data()


class TestDataValidation:
    """Test data validation and transformation."""
    
    def test_numpy_array_conversion_validation(self):
        """Test that numpy array conversion works and validates correctly."""
        from conversion_subnet.data_api import validate_features
        
        # Valid 2D list should convert to numpy array
        valid_data = [[1, 2, 3], [4, 5, 6]]
        result = validate_features(valid_data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        
        # Test numeric data conversion
        numeric_data = [[1.5, 2.0, 3.14], [4.0, 5.5, 6.0]]
        result = validate_features(numeric_data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        expected = np.array([[1.5, 2.0, 3.14], [4.0, 5.5, 6.0]])
        np.testing.assert_array_equal(result, expected)
        
        # Invalid data should raise DataValidationError (not ValueError)
        with pytest.raises(DataValidationError):
            validate_features("not a list")
            
        with pytest.raises(DataValidationError):
            validate_features([1, 2, 3])  # 1D, not 2D
            
        with pytest.raises(DataValidationError):
            validate_features([])  # Empty list
    
    def test_features_list_validation(self):
        """Test that features list validation works correctly."""
        from conversion_subnet.data_api import validate_feature_names
        
        # Valid features list
        valid_features = ['feature_1', 'feature_2', 'feature_3']
        result = validate_feature_names(valid_features, 3)  # Should not raise
        assert result == valid_features
        
        # Invalid: wrong count
        with pytest.raises(DataValidationError, match="Expected 5 feature names"):
            validate_feature_names(valid_features, 5)
        
        # Invalid: not strings
        with pytest.raises(DataValidationError, match="must be string"):
            validate_feature_names([1, 2, 3], 3)
        
        # Invalid: not a list
        with pytest.raises(DataValidationError, match="must be a list"):
            validate_feature_names("not a list", 3)


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeouts are handled correctly."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = asyncio.TimeoutError("Request timed out")
            
            client = create_test_client(timeout=5)
            
            # Should raise APIError (which wraps the timeout error)
            with pytest.raises(APIError, match="Request timeout"):
                await client.fetch_training_data()
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test that network errors are handled correctly."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            client = create_test_client()
            
            # Should raise network error, not swallow it
            with pytest.raises(Exception, match="Network error"):
                await client.fetch_training_data()
    
    @pytest.mark.asyncio
    async def test_json_decode_error_handling(self):
        """Test that JSON decode errors are handled correctly."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.side_effect = ValueError("Invalid JSON")
            mock_resp.text.return_value = "Invalid JSON response"
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = create_test_client()
            
            # Should raise APIError (which wraps the JSON decode error)
            with pytest.raises(APIError, match="Invalid JSON"):
                await client.fetch_training_data()


class TestConfigurationFromEnvironment:
    """Test configuration loading from environment variables."""
    
    @patch.dict('os.environ', {'VOICEFORM_API_KEY': 'env-api-key', 'VOICEFORM_API_BASE_URL': 'https://env-api.com'})
    def test_client_from_env(self):
        """Test client initialization from environment variables."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        client = VoiceFormAPIClient.from_env()
        assert client.config.api_key == 'env-api-key'
        assert client.config.base_url == 'https://env-api.com'
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('conversion_subnet.data_api.core.config.load_dotenv')
    def test_client_from_env_missing_vars(self, mock_load_dotenv):
        """Test that missing environment variables raise appropriate errors."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        # Mock load_dotenv to do nothing so .env files aren't loaded
        mock_load_dotenv.return_value = None
        
        with pytest.raises(ValueError, match="VOICEFORM_API_KEY"):
            VoiceFormAPIClient.from_env()


class TestURLBuilding:
    """Test URL building with query parameters."""
    
    @pytest.mark.asyncio
    async def test_request_url_building(self):
        """Test that requests are made to correct URLs with parameters."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {
                'train_features': [[1, 2], [3, 4]],
                'train_targets': [0, 1],
                'features_list': ['f1', 'f2'],
                'ids': [1, 2],
                'round_number': 5,
                'refresh_date': '2024-01-01T00:00:00Z'
            }
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = create_test_client(base_url="https://api.example.com/v1")
            
            # Test that the correct URL is built for training data
            await client.fetch_training_data(limit=100, offset=10, round_number=1)
            
            # Verify the request was made with correct URL and parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            url = call_args[0][0]  # First positional argument is the URL
            params = call_args[1]['params']  # Parameters are in kwargs
            
            assert url == 'https://api.example.com/v1/bittensor/analytics/train-data'
            assert params['limit'] == 100
            assert params['offset'] == 10
            assert params['roundNumber'] == 1


class TestParameterValidation:
    """Test validation of query parameters."""
    
    @pytest.mark.asyncio
    async def test_fetch_train_data_with_custom_params(self):
        """Test fetching training data with custom parameters."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {
                'train_features': [[1, 2], [3, 4]],
                'train_targets': [0, 1],
                'features_list': ['f1', 'f2'],
                'ids': [1, 2],
                'round_number': 5,
                'refresh_date': '2024-01-01T00:00:00Z'
            }
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = create_test_client()
            
            # Test with custom parameters
            await client.fetch_training_data(limit=50, offset=20, round_number=5)
            
            # Verify the request was made with correct URL and parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            url = call_args[0][0]  # First positional argument is the URL
            params = call_args[1]['params']  # Parameters are in kwargs
            
            assert url == 'https://api.example.com/v1/bittensor/analytics/train-data'
            assert params['limit'] == 50
            assert params['offset'] == 20
            assert params['roundNumber'] == 5
            
    @pytest.mark.asyncio
    async def test_fetch_test_data_with_custom_params(self):
        """Test fetching test data with custom parameters."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {
                'test_features': [[1, 2], [3, 4]],
                'features_list': ['f1', 'f2'],
                'ids': [1, 2],
                'submissionDeadline': '2024-01-15T23:59:59Z'
            }
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = create_test_client()
            
            # Test with custom parameters
            await client.fetch_test_data(limit=25, offset=5)
            
            # Verify the request was made with correct URL and parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            url = call_args[0][0]  # First positional argument is the URL
            params = call_args[1]['params']  # Parameters are in kwargs
            
            assert url == 'https://api.example.com/v1/bittensor/analytics/test-data'
            assert params['limit'] == 25
            assert params['offset'] == 5
            
    def test_parameter_assertions(self):
        """Test that parameter assertions catch invalid values."""
        from conversion_subnet.data_api import VoiceFormAPIClient
        
        client = create_test_client()
        
        # Test invalid limit (should be positive)
        with pytest.raises(AssertionError, match="Limit must be positive"):
            asyncio.run(client.fetch_training_data(limit=0))
            
        # Test invalid offset (should be non-negative)  
        with pytest.raises(AssertionError, match="Offset must be non-negative"):
            asyncio.run(client.fetch_training_data(offset=-1))
            
        # Test invalid round number (should be positive)
        with pytest.raises(AssertionError, match="Round number must be positive"):
            asyncio.run(client.fetch_training_data(round_number=0)) 
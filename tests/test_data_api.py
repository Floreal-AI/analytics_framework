"""
Test module for DataAPIClient - Following TDD principles.

These tests define the expected behavior of the API client before implementation.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

# Import the specific exceptions
from conversion_subnet.data_api.models import ValidationError, APIError


class TestDataAPIClientBasic:
    """Test basic functionality and configuration of DataAPIClient."""
    
    def test_client_requires_api_key(self):
        """Test that client raises error without API key."""
        # This is testing that we don't allow empty/None API keys
        with pytest.raises(ValueError, match="API key cannot be empty"):
            from conversion_subnet.data_api import DataAPIClient
            DataAPIClient(api_key="", base_url="https://example.com")
            
        with pytest.raises(ValueError, match="API key cannot be empty"):
            from conversion_subnet.data_api import DataAPIClient
            DataAPIClient(api_key=None, base_url="https://example.com")
    
    def test_client_requires_base_url(self):
        """Test that client requires valid base URL."""
        with pytest.raises(ValueError, match="Base URL cannot be empty"):
            from conversion_subnet.data_api import DataAPIClient
            DataAPIClient(api_key="test-key", base_url="")
            
    def test_client_validates_url_format(self):
        """Test that client validates URL format."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            from conversion_subnet.data_api import DataAPIClient
            DataAPIClient(api_key="test-key", base_url="not-a-url")
    
    def test_client_initialization_success(self):
        """Test successful client initialization."""
        from conversion_subnet.data_api import DataAPIClient
        client = DataAPIClient(
            api_key="test-key", 
            base_url="https://api.example.com"
        )
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.example.com"
        assert client.timeout > 0  # Should have reasonable default timeout
        

class TestTrainDataFetching:
    """Test training data fetching functionality."""
    
    @pytest.fixture
    def mock_train_response(self):
        """Mock response data for training endpoint."""
        return {
            'train_features': np.random.rand(100, 5).tolist(),  # 100 samples, 5 features
            'train_targets': np.random.randint(0, 2, 100).tolist(),  # binary targets
            'features_list': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            'round_number': 42,
            'refresh_date': datetime.now().isoformat()
        }
    
    @pytest.mark.asyncio
    async def test_fetch_train_data_success(self, mock_train_response):
        """Test successful training data fetch."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup mock response
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = mock_train_response
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
            result = await client.fetch_train_data()
            
            # Assert response structure
            assert 'train_features' in result
            assert 'train_targets' in result
            assert 'features_list' in result
            assert 'round_number' in result
            assert 'refresh_date' in result
            
            # Assert data types and shapes
            assert isinstance(result['train_features'], np.ndarray)
            assert isinstance(result['train_targets'], np.ndarray)
            assert isinstance(result['features_list'], list)
            assert isinstance(result['round_number'], int)
            assert isinstance(result['refresh_date'], str)
            
            # Assert shapes match
            assert result['train_features'].shape[0] == result['train_targets'].shape[0]
            assert len(result['features_list']) == result['train_features'].shape[1]
            
            # Assert API call was made correctly
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert 'x-api-key' in call_args[1]['headers']
            assert call_args[1]['headers']['x-api-key'] == 'test-key'
    
    @pytest.mark.asyncio
    async def test_fetch_train_data_http_error(self):
        """Test handling of HTTP errors when fetching training data."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup mock error response
            mock_resp = AsyncMock()
            mock_resp.status = 404
            mock_resp.text.return_value = "Not Found"
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
            
            # Should raise specific exception, not swallow it
            with pytest.raises(APIError) as exc_info:
                await client.fetch_train_data()
            
            assert "404" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_fetch_train_data_invalid_response_structure(self):
        """Test handling of invalid response structure."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup mock response with invalid structure
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {'invalid': 'structure'}
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
            
            # Should raise validation error, not return invalid data
            with pytest.raises(ValidationError) as exc_info:
                await client.fetch_train_data()
            
            assert "required field" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()


class TestTestDataFetching:
    """Test test data fetching functionality."""
    
    @pytest.fixture
    def mock_test_response(self):
        """Mock response data for test endpoint."""
        return {
            'train_features': np.random.rand(50, 5).tolist(),  # 50 test samples, 5 features
            'features_list': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            'submissionDeadline': '2024-01-15T23:59:59Z'
        }
    
    @pytest.mark.asyncio
    async def test_fetch_test_data_success(self, mock_test_response):
        """Test successful test data fetch."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = mock_test_response
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
            result = await client.fetch_test_data()
            
            # Assert response structure
            assert 'train_features' in result  # Note: called 'train_features' in test response too
            assert 'features_list' in result
            assert 'submissionDeadline' in result
            
            # Assert data types
            assert isinstance(result['train_features'], np.ndarray)
            assert isinstance(result['features_list'], list)
            assert isinstance(result['submissionDeadline'], str)
            
            # Assert feature list matches array dimensions
            assert len(result['features_list']) == result['train_features'].shape[1]
    
    @pytest.mark.asyncio
    async def test_fetch_test_data_invalid_submission_deadline(self):
        """Test handling of invalid submissionDeadline values."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Test empty submissionDeadline
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {
                'train_features': [[1, 2, 3], [4, 5, 6]],
                'features_list': ['feature_1', 'feature_2', 'feature_3'],
                'submissionDeadline': ''  # Invalid: empty string
            }
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
            
            # Should raise validation error for empty deadline
            with pytest.raises(ValidationError, match="submissionDeadline must be a non-empty string"):
                await client.fetch_test_data()
                
            # Test missing submissionDeadline field
            mock_resp.json.return_value = {
                'train_features': [[1, 2, 3], [4, 5, 6]],
                'features_list': ['feature_1', 'feature_2', 'feature_3']
                # Missing submissionDeadline field
            }
            
            with pytest.raises(ValidationError, match="Missing required fields.*submissionDeadline"):
                await client.fetch_test_data()


class TestDataValidation:
    """Test data validation and transformation."""
    
    def test_numpy_array_conversion_validation(self):
        """Test that numpy array conversion works and validates correctly."""
        from conversion_subnet.data_api.client import _validate_and_convert_features
        
        # Valid 2D list should convert to numpy array
        valid_data = [[1, 2, 3], [4, 5, 6]]
        result = _validate_and_convert_features(valid_data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        
        # Test boolean string conversion
        boolean_data = [['True', 'False', '1'], ['false', 'true', '0']]
        result = _validate_and_convert_features(boolean_data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        # Check that booleans were converted to numeric values
        expected = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(result, expected)
        
        # Test mixed numeric and boolean strings
        mixed_data = [['1.5', 'True', '2.0'], ['False', '3.14', '0']]
        result = _validate_and_convert_features(mixed_data)
        assert isinstance(result, np.ndarray)
        expected_mixed = np.array([[1.5, 1.0, 2.0], [0.0, 3.14, 0.0]])
        np.testing.assert_array_equal(result, expected_mixed)
        
        # Invalid data should raise ValidationError (not ValueError)
        with pytest.raises(ValidationError):
            _validate_and_convert_features("not a list")
            
        with pytest.raises(ValidationError):
            _validate_and_convert_features([1, 2, 3])  # 1D, not 2D
            
        with pytest.raises(ValidationError):
            _validate_and_convert_features([[1, 2], [3]])  # Irregular shape
    
    def test_features_list_validation(self):
        """Test that features list validation works correctly."""
        from conversion_subnet.data_api.client import _validate_features_list
        
        # Valid features list
        valid_features = ['feature_1', 'feature_2', 'feature_3']
        _validate_features_list(valid_features, 3)  # Should not raise
        
        # Invalid: wrong count
        with pytest.raises(ValidationError, match="Features list length"):
            _validate_features_list(valid_features, 5)
        
        # Invalid: not strings
        with pytest.raises(ValidationError, match="All features must be strings"):
            _validate_features_list([1, 2, 3], 3)
        
        # Invalid: empty strings
        with pytest.raises(ValidationError, match="Feature names cannot be empty"):
            _validate_features_list(['', 'feature_2'], 2)


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeouts are handled correctly."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = asyncio.TimeoutError("Request timed out")
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com", timeout=5)
            
            # Should raise timeout error, not swallow it
            with pytest.raises(asyncio.TimeoutError):
                await client.fetch_train_data()
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test that network errors are handled correctly."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
            
            # Should raise network error, not swallow it
            with pytest.raises(Exception, match="Network error"):
                await client.fetch_train_data()
    
    @pytest.mark.asyncio
    async def test_json_decode_error_handling(self):
        """Test that JSON decode errors are handled correctly."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.side_effect = ValueError("Invalid JSON")
            mock_resp.text.return_value = "Invalid JSON response"
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
            
            # Should raise ValidationError (which wraps the JSON decode error)
            with pytest.raises(ValidationError, match="Invalid JSON"):
                await client.fetch_train_data()


class TestConfigurationFromEnvironment:
    """Test configuration loading from environment variables."""
    
    @patch.dict('os.environ', {'VOICEFORM_API_KEY': 'env-api-key', 'VOICEFORM_API_BASE_URL': 'https://env-api.com'})
    def test_client_from_env(self):
        """Test client initialization from environment variables."""
        from conversion_subnet.data_api import DataAPIClient
        
        client = DataAPIClient.from_env()
        assert client.api_key == 'env-api-key'
        assert client.base_url == 'https://env-api.com'
    
    @patch.dict('os.environ', {}, clear=True)
    def test_client_from_env_missing_vars(self):
        """Test that missing environment variables raise appropriate errors."""
        from conversion_subnet.data_api import DataAPIClient
        
        with pytest.raises(ValueError, match="VOICEFORM_API_KEY"):
            DataAPIClient.from_env()


class TestURLBuilding:
    """Test URL building with query parameters."""
    
    def test_build_url_with_params(self):
        """Test URL building with query parameters."""
        from conversion_subnet.data_api import DataAPIClient
        
        client = DataAPIClient(
            api_key="test-key",
            base_url="https://api.example.com/v1"
        )
        
        # Test train endpoint URL building
        params = {'limit': 100, 'offset': 10, 'roundNumber': 1}
        url = client._build_url_with_params('train-data', params)
        
        assert 'https://api.example.com/v1/train-data' in url
        assert 'limit=100' in url
        assert 'offset=10' in url
        assert 'roundNumber=1' in url
        
    def test_build_url_without_params(self):
        """Test URL building without query parameters."""
        from conversion_subnet.data_api import DataAPIClient
        
        client = DataAPIClient(
            api_key="test-key",
            base_url="https://api.example.com"
        )
        
        url = client._build_url_with_params('test-data', {})
        assert url == 'https://api.example.com/test-data'
        
    def test_build_url_strips_leading_slash(self):
        """Test URL building strips leading slash from endpoint."""
        from conversion_subnet.data_api import DataAPIClient
        
        client = DataAPIClient(
            api_key="test-key", 
            base_url="https://api.example.com"
        )
        
        url1 = client._build_url_with_params('train-data', {})
        url2 = client._build_url_with_params('/train-data', {})
        
        assert url1 == url2  # Should be identical after stripping


class TestParameterValidation:
    """Test validation of query parameters."""
    
    @pytest.mark.asyncio
    async def test_fetch_train_data_with_custom_params(self):
        """Test fetching training data with custom parameters."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {
                'train_features': [[1, 2], [3, 4]],
                'train_targets': [0, 1],
                'features_list': ['f1', 'f2'],
                'round_number': 5,
                'refresh_date': '2024-01-01T00:00:00Z'
            }
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
            
            # Test with custom parameters
            await client.fetch_train_data(limit=50, offset=20, round_number=5)
            
            # Verify the request was made with correct URL
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            url = call_args[0][0]  # First positional argument is the URL
            
            assert 'limit=50' in url
            assert 'offset=20' in url
            assert 'roundNumber=5' in url
            
    @pytest.mark.asyncio
    async def test_fetch_test_data_with_custom_params(self):
        """Test fetching test data with custom parameters."""
        from conversion_subnet.data_api import DataAPIClient
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {
                'train_features': [[1, 2], [3, 4]],
                'features_list': ['f1', 'f2'],
                'submissionDeadline': '2024-01-15T23:59:59Z'
            }
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
            
            # Test with custom parameters
            await client.fetch_test_data(limit=25, offset=5)
            
            # Verify the request was made with correct URL
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            url = call_args[0][0]  # First positional argument is the URL
            
            assert 'limit=25' in url
            assert 'offset=5' in url
            
    def test_parameter_assertions(self):
        """Test that parameter assertions catch invalid values."""
        from conversion_subnet.data_api import DataAPIClient
        
        client = DataAPIClient(api_key="test-key", base_url="https://api.example.com")
        
        # Test invalid limit (should be positive)
        with pytest.raises(AssertionError, match="Limit must be positive"):
            asyncio.run(client.fetch_train_data(limit=0))
            
        # Test invalid offset (should be non-negative)  
        with pytest.raises(AssertionError, match="Offset must be non-negative"):
            asyncio.run(client.fetch_train_data(offset=-1))
            
        # Test invalid round number (should be positive)
        with pytest.raises(AssertionError, match="Round number must be positive"):
            asyncio.run(client.fetch_train_data(round_number=0)) 
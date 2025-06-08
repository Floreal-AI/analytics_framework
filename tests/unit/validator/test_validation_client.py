"""
Unit tests for the validation API client.
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import patch, AsyncMock, MagicMock
from dataclasses import dataclass

from conversion_subnet.validator.validation_client import (
    ValidationAPIClient,
    ValidationResult,
    ValidationError,
    configure_default_validation_client,
    get_default_validation_client
)


class TestValidationAPIClient:
    """Test suite for ValidationAPIClient."""

    @pytest.fixture
    def client(self):
        """Create a test validation client."""
        return ValidationAPIClient(
            base_url="https://test-api.com/v1",
            api_key="test-api-key",
            timeout=30.0
        )

    def test_init_valid_params(self):
        """Test client initialization with valid parameters."""
        client = ValidationAPIClient("https://api.com", "key123", 15.0)
        assert client.base_url == "https://api.com"
        assert client.api_key == "key123"
        assert client.timeout == 15.0

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base_url."""
        client = ValidationAPIClient("https://api.com/v1/", "key123")
        assert client.base_url == "https://api.com/v1"

    def test_init_invalid_params(self):
        """Test client initialization with invalid parameters."""
        # Empty base_url
        with pytest.raises(AssertionError, match="base_url cannot be empty"):
            ValidationAPIClient("", "key123")
        
        # Empty api_key
        with pytest.raises(AssertionError, match="api_key cannot be empty"):
            ValidationAPIClient("https://api.com", "")
        
        # Invalid timeout
        with pytest.raises(AssertionError, match="timeout must be positive"):
            ValidationAPIClient("https://api.com", "key123", -1.0)

    @pytest.mark.asyncio
    async def test_get_validation_result_success(self, client):
        """Test successful validation result retrieval."""
        test_pk = "test-uuid-123"
        mock_response_data = {
            "test_pk": test_pk,
            "labels": True,
            "submissionDeadline": "2025-06-05T20:34:49.944Z"
        }

        # Mock aiohttp session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await client.get_validation_result(test_pk)
        
        # Verify result
        assert isinstance(result, ValidationResult)
        assert result.test_pk == test_pk
        assert result.labels is True
        assert result.submission_deadline == "2025-06-05T20:34:49.944Z"
        assert result.response_time > 0

        # Verify API call was made correctly
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert "test_pk=test-uuid-123" in str(call_args)
        assert "x-api-key" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_validation_result_false_labels(self, client):
        """Test validation result with labels=false."""
        test_pk = "test-uuid-456"
        mock_response_data = {
            "test_pk": test_pk,
            "labels": False,
            "submissionDeadline": "2025-06-05T20:34:49.944Z"
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await client.get_validation_result(test_pk)
        
        assert result.labels is False

    @pytest.mark.asyncio
    async def test_get_validation_result_empty_test_pk(self, client):
        """Test validation with empty test_pk."""
        with pytest.raises(ValidationError, match="test_pk cannot be empty"):
            await client.get_validation_result("")
        
        with pytest.raises(ValidationError, match="test_pk cannot be empty"):
            await client.get_validation_result("   ")

    @pytest.mark.asyncio
    async def test_get_validation_result_http_error(self, client):
        """Test handling of HTTP error responses."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(ValidationError, match="API call failed with status 404"):
                await client.get_validation_result("test-pk")

    @pytest.mark.asyncio
    async def test_get_validation_result_missing_fields(self, client):
        """Test handling of response with missing fields."""
        test_cases = [
            {"labels": True, "submissionDeadline": "2025-06-05T20:34:49.944Z"},  # Missing test_pk
            {"test_pk": "test-123", "submissionDeadline": "2025-06-05T20:34:49.944Z"},  # Missing labels
            {"test_pk": "test-123", "labels": True},  # Missing submissionDeadline
        ]

        for response_data in test_cases:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=response_data)
            
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            with patch('aiohttp.ClientSession', return_value=mock_session):
                with pytest.raises(ValidationError, match="Missing required field"):
                    await client.get_validation_result("test-pk")

    @pytest.mark.asyncio
    async def test_get_validation_result_mismatched_test_pk(self, client):
        """Test handling of response with mismatched test_pk."""
        mock_response_data = {
            "test_pk": "different-pk",
            "labels": True,
            "submissionDeadline": "2025-06-05T20:34:49.944Z"
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(ValidationError, match="Response test_pk different-pk doesn't match request test-pk"):
                await client.get_validation_result("test-pk")

    @pytest.mark.asyncio
    async def test_get_validation_result_invalid_labels_format(self, client):
        """Test handling of invalid labels values."""
        invalid_labels_cases = [
            {"test_pk": "test-123", "labels": "invalid", "submissionDeadline": "2025-06-05T20:34:49.944Z"},
            {"test_pk": "test-123", "labels": 2, "submissionDeadline": "2025-06-05T20:34:49.944Z"},
            {"test_pk": "test-123", "labels": [], "submissionDeadline": "2025-06-05T20:34:49.944Z"},
        ]

        for response_data in invalid_labels_cases:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=response_data)
            
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            with patch('aiohttp.ClientSession', return_value=mock_session):
                with pytest.raises(ValidationError, match="Invalid labels value|labels must be"):
                    await client.get_validation_result("test-pk")

    @pytest.mark.asyncio
    async def test_get_validation_result_timeout(self, client):
        """Test handling of request timeout."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            mock_session.get.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(ValidationError, match="Request timed out after 30.0 seconds"):
                await client.get_validation_result("test-pk")

    @pytest.mark.asyncio
    async def test_get_validation_result_client_error(self, client):
        """Test handling of aiohttp client errors."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
            
            with pytest.raises(ValidationError, match="HTTP client error: Connection failed"):
                await client.get_validation_result("test-pk")

    @pytest.mark.asyncio
    async def test_validate_multiple_predictions_success(self, client):
        """Test successful validation of multiple predictions."""
        test_pks = ["pk1", "pk2", "pk3"]
        mock_responses = [
            {"test_pk": "pk1", "labels": True, "submissionDeadline": "2025-06-05T20:34:49.944Z"},
            {"test_pk": "pk2", "labels": False, "submissionDeadline": "2025-06-05T20:34:49.944Z"},
            {"test_pk": "pk3", "labels": True, "submissionDeadline": "2025-06-05T20:34:49.944Z"},
        ]

        async def mock_get_validation_result(test_pk):
            for response in mock_responses:
                if response["test_pk"] == test_pk:
                    return ValidationResult(
                        test_pk=response["test_pk"],
                        labels=response["labels"],
                        submission_deadline=response["submissionDeadline"],
                        response_time=0.1
                    )
            raise ValidationError("Not found")

        with patch.object(client, 'get_validation_result', side_effect=mock_get_validation_result):
            results = await client.validate_multiple_predictions(test_pks)
        
        assert len(results) == 3
        assert results[0].test_pk == "pk1"
        assert results[0].labels is True
        assert results[1].labels is False
        assert results[2].labels is True

    @pytest.mark.asyncio
    async def test_validate_multiple_predictions_partial_failure(self, client):
        """Test validation with some failures."""
        test_pks = ["pk1", "pk2", "pk3"]
        
        async def mock_get_validation_result(test_pk):
            if test_pk == "pk1":
                return ValidationResult("pk1", True, "2025-06-05T20:34:49.944Z", 0.1)
            elif test_pk == "pk2":
                raise ValidationError("Failed")
            else:  # pk3
                return ValidationResult("pk3", False, "2025-06-05T20:34:49.944Z", 0.1)

        with patch.object(client, 'get_validation_result', side_effect=mock_get_validation_result):
            results = await client.validate_multiple_predictions(test_pks)
        
        # Should get 2 successful results out of 3
        assert len(results) == 2
        assert results[0].test_pk == "pk1"
        assert results[1].test_pk == "pk3"

    @pytest.mark.asyncio
    async def test_validate_multiple_predictions_empty_list(self, client):
        """Test validation with empty test_pks list."""
        results = await client.validate_multiple_predictions([])
        assert results == []


class TestDefaultClientManagement:
    """Test suite for default client management functions."""

    def teardown_method(self):
        """Clean up after each test."""
        # Reset global state
        import conversion_subnet.validator.validation_client
        conversion_subnet.validator.validation_client._default_client = None

    def test_configure_default_validation_client(self):
        """Test configuring the default client."""
        configure_default_validation_client("https://api.com", "key123", 25.0)
        
        client = get_default_validation_client()
        assert client.base_url == "https://api.com"
        assert client.api_key == "key123"
        assert client.timeout == 25.0

    def test_get_default_validation_client_not_configured(self):
        """Test getting default client when not configured."""
        with pytest.raises(RuntimeError, match="Default validation client not configured"):
            get_default_validation_client()

    def test_reconfigure_default_client(self):
        """Test reconfiguring the default client."""
        # First configuration
        configure_default_validation_client("https://api1.com", "key1", 10.0)
        client1 = get_default_validation_client()
        assert client1.base_url == "https://api1.com"
        
        # Reconfigure
        configure_default_validation_client("https://api2.com", "key2", 20.0)
        client2 = get_default_validation_client()
        assert client2.base_url == "https://api2.com"
        assert client2.api_key == "key2"


@pytest.mark.asyncio
class TestValidationResultDataclass:
    """Test suite for ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating ValidationResult instances."""
        result = ValidationResult(
            test_pk="test-123",
            labels=True,
            submission_deadline="2025-06-05T20:34:49.944Z",
            response_time=0.5
        )
        
        assert result.test_pk == "test-123"
        assert result.labels is True
        assert result.submission_deadline == "2025-06-05T20:34:49.944Z"
        assert result.response_time == 0.5

    def test_validation_result_default_response_time(self):
        """Test ValidationResult with default response_time."""
        result = ValidationResult(
            test_pk="test-123",
            labels=False,
            submission_deadline="2025-06-05T20:34:49.944Z"
        )
        
        assert result.response_time == 0.0


class TestValidationError:
    """Test suite for ValidationError exception."""
    
    def test_validation_error_creation(self):
        """Test creating ValidationError instances."""
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"
        
        # Test that it's a proper exception
        with pytest.raises(ValidationError, match="Test error message"):
            raise error

    def test_validation_error_inheritance(self):
        """Test that ValidationError inherits from Exception."""
        error = ValidationError("Test")
        assert isinstance(error, Exception) 
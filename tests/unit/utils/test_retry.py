"""
Unit tests for the retry utility module.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, patch

from conversion_subnet.utils.retry import (
    RetryConfig, RetryStrategy, RetryableOperation,
    retry_network_operation, retry_dendrite_call
)


class TestRetryConfig:
    """Test suite for RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert config.base_timeout == 90.0
        assert config.timeout_multiplier == 1.5
        assert config.max_timeout == 300.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert len(config.retryable_exceptions) > 0
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            base_timeout=120.0,
            strategy=RetryStrategy.LINEAR_BACKOFF
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.base_timeout == 120.0
        assert config.strategy == RetryStrategy.LINEAR_BACKOFF
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        RetryConfig(max_attempts=1, base_delay=1.0, max_delay=30.0, 
                   base_timeout=60.0, timeout_multiplier=1.0, max_timeout=60.0)
        
        # Invalid configs should raise AssertionError
        with pytest.raises(AssertionError):
            RetryConfig(max_attempts=0)  # max_attempts must be >= 1
        
        with pytest.raises(AssertionError):
            RetryConfig(base_delay=0)  # base_delay must be positive
        
        with pytest.raises(AssertionError):
            RetryConfig(max_delay=1.0, base_delay=2.0)  # max_delay must be >= base_delay
        
        with pytest.raises(AssertionError):
            RetryConfig(base_timeout=0)  # base_timeout must be positive
        
        with pytest.raises(AssertionError):
            RetryConfig(timeout_multiplier=0.5)  # timeout_multiplier must be >= 1.0
        
        with pytest.raises(AssertionError):
            RetryConfig(max_timeout=30.0, base_timeout=60.0)  # max_timeout must be >= base_timeout


class TestRetryableOperation:
    """Test suite for RetryableOperation."""

    def test_delay_calculation_exponential(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=10.0,
            jitter=False
        )
        operation = RetryableOperation(config)
        
        # Test exponential growth
        assert operation._calculate_delay(1) == 1.0  # 1.0 * 2^0
        assert operation._calculate_delay(2) == 2.0  # 1.0 * 2^1
        assert operation._calculate_delay(3) == 4.0  # 1.0 * 2^2
        assert operation._calculate_delay(4) == 8.0  # 1.0 * 2^3
        
        # Test max delay cap
        assert operation._calculate_delay(5) == 10.0  # Capped at max_delay
    
    def test_delay_calculation_linear(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            base_delay=2.0,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            max_delay=20.0,
            jitter=False
        )
        operation = RetryableOperation(config)
        
        # Test linear growth
        assert operation._calculate_delay(1) == 2.0  # 2.0 * 1
        assert operation._calculate_delay(2) == 4.0  # 2.0 * 2
        assert operation._calculate_delay(3) == 6.0  # 2.0 * 3
        
        # Test max delay cap
        assert operation._calculate_delay(15) == 20.0  # Capped at max_delay
    
    def test_delay_calculation_fixed(self):
        """Test fixed delay calculation."""
        config = RetryConfig(
            base_delay=3.0,
            strategy=RetryStrategy.FIXED_DELAY,
            jitter=False
        )
        operation = RetryableOperation(config)
        
        # Test fixed delay
        assert operation._calculate_delay(1) == 3.0
        assert operation._calculate_delay(2) == 3.0
        assert operation._calculate_delay(10) == 3.0
    
    def test_timeout_calculation(self):
        """Test timeout calculation with multiplier."""
        config = RetryConfig(
            base_timeout=60.0,
            timeout_multiplier=1.5,
            max_timeout=200.0
        )
        operation = RetryableOperation(config)
        
        # Test timeout growth
        assert operation._calculate_timeout(1) == 60.0  # 60.0 * 1.5^0
        assert operation._calculate_timeout(2) == 90.0  # 60.0 * 1.5^1
        assert operation._calculate_timeout(3) == 135.0  # 60.0 * 1.5^2
        
        # Test max timeout cap
        assert operation._calculate_timeout(10) == 200.0  # Capped at max_timeout
    
    def test_retryable_exception_check(self):
        """Test retryable exception checking."""
        config = RetryConfig(
            retryable_exceptions=[asyncio.TimeoutError, ConnectionError]
        )
        operation = RetryableOperation(config)
        
        # Test retryable exceptions
        assert operation._is_retryable_exception(asyncio.TimeoutError())
        assert operation._is_retryable_exception(ConnectionError())
        
        # Test non-retryable exceptions
        assert not operation._is_retryable_exception(ValueError())
        assert not operation._is_retryable_exception(KeyError())

    @pytest.mark.asyncio
    async def test_successful_operation_first_attempt(self):
        """Test successful operation on first attempt."""
        config = RetryConfig(max_attempts=3)
        operation = RetryableOperation(config)
        
        async def mock_operation():
            return "success"
        
        result = await operation.execute(mock_operation)
        assert result == "success"
        assert operation.attempt_count == 1
    
    @pytest.mark.asyncio
    async def test_successful_operation_after_retries(self):
        """Test successful operation after some retries."""
        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)
        operation = RetryableOperation(config)
        
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise asyncio.TimeoutError("Timeout")
            return "success"
        
        result = await operation.execute(mock_operation)
        assert result == "success"
        assert operation.attempt_count == 3
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_operation_fails_all_attempts(self):
        """Test operation that fails all attempts."""
        config = RetryConfig(max_attempts=2, base_delay=0.01, jitter=False)
        operation = RetryableOperation(config)
        
        async def mock_operation():
            raise asyncio.TimeoutError("Always fails")
        
        with pytest.raises(asyncio.TimeoutError):
            await operation.execute(mock_operation)
        
        assert operation.attempt_count == 2
    
    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test non-retryable exception is not retried."""
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=[asyncio.TimeoutError]
        )
        operation = RetryableOperation(config)
        
        async def mock_operation():
            raise ValueError("Non-retryable error")
        
        with pytest.raises(ValueError):
            await operation.execute(mock_operation)
        
        assert operation.attempt_count == 1  # Should not retry
    
    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        """Test that timeout is enforced on operations."""
        config = RetryConfig(
            max_attempts=2,
            base_timeout=0.1,  # Very short timeout
            base_delay=0.01,
            jitter=False
        )
        operation = RetryableOperation(config)
        
        async def slow_operation():
            await asyncio.sleep(0.2)  # Longer than timeout
            return "success"
        
        with pytest.raises(asyncio.TimeoutError):
            await operation.execute(slow_operation)


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    @pytest.mark.asyncio
    async def test_retry_network_operation(self):
        """Test retry_network_operation convenience function."""
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return "network_success"
        
        with patch('asyncio.sleep', return_value=None):  # Skip actual delays
            result = await retry_network_operation(
                mock_operation,
                max_attempts=3,
                base_timeout=60.0
            )
        
        assert result == "network_success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_dendrite_call(self):
        """Test retry_dendrite_call convenience function."""
        call_count = 0
        
        async def mock_dendrite_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Bittensor error")
            return "dendrite_success"
        
        with patch('asyncio.sleep', return_value=None):  # Skip actual delays
            result = await retry_dendrite_call(
                mock_dendrite_operation,
                max_attempts=4,
                base_timeout=120.0
            )
        
        assert result == "dendrite_success"
        assert call_count == 3


class TestJitterBehavior:
    """Test suite for jitter behavior."""

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delays."""
        config = RetryConfig(
            base_delay=10.0,
            jitter=True,
            strategy=RetryStrategy.FIXED_DELAY
        )
        operation = RetryableOperation(config)
        
        # Generate multiple delays and check they're different
        delays = [operation._calculate_delay(1) for _ in range(10)]
        
        # All delays should be >= base_delay
        assert all(delay >= 10.0 for delay in delays)
        
        # At least some delays should be different (jitter effect)
        # This test might rarely fail due to randomness, but very unlikely
        assert len(set(delays)) > 1, "Jitter should create different delay values"
    
    def test_no_jitter_consistent_delays(self):
        """Test that without jitter, delays are consistent."""
        config = RetryConfig(
            base_delay=10.0,
            jitter=False,
            strategy=RetryStrategy.FIXED_DELAY
        )
        operation = RetryableOperation(config)
        
        # Generate multiple delays and check they're identical
        delays = [operation._calculate_delay(1) for _ in range(10)]
        
        # All delays should be exactly the same
        assert all(delay == 10.0 for delay in delays)
        assert len(set(delays)) == 1, "Without jitter, all delays should be identical"


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    async def test_slow_network_scenario(self):
        """Test behavior with slow network conditions."""
        config = RetryConfig(
            max_attempts=3,
            base_timeout=1.0,
            timeout_multiplier=2.0,
            base_delay=0.1,
            jitter=False
        )
        operation = RetryableOperation(config)
        
        call_count = 0
        
        async def slow_network_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(1.5)  # Exceeds first timeout (1.0s)
                return "should_not_reach"
            elif call_count == 2:
                await asyncio.sleep(1.5)  # Within second timeout (2.0s)
                return "success_after_retry"
        
        result = await operation.execute(slow_network_operation)
        assert result == "success_after_retry"
        assert call_count == 2
    
    async def test_transient_failure_recovery(self):
        """Test recovery from transient failures."""
        config = RetryConfig(
            max_attempts=4,
            base_delay=0.01,
            jitter=False
        )
        operation = RetryableOperation(config)
        
        failure_types = [
            asyncio.TimeoutError("Timeout"),
            ConnectionError("Connection failed"),
            OSError("Network unavailable")
        ]
        call_count = 0
        
        async def transient_failure_operation():
            nonlocal call_count
            if call_count < len(failure_types):
                error = failure_types[call_count]
                call_count += 1
                raise error
            call_count += 1
            return "recovered"
        
        result = await operation.execute(transient_failure_operation)
        assert result == "recovered"
        assert call_count == 4  # 3 failures + 1 success 
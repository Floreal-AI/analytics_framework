"""
Retry utility module with exponential backoff and timeout resilience.

This module provides robust retry mechanisms for network operations
with configurable timeouts, retry counts, and exponential backoff.
"""

import asyncio
import time
from typing import Any, Callable, Optional, Union, List
from dataclasses import dataclass
from enum import Enum

from conversion_subnet.utils.log import logger


class RetryStrategy(Enum):
    """Retry strategy options."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"


@dataclass
class RetryConfig:
    """Configuration for retry behavior with timeout resilience."""
    
    # Basic retry settings
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 30.0  # Maximum delay between retries
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Timeout settings
    base_timeout: float = 90.0  # Increased from 60s for slow networks
    timeout_multiplier: float = 1.5  # Increase timeout with each retry
    max_timeout: float = 300.0  # Maximum timeout (5 minutes)
    
    # Exponential backoff settings
    backoff_multiplier: float = 2.0
    jitter: bool = True  # Add random jitter to prevent thundering herd
    
    # Retry conditions
    retryable_exceptions: List[type] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.retryable_exceptions is None:
            # Default retryable exceptions for network operations
            self.retryable_exceptions = [
                asyncio.TimeoutError,
                ConnectionError,
                OSError,
                Exception,  # Catch-all for unexpected errors
            ]
        
        assert self.max_attempts >= 1, "max_attempts must be at least 1"
        assert self.base_delay > 0, "base_delay must be positive"
        assert self.max_delay >= self.base_delay, "max_delay must be >= base_delay"
        assert self.base_timeout > 0, "base_timeout must be positive"
        assert self.timeout_multiplier >= 1.0, "timeout_multiplier must be >= 1.0"
        assert self.max_timeout >= self.base_timeout, "max_timeout must be >= base_timeout"


class RetryableOperation:
    """Wrapper for retryable operations with comprehensive timeout handling."""
    
    def __init__(self, config: RetryConfig):
        """Initialize with retry configuration."""
        self.config = config
        self.attempt_count = 0
        self.total_time = 0.0
        self.last_exception = None
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        else:  # FIXED_DELAY
            delay = self.config.base_delay
        
        # Cap at max_delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            import random
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay
    
    def _calculate_timeout(self, attempt: int) -> float:
        """Calculate timeout for the given attempt number."""
        timeout = self.config.base_timeout * (self.config.timeout_multiplier ** (attempt - 1))
        return min(timeout, self.config.max_timeout)
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)
    
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with retry logic and exponential backoff.
        
        Args:
            operation: Async function to execute
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: The last exception if all retries fail
        """
        start_time = time.time()
        
        for attempt in range(1, self.config.max_attempts + 1):
            self.attempt_count = attempt
            
            try:
                # Calculate timeout for this attempt
                timeout = self._calculate_timeout(attempt)
                
                logger.debug(
                    f"Attempt {attempt}/{self.config.max_attempts}: "
                    f"timeout={timeout:.1f}s, strategy={self.config.strategy.value}"
                )
                
                # Execute operation with timeout
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=timeout
                )
                
                # Success - log and return
                elapsed = time.time() - start_time
                logger.info(
                    f"Operation succeeded on attempt {attempt}/{self.config.max_attempts} "
                    f"after {elapsed:.2f}s total"
                )
                
                return result
                
            except Exception as e:
                self.last_exception = e
                elapsed = time.time() - start_time
                self.total_time = elapsed
                
                # Check if we should retry
                if attempt >= self.config.max_attempts:
                    logger.error(
                        f"Operation failed after {attempt} attempts and {elapsed:.2f}s total. "
                        f"Final error: {e}"
                    )
                    raise
                
                if not self._is_retryable_exception(e):
                    logger.error(f"Non-retryable exception: {e}")
                    raise
                
                # Calculate delay before next attempt
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt}/{self.config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        raise self.last_exception


# Convenience functions for common retry patterns

async def retry_network_operation(
    operation: Callable,
    *args,
    max_attempts: int = 3,
    base_timeout: float = 90.0,
    **kwargs
) -> Any:
    """
    Retry a network operation with exponential backoff.
    
    Args:
        operation: Async function to execute
        *args: Arguments to pass to operation
        max_attempts: Maximum number of attempts
        base_timeout: Base timeout for each attempt
        **kwargs: Additional keyword arguments for operation
        
    Returns:
        Result of the operation
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_timeout=base_timeout,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF
    )
    
    retryable = RetryableOperation(config)
    return await retryable.execute(operation, *args, **kwargs)


async def retry_dendrite_call(
    operation: Callable,
    *args,
    max_attempts: int = 4,  # More attempts for critical miner communication
    base_timeout: float = 120.0,  # Longer timeout for miner responses
    **kwargs
) -> Any:
    """
    Retry a dendrite call to miners with appropriate timeouts.
    
    Args:
        operation: Async dendrite operation
        *args: Arguments to pass to operation
        max_attempts: Maximum number of attempts
        base_timeout: Base timeout for each attempt
        **kwargs: Additional keyword arguments for operation
        
    Returns:
        Result of the operation
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_timeout=base_timeout,
        max_timeout=180.0,  # 3 minutes max (reduced from 10 minutes)
        base_delay=1.0,  # Start with 1s delay (reduced from 2s)
        max_delay=30.0,  # Cap at 30s between retries (reduced from 60s)
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        timeout_multiplier=1.2,  # More conservative timeout increase (reduced from 1.3)
        retryable_exceptions=[
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
            RuntimeError,  # Include RuntimeError for bittensor issues
            Exception,  # Catch-all
        ]
    )
    
    retryable = RetryableOperation(config)
    return await retryable.execute(operation, *args, **kwargs) 
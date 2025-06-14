"""
Validation API Client for external ground truth validation.

This module provides a robust client to call the external validation API endpoint
to get ground truth labels for miner predictions. Includes retry mechanisms,
circuit breaker pattern, and comprehensive monitoring.
"""

import aiohttp
import asyncio
import time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import warnings
from urllib.parse import urljoin
from enum import Enum
import random

from conversion_subnet.utils.log import logger

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)  # Add ±25% jitter
        return delay

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0  # seconds to wait before moving from OPEN to HALF_OPEN
    
@dataclass
class ValidationMetrics:
    """Metrics for validation API calls."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    client_error_requests: int = 0
    server_error_requests: int = 0
    total_response_time: float = 0.0
    circuit_breaker_trips: int = 0
    retry_attempts: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_requests / max(self.total_requests, 1)
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return self.total_response_time / max(self.successful_requests, 1)

@dataclass
class ValidationResult:
    """Result from validation API."""
    test_pk: str
    labels: bool  # True=1, False=0 for conversion_happened
    submission_deadline: str
    response_time: float = 0.0
    attempt_number: int = 1
    circuit_breaker_state: str = "closed"

class ValidationError(Exception):
    """Raised when validation API call fails."""
    def __init__(self, message: str, attempt_number: int = 1, is_retryable: bool = True):
        super().__init__(message)
        self.attempt_number = attempt_number
        self.is_retryable = is_retryable

class CircuitBreakerError(ValidationError):
    """Raised when circuit breaker is open."""
    def __init__(self, message: str):
        super().__init__(message, is_retryable=False)

class ValidationAPIClient:
    """Robust client for calling the external validation API with reliability features."""
    
    def __init__(self, 
                 base_url: str, 
                 api_key: str, 
                 timeout: float = 30.0,
                 retry_config: Optional[RetryConfig] = None,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize the validation API client with reliability features.
        
        Args:
            base_url: Base URL for the validation API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            retry_config: Configuration for retry behavior
            circuit_breaker_config: Configuration for circuit breaker
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        
        # Validate inputs
        assert self.base_url, "base_url cannot be empty"
        assert self.api_key, "api_key cannot be empty"
        assert self.timeout > 0, "timeout must be positive"
        
        # Circuit breaker state
        self._circuit_state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        
        # Metrics
        self.metrics = ValidationMetrics()
        
        logger.info(f"ValidationAPIClient initialized with base_url: {self.base_url}")
        logger.info(f"Retry config: max_attempts={self.retry_config.max_attempts}, base_delay={self.retry_config.base_delay}s")
        logger.info(f"Circuit breaker: failure_threshold={self.circuit_breaker_config.failure_threshold}, timeout={self.circuit_breaker_config.timeout}s")
    
    def _check_circuit_breaker(self) -> None:
        """Check circuit breaker state and update if necessary."""
        current_time = time.time()
        
        if self._circuit_state == CircuitBreakerState.OPEN:
            if current_time - self._last_failure_time >= self.circuit_breaker_config.timeout:
                logger.info("Circuit breaker moving from OPEN to HALF_OPEN (testing recovery)")
                self._circuit_state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
            else:
                remaining = self.circuit_breaker_config.timeout - (current_time - self._last_failure_time)
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. Service unavailable for {remaining:.1f} more seconds. "
                    f"Failed {self._failure_count} times, threshold is {self.circuit_breaker_config.failure_threshold}"
                )
        
        elif self._circuit_state == CircuitBreakerState.HALF_OPEN:
            # In half-open state, allow requests but monitor closely
            pass
    
    def _record_success(self) -> None:
        """Record a successful request."""
        self._failure_count = 0
        
        if self._circuit_state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.circuit_breaker_config.success_threshold:
                logger.info(f"Circuit breaker moving from HALF_OPEN to CLOSED after {self._success_count} successes")
                self._circuit_state = CircuitBreakerState.CLOSED
                self._success_count = 0
    
    def _record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._circuit_state == CircuitBreakerState.CLOSED:
            if self._failure_count >= self.circuit_breaker_config.failure_threshold:
                logger.error(f"Circuit breaker TRIPPING: {self._failure_count} consecutive failures (threshold: {self.circuit_breaker_config.failure_threshold})")
                self._circuit_state = CircuitBreakerState.OPEN
                self.metrics.circuit_breaker_trips += 1
        
        elif self._circuit_state == CircuitBreakerState.HALF_OPEN:
            logger.warning("Circuit breaker moving from HALF_OPEN back to OPEN due to failure")
            self._circuit_state = CircuitBreakerState.OPEN
    
    async def get_validation_result(self, test_pk: str) -> ValidationResult:
        """
        Get validation result for a specific test_pk with retry and circuit breaker.
        
        Args:
            test_pk: Test primary key to validate
            
        Returns:
            ValidationResult: The validation result
            
        Raises:
            ValidationError: If API call fails after all retries
            CircuitBreakerError: If circuit breaker is open
        """
        if not test_pk or not test_pk.strip():
            raise ValidationError("test_pk cannot be empty", is_retryable=False)
        
        # Check circuit breaker
        self._check_circuit_breaker()
        
        last_error = None
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                self.metrics.total_requests += 1
                self.metrics.retry_attempts += (attempt - 1)
                
                logger.debug(f"Validation attempt {attempt}/{self.retry_config.max_attempts} for test_pk {test_pk}")
                
                result = await self._make_api_request(test_pk, attempt)
                
                # Success - record and return
                self._record_success()
                self.metrics.successful_requests += 1
                self.metrics.total_response_time += result.response_time
                
                result.attempt_number = attempt
                result.circuit_breaker_state = self._circuit_state.value
                
                if attempt > 1:
                    logger.info(f"Validation succeeded on attempt {attempt} for test_pk {test_pk}")
                
                return result
                
            except ValidationError as e:
                last_error = e
                last_error.attempt_number = attempt
                
                # Record failure
                self._record_failure()
                self.metrics.failed_requests += 1
                
                # Update specific error metrics
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    self.metrics.timeout_requests += 1
                elif "client error" in str(e).lower() or "connection" in str(e).lower():
                    self.metrics.client_error_requests += 1
                elif "status 5" in str(e):  # 5xx server errors
                    self.metrics.server_error_requests += 1
                
                # Check if we should retry
                if not e.is_retryable or attempt >= self.retry_config.max_attempts:
                    logger.error(f"Validation failed permanently for test_pk {test_pk} after {attempt} attempts: {e}")
                    raise e
                
                # Calculate delay for next attempt
                delay = self.retry_config.get_delay(attempt - 1)
                logger.warning(f"Validation attempt {attempt} failed for test_pk {test_pk}: {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        # This should never be reached due to the logic above, but just in case
        raise last_error or ValidationError(f"All {self.retry_config.max_attempts} attempts failed")
    
    async def _make_api_request(self, test_pk: str, attempt: int) -> ValidationResult:
        """Make the actual API request (internal method)."""
        # Use urllib.parse.urljoin for proper URL construction to avoid path duplication
        endpoint = "bittensor/analytics/validation-data"
        url = urljoin(self.base_url.rstrip('/') + '/', endpoint)
        
        # Log the URL construction process for debugging
        if attempt == 1:  # Only log on first attempt to avoid spam
            logger.info(f"Validation URL Construction: base_url='{self.base_url}' + endpoint='{endpoint}' = '{url}'")
        
        params = {"test_pk": test_pk}
        headers = {
            "accept": "*/*",
            "x-api-key": self.api_key
        }
        
        start_time = time.time()
        
        try:
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                logger.debug(f"Making validation API request: {url} with test_pk={test_pk} (attempt {attempt})")
                
                async with session.get(url, params=params, headers=headers) as response:
                    response_time = time.time() - start_time
                    
                    # Log response status
                    logger.debug(f"Validation API Response: {response.status} for {url} (attempt {attempt})")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Validation API Error: HTTP {response.status} for {url}: {error_text}")
                        
                        # Determine if error is retryable
                        is_retryable = response.status >= 500 or response.status in [408, 429, 503, 504]
                        
                        raise ValidationError(
                            f"API call failed with status {response.status}: {error_text}",
                            attempt_number=attempt,
                            is_retryable=is_retryable
                        )
                    
                    data = await response.json()
                    logger.debug(f"Validation API response data: {data}")
                    
                    # Validate response structure
                    required_fields = ["test_pk", "labels", "submissionDeadline"]
                    for field in required_fields:
                        if field not in data:
                            raise ValidationError(
                                f"Missing required field: {field}",
                                attempt_number=attempt,
                                is_retryable=False
                            )
                    
                    # Validate test_pk matches
                    if data["test_pk"] != test_pk:
                        raise ValidationError(
                            f"Response test_pk {data['test_pk']} doesn't match request {test_pk}",
                            attempt_number=attempt,
                            is_retryable=False
                        )
                    
                    # Convert labels (true/false -> 1/0)
                    labels_value = data["labels"]
                    if isinstance(labels_value, bool):
                        conversion_happened = 1 if labels_value else 0
                    elif isinstance(labels_value, (int, str)):
                        # Handle string/int conversion
                        try:
                            conversion_happened = int(labels_value)
                            if conversion_happened not in [0, 1]:
                                raise ValueError(f"labels must be 0 or 1, got {conversion_happened}")
                        except (ValueError, TypeError):
                            raise ValidationError(
                                f"Invalid labels value: {labels_value}",
                                attempt_number=attempt,
                                is_retryable=False
                            )
                    else:
                        raise ValidationError(
                            f"labels must be boolean or int, got {type(labels_value)}",
                            attempt_number=attempt,
                            is_retryable=False
                        )
                    
                    logger.info(f"Validation successful for test_pk {test_pk}: labels={labels_value} (attempt {attempt})")
                    
                    return ValidationResult(
                        test_pk=data["test_pk"],
                        labels=labels_value,  # Keep original boolean for compatibility
                        submission_deadline=data["submissionDeadline"],
                        response_time=response_time
                    )
                    
        except aiohttp.ClientError as e:
            logger.error(f"Validation HTTP client error for {url}: {e}")
            raise ValidationError(f"HTTP client error: {e}", attempt_number=attempt, is_retryable=True)
        except asyncio.TimeoutError:
            logger.error(f"Validation request timeout for {url} after {self.timeout} seconds")
            raise ValidationError(f"Request timed out after {self.timeout} seconds", attempt_number=attempt, is_retryable=True)
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            logger.error(f"Unexpected validation error for {url}: {e}")
            raise ValidationError(f"Unexpected error: {e}", attempt_number=attempt, is_retryable=True)
    
    async def validate_multiple_predictions(self, test_pks: List[str]) -> Tuple[List[ValidationResult], List[Tuple[str, Exception]]]:
        """
        Validate multiple predictions concurrently with comprehensive error tracking.
        
        Args:
            test_pks: List of test primary keys to validate
            
        Returns:
            Tuple of (successful_results, failed_results_with_errors)
            
        Note:
            This method returns both successful and failed results separately
            to provide complete visibility into the validation process.
        """
        if not test_pks:
            return [], []
        
        logger.info(f"Validating {len(test_pks)} predictions concurrently")
        
        # Create tasks for concurrent execution
        tasks = [
            self.get_validation_result(test_pk) 
            for test_pk in test_pks
        ]
        
        # Execute all tasks and gather results
        successful_results = []
        failed_results = []
        
        try:
            # Use asyncio.gather with return_exceptions to handle partial failures
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, outcome in enumerate(outcomes):
                if isinstance(outcome, ValidationResult):
                    successful_results.append(outcome)
                else:
                    # Store the error with the test_pk for detailed reporting
                    failed_results.append((test_pks[i], outcome))
                    logger.error(f"Validation failed for test_pk {test_pks[i]}: {outcome}")
                    
        except Exception as e:
            logger.error(f"Unexpected error in batch validation: {e}")
            # Add all remaining test_pks as failed
            for test_pk in test_pks:
                failed_results.append((test_pk, e))
        
        logger.info(f"Batch validation completed: {len(successful_results)} succeeded, {len(failed_results)} failed")
        return successful_results, failed_results
    
    def get_health_status(self) -> Dict[str, any]:
        """Get comprehensive health status and metrics."""
        return {
            "circuit_breaker": {
                "state": self._circuit_state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "trips": self.metrics.circuit_breaker_trips
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": round(self.metrics.success_rate, 3),
                "average_response_time": round(self.metrics.average_response_time, 3),
                "timeout_requests": self.metrics.timeout_requests,
                "client_error_requests": self.metrics.client_error_requests,
                "server_error_requests": self.metrics.server_error_requests,
                "retry_attempts": self.metrics.retry_attempts
            },
            "configuration": {
                "timeout": self.timeout,
                "max_retry_attempts": self.retry_config.max_attempts,
                "circuit_breaker_failure_threshold": self.circuit_breaker_config.failure_threshold,
                "circuit_breaker_timeout": self.circuit_breaker_config.timeout
            }
        }
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker (for administrative use)."""
        logger.warning("Manually resetting circuit breaker")
        self._circuit_state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0

# Default client instance for convenience
_default_client: Optional[ValidationAPIClient] = None

def get_default_validation_client() -> ValidationAPIClient:
    """
    Get the default validation client instance.
    
    Returns:
        ValidationAPIClient: The default client
        
    Raises:
        RuntimeError: If default client is not configured
    """
    global _default_client
    if _default_client is None:
        raise RuntimeError(
            "Default validation client not configured. "
            "Call configure_default_validation_client() first."
        )
    return _default_client

def configure_default_validation_client(base_url: str, 
                                       api_key: str, 
                                       timeout: float = 30.0,
                                       retry_config: Optional[RetryConfig] = None,
                                       circuit_breaker_config: Optional[CircuitBreakerConfig] = None) -> None:
    """
    Configure the default validation client with reliability features.
    
    Args:
        base_url: Base URL for the validation API
        api_key: API key for authentication
        timeout: Request timeout in seconds
        retry_config: Configuration for retry behavior
        circuit_breaker_config: Configuration for circuit breaker
    """
    global _default_client
    _default_client = ValidationAPIClient(base_url, api_key, timeout, retry_config, circuit_breaker_config)
    logger.info("Default validation client configured with reliability features") 
"""
Validation API Client for external ground truth validation.

This module provides a client to call the external validation API endpoint
to get ground truth labels for miner predictions.
"""

import aiohttp
import asyncio
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
import warnings

from conversion_subnet.utils.log import logger

@dataclass
class ValidationResult:
    """Result from validation API."""
    test_pk: str
    labels: bool  # True=1, False=0 for conversion_happened
    submission_deadline: str
    response_time: float = 0.0

class ValidationError(Exception):
    """Raised when validation API call fails."""
    pass

class ValidationAPIClient:
    """Client for calling the external validation API."""
    
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        """
        Initialize the validation API client.
        
        Args:
            base_url: Base URL for the validation API
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Validate inputs
        assert self.base_url, "base_url cannot be empty"
        assert self.api_key, "api_key cannot be empty"
        assert self.timeout > 0, "timeout must be positive"
        
        logger.info(f"ValidationAPIClient initialized with base_url: {self.base_url}")
    
    async def get_validation_result(self, test_pk: str) -> ValidationResult:
        """
        Get validation result for a specific test_pk.
        
        Args:
            test_pk: Test primary key to validate
            
        Returns:
            ValidationResult: The validation result
            
        Raises:
            ValidationError: If API call fails or returns invalid data
        """
        if not test_pk or not test_pk.strip():
            raise ValidationError("test_pk cannot be empty")
        
        url = f"{self.base_url}/bittensor/analytics/validation-data"
        params = {"test_pk": test_pk}
        headers = {
            "accept": "*/*",
            "x-api-key": self.api_key
        }
        
        start_time = time.time()
        
        try:
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                logger.debug(f"Calling validation API: {url} with test_pk={test_pk}")
                
                async with session.get(url, params=params, headers=headers) as response:
                    response_time = time.time() - start_time
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValidationError(
                            f"API call failed with status {response.status}: {error_text}"
                        )
                    
                    data = await response.json()
                    logger.debug(f"Validation API response: {data}")
                    
                    # Validate response structure
                    required_fields = ["test_pk", "labels", "submissionDeadline"]
                    for field in required_fields:
                        if field not in data:
                            raise ValidationError(f"Missing required field: {field}")
                    
                    # Validate test_pk matches
                    if data["test_pk"] != test_pk:
                        raise ValidationError(
                            f"Response test_pk {data['test_pk']} doesn't match request {test_pk}"
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
                            raise ValidationError(f"Invalid labels value: {labels_value}")
                    else:
                        raise ValidationError(f"labels must be boolean or int, got {type(labels_value)}")
                    
                    return ValidationResult(
                        test_pk=data["test_pk"],
                        labels=labels_value,  # Keep original boolean for compatibility
                        submission_deadline=data["submissionDeadline"],
                        response_time=response_time
                    )
                    
        except aiohttp.ClientError as e:
            raise ValidationError(f"HTTP client error: {e}")
        except asyncio.TimeoutError:
            raise ValidationError(f"Request timed out after {self.timeout} seconds")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Unexpected error: {e}")
    
    async def validate_multiple_predictions(self, test_pks: List[str]) -> List[ValidationResult]:
        """
        Validate multiple predictions concurrently.
        
        Args:
            test_pks: List of test primary keys to validate
            
        Returns:
            List[ValidationResult]: Results for each test_pk
            
        Note:
            Failed validations will be logged but not raise exceptions.
            Check the returned list length vs input length to detect failures.
        """
        if not test_pks:
            return []
        
        logger.info(f"Validating {len(test_pks)} predictions concurrently")
        
        # Create tasks for concurrent execution
        tasks = [
            self.get_validation_result(test_pk) 
            for test_pk in test_pks
        ]
        
        # Execute all tasks and gather results
        results = []
        try:
            # Use asyncio.gather with return_exceptions to handle partial failures
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, outcome in enumerate(outcomes):
                if isinstance(outcome, ValidationResult):
                    results.append(outcome)
                else:
                    # Log the error but continue with other validations
                    logger.error(f"Validation failed for test_pk {test_pks[i]}: {outcome}")
                    
        except Exception as e:
            logger.error(f"Unexpected error in batch validation: {e}")
        
        logger.info(f"Successfully validated {len(results)}/{len(test_pks)} predictions")
        return results

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

def configure_default_validation_client(base_url: str, api_key: str, timeout: float = 30.0) -> None:
    """
    Configure the default validation client.
    
    Args:
        base_url: Base URL for the validation API
        api_key: API key for authentication
        timeout: Request timeout in seconds
    """
    global _default_client
    _default_client = ValidationAPIClient(base_url, api_key, timeout)
    logger.info("Default validation client configured") 
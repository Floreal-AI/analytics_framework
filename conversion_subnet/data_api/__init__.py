"""
Organized Data API Client

A clean, organized async HTTP client for fetching ML training/test data.

Main classes:
- VoiceFormAPIClient: Async HTTP client for data fetching
- VoiceFormAPIConfig: Configuration management with dotenv support
- TrainingData/TestData: Validated data models
- APIError/DataValidationError: Exception types

Quick start:
```python
from conversion_subnet.data_api import VoiceFormAPIClient

# Load from environment/.env file
client = VoiceFormAPIClient.from_env()

# Fetch training data
training = await client.fetch_training_data(limit=100)

# Fetch test data  
test = await client.fetch_test_data(limit=100)

# Fetch both concurrently
training, test = await client.fetch_both()

# Always clean up
await client.close()
```

For more examples, see: conversion_subnet.data_api.examples.example
For tests, run: python -m pytest conversion_subnet/data_api/unittests/
"""

# Main API
from .core import VoiceFormAPIClient, VoiceFormAPIConfig

# Data models
from .core import TrainingData, TestData

# Exceptions  
from .core import APIError, DataValidationError

# Validators (if needed for advanced usage)
from .core import validate_features, validate_targets, validate_feature_names

__all__ = [
    # Main API
    'VoiceFormAPIClient',
    'VoiceFormAPIConfig',
    
    # Data models
    'TrainingData', 
    'TestData',
    
    # Exceptions
    'APIError',
    'DataValidationError',
    
    # Validators
    'validate_features',
    'validate_targets',
    'validate_feature_names'
] 
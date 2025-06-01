"""
Core functionality for simplified data API client.

Contains the essential components:
- VoiceFormAPIClient: Main async HTTP client
- VoiceFormAPIConfig: Configuration management  
- TrainingData/TestData: Data models
- Validation utilities
"""

from .client import VoiceFormAPIClient
from .config import VoiceFormAPIConfig
from .models import TrainingData, TestData, APIError, DataValidationError
from .validators import validate_features, validate_targets, validate_feature_names

__all__ = [
    'VoiceFormAPIClient',
    'VoiceFormAPIConfig', 
    'TrainingData',
    'TestData',
    'APIError',
    'DataValidationError',
    'validate_features',
    'validate_targets', 
    'validate_feature_names'
] 
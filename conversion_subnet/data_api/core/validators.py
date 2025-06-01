"""
Data validation utilities.

Following principles:
- Single responsibility
- No fallback implementations 
- Raise errors instead of hiding them
- DRY (Don't Repeat Yourself)
"""

from typing import Any, List
import numpy as np
from .models import DataValidationError


def convert_to_numeric_array(data: Any, name: str) -> np.ndarray:
    """
    Convert API data to numeric numpy array.
    
    Args:
        data: Raw data from API
        name: Field name for error messages
        
    Returns:
        np.ndarray: Validated numeric array
        
    Raises:
        DataValidationError: If conversion fails or data is invalid
    """
    if not isinstance(data, list):
        raise DataValidationError(f"{name} must be a list, got {type(data)}")
    
    if not data:
        raise DataValidationError(f"{name} cannot be empty")
    
    try:
        # Convert to numpy array directly - let numpy handle the conversion
        array = np.array(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise DataValidationError(f"Failed to convert {name} to numeric array: {e}")
    
    # Validate no NaN or infinite values
    if np.any(np.isnan(array)):
        raise DataValidationError(f"{name} contains NaN values")
    
    if np.any(np.isinf(array)):
        raise DataValidationError(f"{name} contains infinite values")
    
    return array


def validate_features(raw_features: Any) -> np.ndarray:
    """
    Validate and convert features to 2D numpy array.
    
    Args:
        raw_features: Raw features data from API
        
    Returns:
        np.ndarray: 2D features array
        
    Raises:
        DataValidationError: If validation fails
    """
    features = convert_to_numeric_array(raw_features, "features")
    
    if features.ndim != 2:
        raise DataValidationError(f"Features must be 2D array, got {features.ndim}D")
    
    return features


def validate_targets(raw_targets: Any) -> np.ndarray:
    """
    Validate and convert targets to 1D numpy array.
    
    Args:
        raw_targets: Raw targets data from API
        
    Returns:
        np.ndarray: 1D targets array
        
    Raises:
        DataValidationError: If validation fails
    """
    targets = convert_to_numeric_array(raw_targets, "targets")
    
    # Handle 2D single-column case by flattening
    if targets.ndim == 2:
        if targets.shape[1] != 1:
            raise DataValidationError(
                f"2D targets must have exactly 1 column, got {targets.shape[1]}"
            )
        targets = targets.flatten()
    elif targets.ndim != 1:
        raise DataValidationError(f"Targets must be 1D array, got {targets.ndim}D")
    
    return targets


def validate_feature_names(raw_names: Any, expected_count: int) -> List[str]:
    """
    Validate feature names list.
    
    Args:
        raw_names: Raw feature names from API
        expected_count: Expected number of features
        
    Returns:
        List[str]: Validated feature names
        
    Raises:
        DataValidationError: If validation fails
    """
    if not isinstance(raw_names, list):
        raise DataValidationError(f"Feature names must be a list, got {type(raw_names)}")
    
    if len(raw_names) != expected_count:
        raise DataValidationError(
            f"Expected {expected_count} feature names, got {len(raw_names)}"
        )
    
    # Ensure all names are strings
    for i, name in enumerate(raw_names):
        if not isinstance(name, str):
            raise DataValidationError(f"Feature name {i} must be string, got {type(name)}")
    
    return raw_names 
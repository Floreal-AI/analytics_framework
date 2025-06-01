"""
Simplified data models for API responses.

Following principles:
- Clear, unambiguous naming
- Type safety
- No hidden complexity
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class TrainingData:
    """Training dataset with features and targets."""
    features: np.ndarray          # 2D array: [samples, features]
    targets: np.ndarray           # 1D array: [samples]
    feature_names: List[str]      # List of feature column names
    round_number: int             # Current training round
    updated_at: str              # ISO timestamp of data refresh
    
    def __post_init__(self):
        """Validate data integrity with assertions."""
        # Shape validations
        assert self.features.ndim == 2, f"Features must be 2D, got {self.features.ndim}D"
        assert self.targets.ndim == 1, f"Targets must be 1D, got {self.targets.ndim}D"
        
        # Size consistency
        n_samples, n_features = self.features.shape
        assert len(self.targets) == n_samples, \
            f"Feature/target count mismatch: {n_samples} vs {len(self.targets)}"
        assert len(self.feature_names) == n_features, \
            f"Feature name count mismatch: {n_features} vs {len(self.feature_names)}"
        
        # Data quality
        assert not np.any(np.isnan(self.features)), "Features contain NaN values"
        assert not np.any(np.isnan(self.targets)), "Targets contain NaN values"
        assert not np.any(np.isinf(self.features)), "Features contain infinite values"
        assert not np.any(np.isinf(self.targets)), "Targets contain infinite values"
        
        # Metadata validation
        assert self.round_number > 0, f"Round number must be positive, got {self.round_number}"
        assert self.updated_at.strip(), "Updated timestamp cannot be empty"


@dataclass(frozen=True)
class TestData:
    """Test dataset for prediction submission."""
    features: np.ndarray          # 2D array: [samples, features]
    feature_names: List[str]      # List of feature column names (must match training)
    submission_deadline: str      # ISO timestamp for submission deadline
    
    def __post_init__(self):
        """Validate data integrity with assertions."""
        # Shape validations
        assert self.features.ndim == 2, f"Features must be 2D, got {self.features.ndim}D"
        
        # Size consistency
        n_samples, n_features = self.features.shape
        assert len(self.feature_names) == n_features, \
            f"Feature name count mismatch: {n_features} vs {len(self.feature_names)}"
        
        # Data quality
        assert not np.any(np.isnan(self.features)), "Features contain NaN values"
        assert not np.any(np.isinf(self.features)), "Features contain infinite values"
        
        # Metadata validation
        assert self.submission_deadline.strip(), "Submission deadline cannot be empty"


class APIError(Exception):
    """API communication error - no fallbacks."""
    
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class DataValidationError(Exception):
    """Data validation error - no fallbacks."""
    pass 
"""
Unit tests for the feature_validation module.
"""

import pytest
from typing import Dict, Any

from conversion_subnet.utils.feature_validation import (
    validate_features, find_missing_or_invalid_features, get_numeric_features
)
from conversion_subnet.constants import REQUIRED_FEATURES


class TestFeatureValidation:
    """Test suite for feature validation functions."""

    def test_validate_features_valid(self, sample_features):
        """Test that validate_features passes with valid features."""
        # Valid features should not raise an exception
        result = validate_features(sample_features)
        assert result == sample_features

    def test_validate_features_missing(self, sample_features):
        """Test that validate_features raises an exception when required features are missing."""
        # Create a copy with a required feature removed
        invalid_features = sample_features.copy()
        del invalid_features['session_id']
        
        with pytest.raises(ValueError) as excinfo:
            validate_features(invalid_features)
        
        assert "Missing required features" in str(excinfo.value)
        assert "session_id" in str(excinfo.value)

    def test_validate_features_wrong_type(self, sample_features):
        """Test that validate_features raises an exception when features have wrong types."""
        # Create a copy with an invalid type
        invalid_features = sample_features.copy()
        invalid_features['has_target_entity'] = "not_an_int"
        
        with pytest.raises(ValueError) as excinfo:
            validate_features(invalid_features)
        
        assert "Feature type validation errors" in str(excinfo.value)
        assert "has_target_entity" in str(excinfo.value)

    def test_find_missing_or_invalid_features(self, sample_features):
        """Test find_missing_or_invalid_features function."""
        # Valid case - should return empty set and list
        missing, invalid = find_missing_or_invalid_features(sample_features)
        assert len(missing) == 0
        assert len(invalid) == 0
        
        # Missing feature case
        incomplete_features = {k: v for k, v in sample_features.items() 
                             if k != 'session_id'}
        missing, invalid = find_missing_or_invalid_features(incomplete_features)
        assert 'session_id' in missing
        assert len(invalid) == 0
        
        # Invalid type case
        wrong_type_features = sample_features.copy()
        wrong_type_features['has_target_entity'] = "not_an_int"
        missing, invalid = find_missing_or_invalid_features(wrong_type_features)
        assert len(missing) == 0
        assert len(invalid) == 1
        assert "has_target_entity" in invalid[0]

    def test_get_numeric_features(self, sample_features):
        """Test get_numeric_features function."""
        numeric_features = get_numeric_features(sample_features)
        
        # Should return a list of 40 float values
        assert isinstance(numeric_features, list)
        assert len(numeric_features) == 40
        assert all(isinstance(val, float) for val in numeric_features)
        
        # Test with missing features - should fill with zeros
        incomplete_features = {k: v for k, v in sample_features.items() 
                             if k not in ['hour_of_day', 'day_of_week']}
        numeric_features = get_numeric_features(incomplete_features)
        assert len(numeric_features) == 40  # Should still return 40 features
        
        # Test with non-numeric feature - should convert to 0.0
        invalid_features = sample_features.copy()
        invalid_features['hour_of_day'] = "fourteen"  # Non-numeric
        numeric_features = get_numeric_features(invalid_features)
        assert len(numeric_features) == 40 
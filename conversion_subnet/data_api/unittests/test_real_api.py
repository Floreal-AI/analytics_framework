#!/usr/bin/env python3
"""
Real API integration tests using .env configuration.

This test module performs actual API calls to validate:
- Configuration loading from .env
- Training data fetching functionality
- Data structure validation
- CSV saving functionality
- Error handling without fallbacks

Following principles:
- Test-Driven Development (TDD)
- Many assertions to catch issues early
- No fallback implementations
- Explicit error handling
- Avoid complexity

Run with: python -m pytest conversion_subnet/data_api/unittests/test_real_api.py -v
"""

import pytest
import pytest_asyncio
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
import os
import tempfile
from unittest.mock import patch

from conversion_subnet.data_api.core import (
    VoiceFormAPIClient,
    VoiceFormAPIConfig,
    TrainingData,
    TestData,
    APIError,
    DataValidationError
)


class TestRealAPIWithEnv:
    """Test real API functionality using .env configuration."""
    
    @pytest.fixture
    def env_path(self):
        """Get path to .env file."""
        return Path(__file__).parent.parent / ".env"
    
    @pytest.fixture
    def config_from_env(self, env_path):
        """Load configuration from .env file."""
        if not env_path.exists():
            pytest.skip(f".env file not found: {env_path}")
        
        try:
            config = VoiceFormAPIConfig.from_env(str(env_path))
            return config
        except Exception as e:
            pytest.skip(f"Failed to load .env configuration: {e}")
    
    @pytest_asyncio.fixture
    async def client(self, config_from_env):
        """Create API client from .env configuration."""
        client = VoiceFormAPIClient(config_from_env)
        yield client
        await client.close()
    
    def test_env_file_exists(self, env_path):
        """Test that .env file exists and is readable."""
        # Basic file existence
        assert env_path.exists(), f".env file must exist at {env_path}"
        assert env_path.is_file(), f".env must be a file, not directory"
        assert os.access(env_path, os.R_OK), f".env file must be readable"
        
        # File is not empty
        assert env_path.stat().st_size > 0, ".env file cannot be empty"
        
        print(f"✅ .env file exists and is accessible: {env_path}")
    
    def test_config_validation(self, config_from_env):
        """Test configuration validation with many assertions."""
        # API key validation
        assert config_from_env.api_key, "API key cannot be empty"
        assert isinstance(config_from_env.api_key, str), "API key must be string"
        assert len(config_from_env.api_key) > 10, "API key seems too short"
        assert config_from_env.api_key.strip() == config_from_env.api_key, "API key cannot have whitespace"
        
        # Base URL validation  
        assert config_from_env.base_url, "Base URL cannot be empty"
        assert isinstance(config_from_env.base_url, str), "Base URL must be string"
        assert config_from_env.base_url.startswith(('http://', 'https://')), "Base URL must be valid HTTP(S)"
        assert not config_from_env.base_url.endswith('/'), "Base URL should not end with slash"
        
        # Timeout validation
        assert config_from_env.timeout_seconds > 0, "Timeout must be positive"
        assert isinstance(config_from_env.timeout_seconds, int), "Timeout must be integer"
        assert config_from_env.timeout_seconds <= 300, "Timeout should be reasonable (<= 5 minutes)"
        
        print(f"✅ Configuration validation passed")
        print(f"   API key length: {len(config_from_env.api_key)}")
        print(f"   Base URL: {config_from_env.base_url}")
        print(f"   Timeout: {config_from_env.timeout_seconds}s")
    
    @pytest.mark.asyncio
    async def test_fetch_training_data_real_api(self, client):
        """Test fetching real training data with comprehensive validation."""
        print("\n🔄 Testing real API training data fetch...")
        
        # Test with small dataset first
        training_data = await client.fetch_training_data(
            limit=10,           # Small sample for testing
            offset=0,           # Start from beginning
            round_number=1,     # First round
            save=False         # Don't save during testing
        )
        
        # Validate return type
        assert isinstance(training_data, TrainingData), "Must return TrainingData instance"
        
        # Validate features array
        assert hasattr(training_data, 'features'), "Must have features attribute"
        assert isinstance(training_data.features, np.ndarray), "Features must be numpy array"
        assert training_data.features.ndim == 2, "Features must be 2D array"
        assert training_data.features.shape[0] > 0, "Must have at least one sample"
        assert training_data.features.shape[1] > 0, "Must have at least one feature"
        assert training_data.features.shape[0] <= 10, "Should not exceed requested limit"
        
        # Validate targets array
        assert hasattr(training_data, 'targets'), "Must have targets attribute"
        assert isinstance(training_data.targets, np.ndarray), "Targets must be numpy array"
        assert training_data.targets.ndim == 1, "Targets must be 1D array"
        assert training_data.targets.shape[0] == training_data.features.shape[0], "Targets must match features count"
        
        # Validate feature names
        assert hasattr(training_data, 'feature_names'), "Must have feature_names attribute"
        assert isinstance(training_data.feature_names, list), "Feature names must be list"
        assert len(training_data.feature_names) == training_data.features.shape[1], "Feature names must match feature count"
        assert all(isinstance(name, str) for name in training_data.feature_names), "All feature names must be strings"
        assert all(name.strip() for name in training_data.feature_names), "Feature names cannot be empty"
        
        # Validate metadata
        assert hasattr(training_data, 'round_number'), "Must have round_number attribute"
        assert isinstance(training_data.round_number, int), "Round number must be integer"
        assert training_data.round_number > 0, "Round number must be positive"
        
        assert hasattr(training_data, 'updated_at'), "Must have updated_at attribute"
        assert isinstance(training_data.updated_at, str), "Updated at must be string"
        assert training_data.updated_at.strip(), "Updated at cannot be empty"
        
        # Validate data quality
        assert not np.isnan(training_data.features).any(), "Features cannot contain NaN values"
        assert not np.isinf(training_data.features).any(), "Features cannot contain infinite values"
        assert np.isfinite(training_data.features).all(), "All features must be finite"
        
        # Validate target values
        unique_targets = np.unique(training_data.targets)
        assert len(unique_targets) > 0, "Must have at least one unique target value"
        assert len(unique_targets) <= training_data.targets.shape[0], "Cannot have more unique targets than samples"
        assert all(np.isfinite(target) for target in unique_targets), "All targets must be finite"
        
        print(f"✅ Training data validation passed")
        print(f"   Shape: {training_data.features.shape}")
        print(f"   Features: {len(training_data.feature_names)}")
        print(f"   Targets: {len(unique_targets)} unique values")
        print(f"   Round: {training_data.round_number}")
        print(f"   Updated: {training_data.updated_at}")
        
        return training_data
    
    @pytest.mark.asyncio
    async def test_fetch_test_data_real_api(self, client):
        """Test fetching real test data with comprehensive validation."""
        print("\n🔄 Testing real API test data fetch...")
        
        # Test with small dataset first
        test_data = await client.fetch_test_data(
            limit=10,           # Small sample for testing
            offset=0,           # Start from beginning
            save=False         # Don't save during testing
        )
        
        # Validate return type
        assert isinstance(test_data, TestData), "Must return TestData instance"
        
        # Validate features array
        assert hasattr(test_data, 'features'), "Must have features attribute"
        assert isinstance(test_data.features, np.ndarray), "Features must be numpy array"
        assert test_data.features.ndim == 2, "Features must be 2D array"
        assert test_data.features.shape[0] > 0, "Must have at least one sample"
        assert test_data.features.shape[1] > 0, "Must have at least one feature"
        assert test_data.features.shape[0] <= 10, "Should not exceed requested limit"
        
        # Validate feature names
        assert hasattr(test_data, 'feature_names'), "Must have feature_names attribute"
        assert isinstance(test_data.feature_names, list), "Feature names must be list"
        assert len(test_data.feature_names) == test_data.features.shape[1], "Feature names must match feature count"
        assert all(isinstance(name, str) for name in test_data.feature_names), "All feature names must be strings"
        assert all(name.strip() for name in test_data.feature_names), "Feature names cannot be empty"
        
        # Validate metadata
        assert hasattr(test_data, 'submission_deadline'), "Must have submission_deadline attribute"
        assert isinstance(test_data.submission_deadline, str), "Submission deadline must be string"
        assert test_data.submission_deadline.strip(), "Submission deadline cannot be empty"
        
        # Validate data quality
        assert not np.isnan(test_data.features).any(), "Features cannot contain NaN values"
        assert not np.isinf(test_data.features).any(), "Features cannot contain infinite values"
        assert np.isfinite(test_data.features).all(), "All features must be finite"
        
        print(f"✅ Test data validation passed")
        print(f"   Shape: {test_data.features.shape}")
        print(f"   Features: {len(test_data.feature_names)}")
        print(f"   Deadline: {test_data.submission_deadline}")
        
        return test_data
    
    @pytest.mark.asyncio
    async def test_data_compatibility(self, client):
        """Test compatibility between training and test data."""
        print("\n🔄 Testing data compatibility...")
        
        # Fetch both datasets
        training_data = await client.fetch_training_data(limit=5, save=False)
        test_data = await client.fetch_test_data(limit=5, save=False)
        
        # Feature compatibility
        assert training_data.feature_names == test_data.feature_names, "Feature names must match exactly"
        assert len(training_data.feature_names) == len(test_data.feature_names), "Feature count must match"
        assert training_data.features.shape[1] == test_data.features.shape[1], "Feature dimensions must match"
        
        # Data type compatibility
        assert training_data.features.dtype == test_data.features.dtype, "Feature data types must match"
        
        # Range compatibility (basic check)
        train_min, train_max = training_data.features.min(), training_data.features.max()
        test_min, test_max = test_data.features.min(), test_data.features.max()
        
        # Features should be in similar ranges (not strict requirement but good to check)
        range_overlap_min = max(train_min, test_min)
        range_overlap_max = min(train_max, test_max)
        assert range_overlap_min <= range_overlap_max, "Feature ranges should have some overlap"
        
        print(f"✅ Data compatibility validated")
        print(f"   Feature names match: {len(training_data.feature_names)} features")
        print(f"   Training range: [{train_min:.3f}, {train_max:.3f}]")
        print(f"   Test range: [{test_min:.3f}, {test_max:.3f}]")
    
    @pytest.mark.asyncio
    async def test_csv_saving_functionality(self, client):
        """Test CSV saving functionality with real data."""
        print("\n🔄 Testing CSV saving functionality...")
        
        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Fetch training data with CSV saving
                training_data = await client.fetch_training_data(
                    limit=5, 
                    round_number=1,
                    save=True  # Enable CSV saving
                )
                
                # Check training CSV file was created
                train_csv_path = Path("data/train_data.csv")
                assert train_csv_path.exists(), "Training CSV file must be created"
                assert train_csv_path.is_file(), "Training CSV must be a file"
                assert train_csv_path.stat().st_size > 0, "Training CSV cannot be empty"
                
                # Validate training CSV content
                train_df = pd.read_csv(train_csv_path)
                
                # CSV structure validation
                assert train_df.shape[0] == training_data.features.shape[0], "CSV rows must match data samples"
                assert train_df.shape[1] == training_data.features.shape[1] + 1, "CSV must have features + target column"
                
                # Column validation
                expected_columns = training_data.feature_names + ['target']
                assert list(train_df.columns) == expected_columns, "CSV columns must match feature names + target"
                
                # Data validation
                feature_columns = train_df.drop('target', axis=1)
                np.testing.assert_array_almost_equal(
                    feature_columns.values, 
                    training_data.features,
                    err_msg="CSV feature data must match original data"
                )
                np.testing.assert_array_equal(
                    train_df['target'].values,
                    training_data.targets,
                    err_msg="CSV target data must match original data"
                )
                
                # Fetch test data with CSV saving
                test_data = await client.fetch_test_data(
                    limit=5,
                    save=True  # Enable CSV saving
                )
                
                # Check test CSV file was created
                test_csv_path = Path("data/test_data.csv")
                assert test_csv_path.exists(), "Test CSV file must be created"
                assert test_csv_path.is_file(), "Test CSV must be a file"
                assert test_csv_path.stat().st_size > 0, "Test CSV cannot be empty"
                
                # Validate test CSV content
                test_df = pd.read_csv(test_csv_path)
                
                # CSV structure validation
                assert test_df.shape[0] == test_data.features.shape[0], "CSV rows must match data samples"
                assert test_df.shape[1] == test_data.features.shape[1], "CSV must have only feature columns"
                
                # Column validation
                assert list(test_df.columns) == test_data.feature_names, "CSV columns must match feature names"
                
                # Data validation
                np.testing.assert_array_almost_equal(
                    test_df.values,
                    test_data.features,
                    err_msg="CSV test data must match original data"
                )
                
                print(f"✅ CSV saving validation passed")
                print(f"   Training CSV: {train_csv_path} ({train_df.shape})")
                print(f"   Test CSV: {test_csv_path} ({test_df.shape})")
                
            finally:
                os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self, client):
        """Test parameter validation with assertions."""
        print("\n🔄 Testing parameter validation...")
        
        # Test invalid limit
        with pytest.raises(AssertionError, match="Limit must be positive"):
            await client.fetch_training_data(limit=0)
        
        with pytest.raises(AssertionError, match="Limit must be positive"):
            await client.fetch_training_data(limit=-1)
        
        # Test invalid offset
        with pytest.raises(AssertionError, match="Offset must be non-negative"):
            await client.fetch_training_data(offset=-1)
        
        # Test invalid round number
        with pytest.raises(AssertionError, match="Round number must be positive"):
            await client.fetch_training_data(round_number=0)
        
        with pytest.raises(AssertionError, match="Round number must be positive"):
            await client.fetch_training_data(round_number=-1)
        
        print(f"✅ Parameter validation works correctly")
    
    @pytest.mark.asyncio
    async def test_error_handling_no_fallbacks(self, config_from_env):
        """Test error handling without fallbacks."""
        print("\n🔄 Testing error handling...")
        
        # Test with invalid API key (should fail immediately, no fallbacks)
        invalid_config = VoiceFormAPIConfig(
            api_key="invalid-key-12345",
            base_url=config_from_env.base_url,
            timeout_seconds=config_from_env.timeout_seconds
        )
        
        client = VoiceFormAPIClient(invalid_config)
        
        try:
            # This should raise APIError, not fall back to anything
            with pytest.raises(APIError):
                await client.fetch_training_data(limit=1)
            
            print(f"✅ Error handling works - no fallbacks used")
            
        finally:
            await client.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_fetching(self, client):
        """Test concurrent data fetching."""
        print("\n🔄 Testing concurrent fetching...")
        
        # Test fetch_both method
        training_data, test_data = await client.fetch_both(
            train_limit=5,
            train_offset=0,
            round_number=1,
            test_limit=5,
            test_offset=0,
            save_training=False,
            save_test=False
        )
        
        # Validate both datasets
        assert isinstance(training_data, TrainingData), "Training data must be TrainingData instance"
        assert isinstance(test_data, TestData), "Test data must be TestData instance"
        
        # Validate compatibility
        assert training_data.feature_names == test_data.feature_names, "Feature names must match"
        assert training_data.features.shape[1] == test_data.features.shape[1], "Feature dimensions must match"
        
        print(f"✅ Concurrent fetching works correctly")
        print(f"   Training: {training_data.features.shape}")
        print(f"   Test: {test_data.features.shape}")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short']) 
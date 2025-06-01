"""
Unit tests for the dataset utility module.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from conversion_subnet.utils.dataset import StructuredDataset


class TestStructuredDataset:
    """Test suite for StructuredDataset class."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        data = {
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'feature3': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_init_with_valid_data(self, temp_csv_file):
        """Test dataset initialization with valid CSV file."""
        dataset = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target',
            test_size=0.3
        )
        
        # Verify initialization
        assert dataset.data_path == temp_csv_file
        assert dataset.target_column == 'target'
        assert dataset.test_size == 0.3
        assert dataset.scaler is not None
        
        # Verify data loading
        assert hasattr(dataset, 'df')
        assert hasattr(dataset, 'X')
        assert hasattr(dataset, 'y')
        assert hasattr(dataset, 'X_train')
        assert hasattr(dataset, 'X_test')
        assert hasattr(dataset, 'y_train')
        assert hasattr(dataset, 'y_test')
        
        # Verify data shapes
        assert len(dataset.feature_columns) == 3  # 3 features
        assert dataset.X.shape[1] == 3  # 3 features
        assert dataset.X.shape[0] == 10  # 10 samples
        assert len(dataset.y) == 10  # 10 targets
        
        # Verify train/test split (30% test = 3 samples, 70% train = 7 samples)
        assert len(dataset.X_train) == 7
        assert len(dataset.X_test) == 3
        assert len(dataset.y_train) == 7
        assert len(dataset.y_test) == 3

    def test_init_with_default_test_size(self, temp_csv_file):
        """Test dataset initialization with default test size."""
        dataset = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        assert dataset.test_size == 0.2
        # With 10 samples and 20% test size: 8 train, 2 test
        assert len(dataset.X_train) == 8
        assert len(dataset.X_test) == 2

    def test_feature_columns_extraction(self, temp_csv_file):
        """Test that feature columns are correctly extracted."""
        dataset = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        expected_features = ['feature1', 'feature2', 'feature3']
        assert dataset.feature_columns == expected_features
        assert 'target' not in dataset.feature_columns

    def test_data_scaling(self, temp_csv_file):
        """Test that features are properly scaled."""
        dataset = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        # Check that scaler was fitted
        assert hasattr(dataset.scaler, 'mean_')
        assert hasattr(dataset.scaler, 'scale_')
        
        # Verify that training data has been scaled (mean ≈ 0, std ≈ 1)
        train_means = np.mean(dataset.X_train, axis=0)
        train_stds = np.std(dataset.X_train, axis=0)
        
        # Should be close to 0 mean and 1 std (with some tolerance for small dataset)
        assert np.allclose(train_means, 0, atol=1e-10)
        assert np.allclose(train_stds, 1, atol=1e-10)

    def test_get_random_sample_train(self, temp_csv_file):
        """Test getting random samples from training set."""
        dataset = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        # Test multiple samples to ensure randomness works
        for _ in range(10):
            features, label = dataset.get_random_sample(split='train')
            
            # Verify return types and shapes
            assert isinstance(features, list)
            assert isinstance(label, (int, np.integer))
            assert len(features) == 3  # 3 features
            assert label in [0, 1]  # binary classification

    def test_get_random_sample_test(self, temp_csv_file):
        """Test getting random samples from test set."""
        dataset = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        for _ in range(10):
            features, label = dataset.get_random_sample(split='test')
            
            assert isinstance(features, list)
            assert isinstance(label, (int, np.integer))
            assert len(features) == 3
            assert label in [0, 1]

    def test_get_random_sample_default_split(self, temp_csv_file):
        """Test getting random samples with default split (test)."""
        dataset = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        # Default should be 'test'
        features, label = dataset.get_random_sample()
        
        assert isinstance(features, list)
        assert isinstance(label, (int, np.integer))
        assert len(features) == 3

    def test_get_feature_names(self, temp_csv_file):
        """Test getting feature names."""
        dataset = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        feature_names = dataset.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert feature_names == ['feature1', 'feature2', 'feature3']
        assert 'target' not in feature_names

    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        with pytest.raises(FileNotFoundError):
            StructuredDataset(
                data_path='/nonexistent/path/file.csv',
                target_column='target'
            )

    def test_invalid_target_column(self, temp_csv_file):
        """Test handling of invalid target column."""
        with pytest.raises(KeyError):
            StructuredDataset(
                data_path=temp_csv_file,
                target_column='nonexistent_column'
            )

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        # Create empty CSV
        empty_data = pd.DataFrame()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            empty_data.to_csv(f.name, index=False)
            
        try:
            with pytest.raises((ValueError, KeyError)):
                StructuredDataset(
                    data_path=f.name,
                    target_column='target'
                )
        finally:
            os.unlink(f.name)

    def test_single_feature_dataset(self):
        """Test dataset with only one feature column."""
        # Create single feature dataset
        data = pd.DataFrame({
            'single_feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            
        try:
            dataset = StructuredDataset(
                data_path=f.name,
                target_column='target'
            )
            
            assert len(dataset.feature_columns) == 1
            assert dataset.feature_columns == ['single_feature']
            assert dataset.X.shape[1] == 1
            
            features, label = dataset.get_random_sample()
            assert len(features) == 1
            
        finally:
            os.unlink(f.name)

    def test_different_test_sizes(self, temp_csv_file):
        """Test different test size configurations."""
        test_sizes = [0.1, 0.3, 0.5, 0.9]
        
        for test_size in test_sizes:
            dataset = StructuredDataset(
                data_path=temp_csv_file,
                target_column='target',
                test_size=test_size
            )
            
            total_samples = 10
            expected_test_size = int(total_samples * test_size)
            expected_train_size = total_samples - expected_test_size
            
            assert len(dataset.X_test) == expected_test_size
            assert len(dataset.X_train) == expected_train_size

    def test_data_integrity_after_scaling(self, temp_csv_file):
        """Test that data relationships are preserved after scaling."""
        dataset = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        # Original data should be preserved in df
        original_features = dataset.df[dataset.feature_columns].values
        assert original_features.shape == dataset.X.shape
        
        # Verify we can inverse transform scaled data
        original_train = dataset.scaler.inverse_transform(dataset.X_train)
        original_test = dataset.scaler.inverse_transform(dataset.X_test)
        
        # Check shapes are consistent
        assert original_train.shape == dataset.X_train.shape
        assert original_test.shape == dataset.X_test.shape

    def test_random_state_consistency(self, temp_csv_file):
        """Test that random state produces consistent splits."""
        dataset1 = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        dataset2 = StructuredDataset(
            data_path=temp_csv_file,
            target_column='target'
        )
        
        # Should produce identical splits due to fixed random_state=42
        np.testing.assert_array_equal(dataset1.X_train, dataset2.X_train)
        np.testing.assert_array_equal(dataset1.X_test, dataset2.X_test)
        np.testing.assert_array_equal(dataset1.y_train, dataset2.y_train)
        np.testing.assert_array_equal(dataset1.y_test, dataset2.y_test) 
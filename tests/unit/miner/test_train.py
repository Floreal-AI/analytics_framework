"""
Unit tests for the miner training module.
"""

import os
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from conversion_subnet.miner.train import (
    ConversationDataset, load_data, train_model, save_model, load_model
)


class TestConversationDataset:
    """Test suite for the ConversationDataset class."""

    def test_init(self, sample_dataset):
        """Test dataset initialization."""
        X, y = sample_dataset
        dataset = ConversationDataset(X, y)
        
        assert isinstance(dataset.features, torch.Tensor)
        assert isinstance(dataset.labels, torch.Tensor)
        assert dataset.features.shape == (10, 40)
        assert dataset.labels.shape == (10,)
        assert dataset.features.dtype == torch.float32
        assert dataset.labels.dtype == torch.float32

    def test_len(self, sample_dataset):
        """Test __len__ method."""
        X, y = sample_dataset
        dataset = ConversationDataset(X, y)
        
        assert len(dataset) == 10

    def test_getitem(self, sample_dataset):
        """Test __getitem__ method."""
        X, y = sample_dataset
        dataset = ConversationDataset(X, y)
        
        # Get first item
        features, label = dataset[0]
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert features.shape == (40,)
        assert label.shape == ()
        
        # Check values
        assert torch.allclose(features, torch.tensor(X[0], dtype=torch.float32))
        assert label.item() == y[0]


class TestDataLoading:
    """Test suite for data loading functions."""

    def test_load_data(self, tmp_path):
        """Test load_data function with a temporary CSV file."""
        # Create a sample DataFrame
        data = {
            'session_id': [f'session-{i}' for i in range(10)],
            'conversion_happened': [i % 2 for i in range(10)],
            'time_to_conversion_seconds': [60.0 if i % 2 else -1.0 for i in range(10)]
        }
        
        # Add some feature columns
        for i in range(5):
            data[f'feature_{i}'] = np.random.rand(10)
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Load data
        X_train, X_test, y_train, y_test = load_data(csv_path, test_size=0.3)
        
        # Check shapes
        assert X_train.shape[0] == 7  # 70% of 10
        assert X_test.shape[0] == 3   # 30% of 10
        assert X_train.shape[1] == 5  # 5 feature columns
        assert len(y_train) == 7
        assert len(y_test) == 3
        
        # Check that non-feature columns are dropped
        feature_cols = [f'feature_{i}' for i in range(5)]
        loaded_cols = df.drop(['conversion_happened', 'time_to_conversion_seconds', 'session_id'], 
                              axis=1).columns
        assert set(feature_cols) == set(loaded_cols)

    def test_load_data_missing_columns(self, tmp_path):
        """Test load_data function with missing required columns."""
        # Create a DataFrame without conversion_happened
        data = {'feature_1': np.random.rand(5)}
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = tmp_path / "invalid_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            load_data(csv_path)
        
        assert "CSV must contain 'conversion_happened' column" in str(excinfo.value)


class TestModelTraining:
    """Test suite for model training functions."""

    @patch('torch.utils.data.DataLoader')
    @patch('torch.optim.Adam')
    @patch('torch.nn.BCELoss')
    def test_train_model(self, mock_bce_loss, mock_adam, mock_dataloader, sample_dataset):
        """Test train_model function."""
        X_train, y_train = sample_dataset
        X_val, y_val = sample_dataset  # Reuse for validation
        
        # Create mock miner with mock model
        mock_miner = MagicMock()
        mock_model = MagicMock()
        mock_miner.model = mock_model
        
        # Mock the BCELoss class and its instance
        mock_criterion = MagicMock()
        mock_bce_loss.return_value = mock_criterion
        # Have the criterion return a mock tensor that won't need backward
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5  # Mock loss value
        mock_criterion.return_value = mock_loss
        
        # Mock DataLoader instances
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_dataloader.side_effect = [mock_train_loader, mock_val_loader]
        
        # Mock optimizer
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer
        
        # Configure mock_train_loader to yield sample data
        features = torch.randn(4, 40)
        labels = torch.randint(0, 2, (4,)).float()
        sample_batch = (features, labels)
        mock_train_loader.__iter__.return_value = [sample_batch]
        mock_train_loader.__len__.return_value = 1
        
        # Configure mock_val_loader to yield sample data
        mock_val_loader.__iter__.return_value = [sample_batch]
        mock_val_loader.__len__.return_value = 1
        
        # Configure the model output
        output_tensor = torch.rand(4)
        mock_model.return_value = output_tensor
        
        # Call train_model
        history = train_model(
            miner=mock_miner,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=1,
            batch_size=4,
            learning_rate=0.001,
            device="cpu"
        )
        
        # Check that optimizer and DataLoader were created correctly
        mock_adam.assert_called_once()
        assert mock_dataloader.call_count == 2
        
        # Check that the model was put in train mode
        mock_model.train.assert_called()
        
        # Check that the model was put in eval mode
        mock_model.eval.assert_called()
        
        # Check history structure
        assert 'train_loss' in history
        assert 'train_acc' in history
        assert 'val_loss' in history
        assert 'val_acc' in history

    @patch('torch.save')
    def test_save_model(self, mock_save, tmp_checkpoint_dir):
        """Test save_model function."""
        # Create mock miner with mock model
        mock_miner = MagicMock()
        mock_model = MagicMock()
        mock_miner.model = mock_model
        
        # Mock state_dict to return a dictionary
        mock_model.state_dict.return_value = {"layer1.weight": torch.randn(5, 5)}
        
        # Create the directory if it doesn't exist
        tmp_checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Create an empty file manually since we've mocked torch.save
        model_path = tmp_checkpoint_dir / "test_model.pt"
        model_path.touch()
        
        # Call save_model
        save_model(mock_miner, tmp_checkpoint_dir, "test_model.pt")
        
        # Check that the file exists (we created it manually)
        assert model_path.exists()
        
        # Check that torch.save was called with the correct arguments
        mock_model.state_dict.assert_called_once()
        mock_save.assert_called_once()

    @patch('torch.load')
    def test_load_model(self, mock_load, tmp_checkpoint_dir):
        """Test load_model function."""
        # Create mock miner with mock model
        mock_miner = MagicMock()
        mock_model = MagicMock()
        mock_miner.model = mock_model
        
        # Configure device property to return a string instead of a MagicMock
        mock_miner.device = "cpu"
        
        # Mock torch.load to return a state_dict
        mock_state_dict = {"layer1.weight": torch.randn(5, 5)}
        mock_load.return_value = mock_state_dict
        
        # Create a dummy file so exists() check passes
        model_path = tmp_checkpoint_dir / "test_model.pt"
        model_path.parent.mkdir(exist_ok=True, parents=True)
        model_path.touch()
        
        # Call load_model
        load_model(mock_miner, model_path)
        
        # Check that model.load_state_dict was called
        mock_model.load_state_dict.assert_called_once_with(mock_state_dict)
        mock_load.assert_called_once()
        
        # Test with non-existent file
        with pytest.raises(ValueError) as excinfo:
            load_model(mock_miner, tmp_checkpoint_dir / "nonexistent.pt")
        
        assert "does not exist" in str(excinfo.value) 
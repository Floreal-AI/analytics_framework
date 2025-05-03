"""
Model training script for the BinaryClassificationMiner.

This module provides functions to train a binary classification model
for predicting conversation outcomes. It includes:
- Dataset loading and preprocessing
- Model training and evaluation
- Checkpoint saving and loading
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from conversion_subnet.utils.log import logger
from conversion_subnet.protocol import ConversationFeatures
from conversion_subnet.utils.feature_validation import get_numeric_features
from conversion_subnet.miner.miner import BinaryClassificationMiner
from conversion_subnet.utils.configuration import config as default_config

class ConversationDataset(torch.utils.data.Dataset):
    """PyTorch dataset for conversation features and outcomes."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, n_features)
            labels (np.ndarray): Label vector of shape (n_samples,)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        return self.features[idx], self.labels[idx]

def load_data(
    csv_path: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split data from a CSV file.
    
    Args:
        csv_path (Union[str, Path]): Path to CSV file
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, X_test, y_train, y_test
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    if 'conversion_happened' not in df.columns:
        raise ValueError("CSV must contain 'conversion_happened' column")
    
    # Extract features and labels
    y = df['conversion_happened'].values
    
    # Drop non-feature columns
    X = df.drop(['conversion_happened', 'time_to_conversion_seconds', 'session_id'], 
                 errors='ignore', axis=1).values
    
    # Split data
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size_int = int(test_size * len(X))
    test_idx, train_idx = indices[:test_size_int], indices[test_size_int:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def train_model(
    miner: BinaryClassificationMiner,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoints_dir: Optional[Path] = None,
    device: str = "cpu"
) -> Dict[str, List[float]]:
    """
    Train a binary classification model.
    
    Args:
        miner (BinaryClassificationMiner): Miner with model to train
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_val (Optional[np.ndarray]): Validation features
        y_val (Optional[np.ndarray]): Validation labels
        epochs (int): Number of epochs to train for
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        checkpoints_dir (Optional[Path]): Directory to save checkpoints
        device (str): Device to train on (cpu, cuda)
        
    Returns:
        Dict[str, List[float]]: Training history
    """
    # Create datasets and data loaders
    train_dataset = ConversationDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = ConversationDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(miner.model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    device_obj = torch.device(device)
    miner.model.to(device_obj)
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training
        miner.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device_obj), labels.to(device_obj)
            
            # Forward pass
            outputs = miner.model(features).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        if val_loader is not None:
            miner.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device_obj), labels.to(device_obj)
                    
                    # Forward pass
                    outputs = miner.model(features).squeeze()
                    loss = criterion(outputs, labels)
                    
                    # Track statistics
                    val_loss += loss.item()
                    predicted = (outputs >= 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
            
            # Save checkpoint if validation loss improved
            if checkpoints_dir is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(miner, checkpoints_dir, f"model_best_epoch{epoch+1}.pt")
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
    
    # Save final model
    if checkpoints_dir is not None:
        save_model(miner, checkpoints_dir, "model_final.pt")
    
    return history

def save_model(
    miner: BinaryClassificationMiner, 
    checkpoints_dir: Path,
    filename: str = "model.pt"
) -> None:
    """
    Save model weights to disk.
    
    Args:
        miner (BinaryClassificationMiner): Miner with model to save
        checkpoints_dir (Path): Directory to save checkpoint to
        filename (str): Filename for checkpoint
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_path = checkpoints_dir / filename
    torch.save(miner.model.state_dict(), checkpoint_path)
    logger.info(f"Saved model checkpoint to {checkpoint_path}")

def load_model(
    miner: BinaryClassificationMiner, 
    checkpoint_path: Path
) -> None:
    """
    Load model weights from disk.
    
    Args:
        miner (BinaryClassificationMiner): Miner to load model into
        checkpoint_path (Path): Path to checkpoint file
    """
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint file {checkpoint_path} does not exist")
    
    miner.model.load_state_dict(torch.load(checkpoint_path, map_location=miner.device))
    logger.info(f"Loaded model checkpoint from {checkpoint_path}")

def main(config=None):
    """
    Main function to train a model from the command line.
    
    Args:
        config: Configuration object (optional)
    """
    # Use default config if none provided
    config = config or default_config
    
    # Parse arguments
    data_path = config.miner.dataset_path if hasattr(config.miner, 'dataset_path') else None
    if data_path is None:
        data_path = Path.cwd() / "data" / "conversations.csv"
        logger.info(f"No dataset path provided, using default: {data_path}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file {data_path} does not exist")
        return
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    X_train, X_test, y_train, y_test = load_data(data_path)
    logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Initialize miner
    logger.info("Initializing miner")
    miner = BinaryClassificationMiner(config)
    
    # Train model
    logger.info("Training model")
    train_model(
        miner=miner,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=config.miner.epochs,
        batch_size=config.miner.batch_size,
        learning_rate=config.miner.learning_rate,
        checkpoints_dir=config.miner.checkpoint_dir,
        device=config.miner.device
    )
    
    logger.info("Training complete")

if __name__ == "__main__":
    main() 
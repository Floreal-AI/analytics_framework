"""
Model training script for the BinaryClassificationMiner.

This module provides functions to train a binary classification model
for predicting conversation outcomes. It includes:
- Dataset loading and preprocessing
- Model training and evaluation (PyTorch and XGBoost)
- Checkpoint saving and loading
- Cross-validation and comprehensive reporting
"""

import os
import torch
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

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
    target_col = 'target' if 'target' in df.columns else 'conversion_happened'
    if target_col not in df.columns:
        raise ValueError(f"CSV must contain '{target_col}' column")
    
    # Extract features and labels
    y = df[target_col].values
    
    # Drop non-feature columns
    columns_to_drop = [target_col, 'time_to_conversion_seconds', 'session_id', 'id']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(existing_columns_to_drop, axis=1).values
    
    logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size_int = int(test_size * len(X))
    test_idx, train_idx = indices[:test_size_int], indices[test_size_int:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model_params: Optional[Dict] = None,
    early_stopping_rounds: int = 10,
    verbose: bool = True
) -> xgb.XGBClassifier:
    """
    Train an XGBoost binary classification model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_val (Optional[np.ndarray]): Validation features
        y_val (Optional[np.ndarray]): Validation labels
        model_params (Optional[Dict]): XGBoost parameters
        early_stopping_rounds (int): Early stopping rounds
        verbose (bool): Whether to print verbose output
        
    Returns:
        xgb.XGBClassifier: Trained XGBoost model
    """
    # Default XGBoost parameters for binary classification
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if model_params:
        default_params.update(model_params)
    
    # Initialize model
    model = xgb.XGBClassifier(**default_params)
    
    # Prepare evaluation set for early stopping
    eval_set = []
    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
    # Train model
    if eval_set and early_stopping_rounds > 0:
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )
    else:
        model.fit(X_train, y_train)
    
    if verbose:
        logger.info("XGBoost model training completed")
        
    return model

def evaluate_model_with_cv(
    model: Union[xgb.XGBClassifier, any],
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate model using stratified k-fold cross-validation.
    
    Args:
        model: The model to evaluate
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        cv_folds (int): Number of CV folds
        random_state (int): Random seed
        
    Returns:
        Dict[str, float]: Cross-validation metrics
    """
    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    cv_roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    cv_results = {
        'cv_accuracy_mean': cv_accuracy.mean(),
        'cv_accuracy_std': cv_accuracy.std(),
        'cv_precision_mean': cv_precision.mean(),
        'cv_precision_std': cv_precision.std(),
        'cv_recall_mean': cv_recall.mean(),
        'cv_recall_std': cv_recall.std(),
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_auc_mean': cv_roc_auc.mean(),
        'cv_roc_auc_std': cv_roc_auc.std()
    }
    
    return cv_results

def generate_model_report(
    model: Union[xgb.XGBClassifier, any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    cv_folds: int = 5
) -> Dict:
    """
    Generate comprehensive model evaluation report.
    
    Args:
        model: Trained model
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        feature_names (Optional[List[str]]): Feature names for importance
        cv_folds (int): Number of CV folds
        
    Returns:
        Dict: Comprehensive model report
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Prediction probabilities
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Training metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_pred_proba)
    }
    
    # Test metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
    }
    
    # Cross-validation metrics
    cv_metrics = evaluate_model_with_cv(model, X_train, y_train, cv_folds)
    
    # Confusion matrices
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        importance_values = model.feature_importances_
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance_values))]
        
        # Sort by importance
        importance_pairs = list(zip(feature_names, importance_values))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        feature_importance = importance_pairs
    
    # Compile report
    report = {
        'model_type': type(model).__name__,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': X_train.shape[1],
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cross_validation': cv_metrics,
        'confusion_matrix_train': train_cm.tolist(),
        'confusion_matrix_test': test_cm.tolist(),
        'feature_importance': feature_importance
    }
    
    return report

def save_xgboost_model(
    model: xgb.XGBClassifier,
    model_dir: Path,
    filename: str = "xgboost_model.pkl"
) -> None:
    """
    Save XGBoost model to disk.
    
    Args:
        model (xgb.XGBClassifier): Model to save
        model_dir (Path): Directory to save model
        filename (str): Filename for model
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = model_dir / filename
    
    # Save using joblib for better XGBoost compatibility
    joblib.dump(model, model_path)
    logger.info(f"Saved XGBoost model to {model_path}")

def load_xgboost_model(model_path: Path) -> xgb.XGBClassifier:
    """
    Load XGBoost model from disk.
    
    Args:
        model_path (Path): Path to model file
        
    Returns:
        xgb.XGBClassifier: Loaded model
    """
    if not model_path.exists():
        raise ValueError(f"Model file {model_path} does not exist")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded XGBoost model from {model_path}")
    return model

def fetch_and_save_data(config) -> Tuple[Path, Path]:
    """
    Fetch training data using the data API and save to data directory.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple[Path, Path]: Paths to train and test data files
    """
    try:
        from conversion_subnet.data_api.core.client import DataAPIClient
        
        # Initialize data API client
        client = DataAPIClient()
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Define file paths
        train_path = data_dir / "train_data.csv"
        test_path = data_dir / "test_data.csv"
        
        logger.info("Fetching training and test data...")
        
        # Fetch both datasets
        client.fetch_both(
            train_csv_filename=str(train_path),
            test_csv_filename=str(test_path)
        )
        
        logger.info(f"Data saved to {train_path} and {test_path}")
        return train_path, test_path
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        # Fall back to existing data if available
        train_path = Path("data/train_data.csv")
        test_path = Path("data/test_data.csv")
        
        if train_path.exists():
            logger.info(f"Using existing training data: {train_path}")
            if not test_path.exists():
                logger.warning(f"Test data not found: {test_path}")
            return train_path, test_path
        else:
            raise ValueError("No training data available and failed to fetch from API")

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

def train_xgboost_with_report(
    config=None,
    force_retrain: bool = True,
    use_data_api: bool = True
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train XGBoost model with comprehensive evaluation and reporting.
    
    Args:
        config: Configuration object (optional)
        force_retrain (bool): If True, always train new model. If False, load existing if available.
        use_data_api (bool): If True, fetch data using API. If False, use existing files.
        
    Returns:
        Tuple[xgb.XGBClassifier, Dict]: Trained model and evaluation report
    """
    # Use default config if none provided
    config = config or default_config
    
    # Define model directory
    model_dir = Path("conversion_subnet/model_weights")
    model_path = model_dir / "xgboost_model.pkl"
    report_path = model_dir / "model_report.json"
    
    # Check if model exists and we don't want to force retrain
    if not force_retrain and model_path.exists():
        logger.info(f"Loading existing XGBoost model from {model_path}")
        model = load_xgboost_model(model_path)
        
        # Load existing report if available
        if report_path.exists():
            import json
            with open(report_path, 'r') as f:
                report = json.load(f)
            logger.info("Loaded existing model report")
        else:
            report = {"message": "Model loaded from existing file, no evaluation report available"}
        
        return model, report
    
    # Fetch or load data
    if use_data_api:
        try:
            train_path, test_path = fetch_and_save_data(config)
        except Exception as e:
            logger.warning(f"Failed to fetch data via API: {e}")
            train_path = Path("data/train_data.csv")
            test_path = Path("data/test_data.csv")
    else:
        train_path = Path("data/train_data.csv")
        test_path = Path("data/test_data.csv")
    
    # Check if data files exist
    if not train_path.exists():
        raise ValueError(f"Training data file {train_path} does not exist")
    
    # Load training data
    logger.info(f"Loading data from {train_path}")
    X_train, X_val, y_train, y_val = load_data(train_path)
    
    # Load test data if available separately
    if test_path.exists():
        logger.info(f"Loading test data from {test_path}")
        df_test = pd.read_csv(test_path)
        target_col = 'target' if 'target' in df_test.columns else 'conversion_happened'
        
        if target_col in df_test.columns:
            y_test = df_test[target_col].values
            columns_to_drop = [target_col, 'time_to_conversion_seconds', 'session_id', 'id']
            existing_columns_to_drop = [col for col in columns_to_drop if col in df_test.columns]
            X_test = df_test.drop(existing_columns_to_drop, axis=1).values
            logger.info(f"Test data loaded: {X_test.shape[0]} samples")
        else:
            X_test, y_test = X_val, y_val
            logger.info("Using validation split as test data")
    else:
        X_test, y_test = X_val, y_val
        logger.info("Using validation split as test data")
    
    # Get feature names for importance analysis
    df = pd.read_csv(train_path)
    target_col = 'target' if 'target' in df.columns else 'conversion_happened'
    columns_to_drop = [target_col, 'time_to_conversion_seconds', 'session_id', 'id']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    feature_names = df.drop(existing_columns_to_drop, axis=1).columns.tolist()
    
    # Train XGBoost model
    logger.info("Training XGBoost model...")
    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=True
    )
    
    # Generate comprehensive report
    logger.info("Generating model evaluation report...")
    report = generate_model_report(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        cv_folds=5
    )
    
    # Save model and report
    save_xgboost_model(model, model_dir)
    
    # Save report as JSON
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Model report saved to {report_path}")
    
    # Print summary
    print_model_summary(report)
    
    return model, report

def print_model_summary(report: Dict) -> None:
    """
    Print a formatted summary of the model evaluation report.
    
    Args:
        report (Dict): Model evaluation report
    """
    print("\n" + "="*80)
    print("           XGBoost Model Training and Evaluation Report")
    print("="*80)
    
    print(f"\nModel Type: {report['model_type']}")
    print(f"Training Samples: {report['training_samples']}")
    print(f"Test Samples: {report['test_samples']}")
    print(f"Number of Features: {report['num_features']}")
    
    # Test metrics
    test_metrics = report['test_metrics']
    print(f"\n📊 Test Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    # Cross-validation metrics
    cv_metrics = report['cross_validation']
    print(f"\n🔄 5-Fold Cross-Validation Results:")
    print(f"  Accuracy:  {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
    print(f"  Precision: {cv_metrics['cv_precision_mean']:.4f} ± {cv_metrics['cv_precision_std']:.4f}")
    print(f"  Recall:    {cv_metrics['cv_recall_mean']:.4f} ± {cv_metrics['cv_recall_std']:.4f}")
    print(f"  F1-Score:  {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
    print(f"  ROC-AUC:   {cv_metrics['cv_roc_auc_mean']:.4f} ± {cv_metrics['cv_roc_auc_std']:.4f}")
    
    # Feature importance
    if report['feature_importance']:
        print(f"\n🎯 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(report['feature_importance'][:10]):
            print(f"  {i+1:2d}. {feature:<30} {importance:.4f}")
    
    # Confusion matrix
    test_cm = report['confusion_matrix_test']
    print(f"\n📈 Test Confusion Matrix:")
    print(f"  Predicted:    0     1")
    print(f"  Actual 0:  {test_cm[0][0]:4d}  {test_cm[0][1]:4d}")
    print(f"  Actual 1:  {test_cm[1][0]:4d}  {test_cm[1][1]:4d}")
    
    print("\n" + "="*80)

def main(config=None, model_type="xgboost", force_retrain=True, use_data_api=True):
    """
    Main function to train a model from the command line.
    
    Args:
        config: Configuration object (optional)
        model_type (str): Type of model to train ("xgboost" or "pytorch")
        force_retrain (bool): If True, always train new model
        use_data_api (bool): If True, fetch data using API
    """
    # Use default config if none provided
    config = config or default_config
    
    if model_type.lower() == "xgboost":
        logger.info("Training XGBoost model...")
        model, report = train_xgboost_with_report(
            config=config,
            force_retrain=force_retrain,
            use_data_api=use_data_api
        )
        logger.info("XGBoost training complete")
        return model, report
    
    else:  # PyTorch training (original functionality)
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
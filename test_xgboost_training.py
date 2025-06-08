#!/usr/bin/env python3
"""
Standalone test script for XGBoost training implementation.
This tests the training pipeline without requiring bittensor dependencies.
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def load_data(csv_path, test_size=0.2, random_state=42):
    """Load and split data from CSV file."""
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
    
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size_int = int(test_size * len(X))
    test_idx, train_idx = indices[:test_size_int], indices[test_size_int:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, y_train, X_val=None, y_val=None, model_params=None, early_stopping_rounds=10, verbose=True):
    """Train an XGBoost binary classification model."""
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
        print("XGBoost model training completed")
        
    return model

def evaluate_model_with_cv(model, X, y, cv_folds=5, random_state=42):
    """Evaluate model using stratified k-fold cross-validation."""
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

def generate_model_report(model, X_train, y_train, X_test, y_test, feature_names=None, cv_folds=5):
    """Generate comprehensive model evaluation report."""
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

def print_model_summary(report):
    """Print a formatted summary of the model evaluation report."""
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

def main():
    """Main function to test XGBoost training."""
    print("Testing XGBoost Training Implementation")
    print("=" * 50)
    
    # Define paths
    data_path = "data/train_data.csv"
    model_dir = Path("conversion_subnet/model_weights")
    model_path = model_dir / "xgboost_model.pkl"
    report_path = model_dir / "model_report.json"
    
    # Create model directory
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"Error: Training data not found at {data_path}")
        return
    
    # Load data
    print(f"Loading data from {data_path}")
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    # Get feature names for importance analysis
    df = pd.read_csv(data_path)
    target_col = 'target' if 'target' in df.columns else 'conversion_happened'
    columns_to_drop = [target_col, 'time_to_conversion_seconds', 'session_id', 'id']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    feature_names = df.drop(existing_columns_to_drop, axis=1).columns.tolist()
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        verbose=True
    )
    
    # Generate comprehensive report
    print("Generating model evaluation report...")
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
    joblib.dump(model, model_path)
    print(f"Saved XGBoost model to {model_path}")
    
    # Save report as JSON (convert numpy types to native Python types)
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert numpy types in the report
    import json
    json_str = json.dumps(report, default=convert_numpy_types, indent=2)
    with open(report_path, 'w') as f:
        f.write(json_str)
    print(f"Model report saved to {report_path}")
    
    # Print summary
    print_model_summary(report)
    
    # Test model loading
    print("\nTesting model loading...")
    loaded_model = joblib.load(model_path)
    test_pred = loaded_model.predict(X_test[:5])
    print(f"Model loaded successfully. Test predictions: {test_pred}")
    
    print("\n✅ XGBoost training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Example script to demonstrate the Binary Classification Miner usage.
"""

import torch
import argparse
import numpy as np
from pathlib import Path

from conversion_subnet.miner.miner import BinaryClassificationMiner
from conversion_subnet.miner.train import load_data, train_model, save_model, load_model
from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.utils.configuration import config
from conversion_subnet.utils.log import logger


def train_demo(args):
    """Train a model and save it."""
    logger.info(f"Loading data from {args.data_path}")
    X_train, X_test, y_train, y_test = load_data(args.data_path, test_size=0.2)
    
    logger.info(f"Creating miner with default config")
    miner = BinaryClassificationMiner(config)
    
    logger.info(f"Training model for {args.epochs} epochs")
    history = train_model(
        miner=miner,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoints_dir=Path(args.checkpoint_dir),
        device=args.device
    )
    
    logger.info(f"Training complete. Final metrics:")
    logger.info(f"  Train accuracy: {history['train_acc'][-1]:.4f}")
    logger.info(f"  Validation accuracy: {history['val_acc'][-1]:.4f}")
    
    # Save model
    save_model(miner, Path(args.checkpoint_dir), args.model_name)
    logger.info(f"Model saved to {args.checkpoint_dir}/{args.model_name}")


def inference_demo(args):
    """Load a model and run inference on sample data."""
    # Create miner and load model
    miner = BinaryClassificationMiner(config)
    model_path = Path(args.checkpoint_dir) / args.model_name
    
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        load_model(miner, model_path)
    else:
        logger.error(f"Model not found at {model_path}. Run with --train first.")
        return
    
    # Create sample features
    logger.info("Creating sample features for inference")
    sample_features = {
        'session_id': 'test-session-1',
        'conversation_duration_seconds': 120.0,
        'conversation_duration_minutes': 2.0,
        'hour_of_day': 14,
        'day_of_week': 2,
        'is_business_hours': 1,
        'is_weekend': 0,
        'time_to_first_response_seconds': 5.0,
        'avg_response_time_seconds': 10.0,
        'max_response_time_seconds': 15.0,
        'min_response_time_seconds': 5.0,
        'avg_agent_response_time_seconds': 8.0,
        'avg_user_response_time_seconds': 12.0,
        'response_time_stddev': 3.0,
        'response_gap_max': 15.0,
        'messages_per_minute': 6.0,
        'total_messages': 12,
        'user_messages_count': 6,
        'agent_messages_count': 6,
        'message_ratio': 1.0,
        'avg_message_length_user': 50.0,
        'max_message_length_user': 80,
        'min_message_length_user': 20,
        'total_chars_from_user': 300,
        'avg_message_length_agent': 60.0,
        'max_message_length_agent': 100,
        'min_message_length_agent': 30,
        'total_chars_from_agent': 360,
        'question_count_agent': 4,
        'questions_per_agent_message': 0.67,
        'question_count_user': 2,
        'questions_per_user_message': 0.33,
        'sequential_user_messages': 1,
        'sequential_agent_messages': 1,
        'entities_collected_count': 3,
        'has_target_entity': 1,
        'avg_entity_confidence': 0.9,
        'min_entity_confidence': 0.8,
        'entity_collection_rate': 0.75,
        'repeated_questions': 0,
        'message_alternation_rate': 0.8
    }
    
    # Create synapse and get prediction
    synapse = ConversionSynapse(features=sample_features)
    prediction = miner.forward(synapse)
    
    # Display results
    logger.info(f"Prediction results:")
    logger.info(f"  Conversion happened: {prediction['conversion_happened']}")
    logger.info(f"  Time to conversion: {prediction['time_to_conversion_seconds']:.2f} seconds")
    logger.info(f"  Confidence: {prediction['confidence']:.4f}")
    
    # Try a few more random examples
    logger.info("\nRunning predictions on random examples:")
    for i in range(3):
        # Randomize some features
        random_features = sample_features.copy()
        random_features['conversation_duration_seconds'] = np.random.uniform(30, 300)
        random_features['has_target_entity'] = np.random.choice([0, 1])
        random_features['entities_collected_count'] = np.random.randint(0, 5)
        random_features['message_ratio'] = np.random.uniform(0.5, 2.0)
        
        synapse = ConversionSynapse(features=random_features)
        prediction = miner.forward(synapse)
        
        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Duration: {random_features['conversation_duration_seconds']:.1f}s")
        logger.info(f"  Has target: {random_features['has_target_entity']}")
        logger.info(f"  Entities: {random_features['entities_collected_count']}")
        logger.info(f"  Message ratio: {random_features['message_ratio']:.2f}")
        logger.info(f"  → Conversion: {prediction['conversion_happened']}")
        logger.info(f"  → Time: {prediction['time_to_conversion_seconds']:.2f}s")
        logger.info(f"  → Confidence: {prediction['confidence']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bittensor Analytics Miner Demo")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--data_path", type=str, default="data/train.csv", help="Path to training data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--model_name", type=str, default="demo_model.pt", help="Name of model file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                         help="Device to train on")
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    Path(args.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    
    if args.train:
        train_demo(args)
    else:
        inference_demo(args) 
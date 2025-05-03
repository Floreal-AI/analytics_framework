# Advanced Custom Miner Implementation Guide

This guide provides detailed instructions for implementing a custom miner for the Bittensor Analytics Framework subnet with advanced techniques to improve your competitive edge.

## Understanding the Core Functionality

Before customizing, understand the key components of a miner:

1. **Synapse Handling**: Processing the incoming `ConversionSynapse` object with conversation features
2. **Model Inference**: Generating predictions using your trained model
3. **Response Formatting**: Creating properly structured responses with predictions and confidence

## Implementing a Custom Miner Class

### Basic Structure

```python
import torch
import torch.nn as nn
import bittensor as bt
from typing import Dict, Optional, Tuple

from conversion_subnet.protocol import ConversionSynapse, PredictionOutput
from conversion_subnet.utils.log import logger
from conversion_subnet.constants import TIMEOUT_SEC

class CustomMiner:
    def __init__(self, config):
        """Initialize the custom miner with configuration"""
        self.config = config
        self.device = torch.device(config.miner.device)
        
        # Initialize your model architecture
        self.classification_model = self._init_classification_model()
        self.regression_model = self._init_regression_model()
        
        # Move models to device
        self.classification_model.to(self.device)
        self.regression_model.to(self.device)
        
        # Optional: Feature scaler for normalization
        self.feature_scaler = None
        
        logger.info(f"Initialized custom miner with models")
    
    def _init_classification_model(self) -> nn.Module:
        """Initialize the classification model for predicting conversion_happened"""
        # Custom architecture - just an example
        model = nn.Sequential(
            nn.Linear(40, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        return model
    
    def _init_regression_model(self) -> nn.Module:
        """Initialize the regression model for predicting time_to_conversion_seconds"""
        # Custom architecture for regression task
        model = nn.Sequential(
            nn.Linear(40, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # ReLU to ensure positive values
        )
        return model
    
    def _preprocess_features(self, features: Dict) -> torch.Tensor:
        """Preprocess and normalize input features"""
        # Extract feature values in consistent order
        feature_values = [
            features.get('conversation_duration_seconds', 0.0),
            features.get('has_target_entity', 0),
            # ... add all other features in fixed order
        ]
        
        # Convert to tensor
        features_tensor = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)
        
        # Apply normalization if scaler exists
        if self.feature_scaler is not None:
            features_tensor = self.feature_scaler(features_tensor)
            
        return features_tensor.to(self.device)
    
    def forward(self, synapse: ConversionSynapse) -> PredictionOutput:
        """Process the input features and return prediction"""
        try:
            # Preprocess features
            features_tensor = self._preprocess_features(synapse.features)
            
            # Classification prediction
            with torch.no_grad():
                classification_output = self.classification_model(features_tensor)
                confidence = float(classification_output.item())
                prediction = int(round(confidence))
                
                # Only perform regression if we predict conversion
                if prediction == 1:
                    time_prediction = self.regression_model(features_tensor)
                    time_seconds = float(time_prediction.item())
                    # Ensure prediction is positive and reasonable
                    time_seconds = max(1.0, min(time_seconds, TIMEOUT_SEC * 2))
                else:
                    time_seconds = -1.0
                
            return {
                'conversion_happened': prediction,
                'time_to_conversion_seconds': time_seconds,
                'confidence': confidence
            }
        except Exception as e:
            # Robust error handling
            logger.error(f"Error in forward: {e}")
            return {
                'conversion_happened': 0,
                'time_to_conversion_seconds': -1.0,
                'confidence': 0.0
            }
            
    async def forward_async(self, synapse: ConversionSynapse) -> PredictionOutput:
        """Async version of forward for compatibility with Bittensor"""
        return self.forward(synapse)
```

## Advanced Techniques

### 1. Feature Engineering

Enhance your model's performance with feature engineering:

```python
def _preprocess_features(self, features: Dict) -> torch.Tensor:
    """Advanced feature engineering for better predictions"""
    # 1. Extract base features
    base_features = [features.get(key, 0.0) for key in self.feature_order]
    
    # 2. Create derived features
    agent_to_user_ratio = features.get('agent_messages_count', 1) / max(1, features.get('user_messages_count', 1))
    avg_message_length_ratio = features.get('avg_message_length_agent', 1) / max(1, features.get('avg_message_length_user', 1))
    question_density = features.get('question_count_agent', 0) / max(1, features.get('agent_messages_count', 1))
    
    # 3. Time-based features
    time_features = [
        np.sin(2 * np.pi * features.get('hour_of_day', 0) / 24),  # Cyclical encoding for hour
        np.cos(2 * np.pi * features.get('hour_of_day', 0) / 24),
        np.sin(2 * np.pi * features.get('day_of_week', 0) / 7),   # Cyclical encoding for day
        np.cos(2 * np.pi * features.get('day_of_week', 0) / 7),
    ]
    
    # 4. Combine all features
    all_features = base_features + [
        agent_to_user_ratio, 
        avg_message_length_ratio,
        question_density
    ] + time_features
    
    # 5. Convert to tensor and normalize
    features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)
    
    if self.feature_scaler is not None:
        features_tensor = self.feature_scaler(features_tensor)
        
    return features_tensor.to(self.device)
```

### 2. Ensemble Models

Improve prediction accuracy with ensemble techniques:

```python
def _init_models(self):
    """Initialize multiple models for ensemble prediction"""
    self.classifiers = [
        self._create_classifier(architecture=1),
        self._create_classifier(architecture=2),
        self._create_classifier(architecture=3)
    ]
    
    self.regressors = [
        self._create_regressor(architecture=1),
        self._create_regressor(architecture=2)
    ]
    
    # Move all models to device
    for model in self.classifiers + self.regressors:
        model.to(self.device)

def forward(self, synapse: ConversionSynapse) -> PredictionOutput:
    """Ensemble prediction combining multiple models"""
    try:
        features_tensor = self._preprocess_features(synapse.features)
        
        # Classification ensemble
        class_outputs = []
        for classifier in self.classifiers:
            with torch.no_grad():
                output = classifier(features_tensor)
                class_outputs.append(output.item())
        
        # Average confidence
        confidence = sum(class_outputs) / len(class_outputs)
        prediction = int(round(confidence))
        
        # Regression ensemble (only if conversion predicted)
        if prediction == 1:
            time_predictions = []
            for regressor in self.regressors:
                with torch.no_grad():
                    output = regressor(features_tensor)
                    time_predictions.append(output.item())
            
            # Average time prediction
            time_seconds = sum(time_predictions) / len(time_predictions)
            time_seconds = max(1.0, min(time_seconds, TIMEOUT_SEC * 2))
        else:
            time_seconds = -1.0
            
        return {
            'conversion_happened': prediction,
            'time_to_conversion_seconds': time_seconds,
            'confidence': confidence
        }
    except Exception as e:
        logger.error(f"Error in forward: {e}")
        return {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0, 'confidence': 0.0}
```

### 3. Adaptive Confidence Calibration

Improve your rewards by calibrating confidence scores:

```python
class ConfidenceCalibrator:
    """Calibrates confidence scores for better diversity rewards"""
    def __init__(self, bins=10):
        self.bins = bins
        self.bin_counts = np.zeros(bins)
        self.bin_correct = np.zeros(bins)
        
    def update(self, confidence, correct):
        """Update calibration statistics"""
        bin_idx = min(int(confidence * self.bins), self.bins - 1)
        self.bin_counts[bin_idx] += 1
        if correct:
            self.bin_correct[bin_idx] += 1
    
    def calibrate(self, confidence):
        """Calibrate a confidence score based on historical accuracy"""
        bin_idx = min(int(confidence * self.bins), self.bins - 1)
        if self.bin_counts[bin_idx] > 0:
            return self.bin_correct[bin_idx] / self.bin_counts[bin_idx]
        return confidence

# In your miner class:
def __init__(self, config):
    # ... other initialization code
    self.calibrator = ConfidenceCalibrator()
    self.history = []  # Store prediction history for offline calibration

def forward(self, synapse: ConversionSynapse) -> PredictionOutput:
    # ... prediction code
    
    # Apply confidence calibration
    calibrated_confidence = self.calibrator.calibrate(confidence)
    
    # Store this prediction for later calibration
    self.history.append({
        'raw_confidence': confidence,
        'calibrated_confidence': calibrated_confidence,
        'prediction': prediction
    })
    
    return {
        'conversion_happened': prediction,
        'time_to_conversion_seconds': time_seconds,
        'confidence': calibrated_confidence  # Use calibrated confidence
    }
```

## Advanced Training Techniques

### 1. Two-Stage Training

Improve accuracy with specialized models:

```python
def train_two_stage_model(
    X_train, y_train, X_val, y_val, 
    epochs=20, 
    batch_size=64,
    learning_rate=0.001
):
    """Train classification and regression models separately"""
    # Step 1: Train classification model
    classification_model = create_classification_model()
    classification_optimizer = torch.optim.Adam(
        classification_model.parameters(), 
        lr=learning_rate
    )
    classification_criterion = nn.BCELoss()
    
    # Train classification model
    for epoch in range(epochs):
        # ... classification training code
    
    # Step 2: Filter data for regression model (only conversion=1 cases)
    conversion_indices = np.where(y_train['conversion_happened'] == 1)[0]
    X_train_reg = X_train[conversion_indices]
    y_train_reg = y_train['time_to_conversion_seconds'][conversion_indices]
    
    # Create and train regression model
    regression_model = create_regression_model()
    regression_optimizer = torch.optim.Adam(
        regression_model.parameters(), 
        lr=learning_rate
    )
    regression_criterion = nn.MSELoss()
    
    # Train regression model
    for epoch in range(epochs):
        # ... regression training code
    
    return classification_model, regression_model
```

### 2. Custom Loss Functions

Optimize directly for the reward function:

```python
class RewardOptimizedLoss(nn.Module):
    """Custom loss function that optimizes for the validator reward function"""
    def __init__(self, class_weight=0.55, reg_weight=0.35, div_weight=0.1):
        super().__init__()
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        self.div_weight = div_weight
        self.bce = nn.BCELoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, 
                class_pred, class_true, 
                reg_pred, reg_true,
                confidence):
        # Binary classification loss
        class_loss = self.bce(class_pred, class_true)
        
        # Regression loss (only for positive cases)
        mask = (class_true == 1).float()
        if mask.sum() > 0:
            reg_loss = self.mse(reg_pred * mask, reg_true * mask)
            reg_loss = reg_loss.sum() / (mask.sum() + 1e-8)
        else:
            reg_loss = torch.tensor(0.0, device=class_pred.device)
        
        # Diversity loss (penalize predictions close to 0.5)
        div_loss = 1 - torch.abs(confidence - 0.5)
        
        # Combined loss
        total_loss = (
            self.class_weight * class_loss + 
            self.reg_weight * reg_loss + 
            self.div_weight * div_loss
        )
        
        return total_loss.mean()
```

## Integration With Bittensor

### 1. Custom Axon Implementation

For advanced integration with the Bittensor network:

```python
import asyncio
import bittensor as bt
from typing import List, Dict, Any

class CustomMinerNeuron:
    def __init__(self, config=None):
        # Initialize wallet, subtensor, metagraph
        self.config = config or bt.Config()
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        
        # Create custom miner
        self.miner = CustomMiner(self.config)
        
        # Initialize axon
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        
        # Register forward function
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist_fn,
        )
        
        # Start axon
        self.axon.start()
        
    async def forward(self, synapse: ConversionSynapse) -> ConversionSynapse:
        """Handle incoming synapse requests from validators"""
        # Measure response time
        start_time = time.time()
        
        # Process prediction
        prediction = await self.miner.forward_async(synapse)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Attach prediction to synapse
        synapse.prediction = prediction
        synapse.confidence = prediction.get('confidence', 0.5)
        synapse.response_time = response_time
        
        return synapse
    
    def blacklist_fn(self, synapse: bt.Synapse) -> bool:
        """Determine if request should be blacklisted"""
        # Example: blacklist if not the correct synapse type or if hotkey not in metagraph
        if not isinstance(synapse, ConversionSynapse):
            return True
        return False
```

### 2. Performance Monitoring

Implement monitoring to track your miner's performance:

```python
class MinerMonitor:
    """Tracks and analyzes miner performance for optimization"""
    def __init__(self):
        self.requests = []
        self.responses = []
        self.response_times = []
        self.last_updated = time.time()
        
    def record_request(self, synapse: ConversionSynapse, uid: int):
        """Record an incoming request"""
        self.requests.append({
            'timestamp': time.time(),
            'uid': uid,
            'features': synapse.features.copy()
        })
        
    def record_response(self, synapse: ConversionSynapse, uid: int):
        """Record a response sent to validator"""
        response_time = time.time() - self.last_updated
        self.responses.append({
            'timestamp': time.time(),
            'uid': uid,
            'prediction': synapse.prediction.copy(),
            'confidence': synapse.confidence,
            'response_time': response_time
        })
        self.response_times.append(response_time)
        self.last_updated = time.time()
        
    def get_stats(self):
        """Get performance statistics"""
        if not self.response_times:
            return {}
            
        return {
            'avg_response_time': sum(self.response_times) / len(self.response_times),
            'max_response_time': max(self.response_times),
            'min_response_time': min(self.response_times),
            'request_count': len(self.requests),
            'avg_confidence': sum(r['confidence'] for r in self.responses) / len(self.responses),
            'positive_rate': sum(1 for r in self.responses if r['prediction']['conversion_happened'] == 1) / len(self.responses)
        }
    
    def log_stats(self, interval=3600):
        """Log statistics periodically"""
        now = time.time()
        if now - self.last_log > interval:
            stats = self.get_stats()
            logger.info(f"Performance stats: {stats}")
            self.last_log = now
```

## Running and Optimization

### 1. Continuous Learning

Implement continuous learning to improve over time:

```python
class ContinuousLearningMiner(CustomMiner):
    """Miner that continuously learns from new data"""
    def __init__(self, config):
        super().__init__(config)
        self.buffer = []  # Store recent predictions
        self.buffer_max_size = 1000
        self.last_training = time.time()
        self.training_interval = 86400  # 24 hours
        
    def update_buffer(self, features, prediction):
        """Add prediction to buffer"""
        self.buffer.append((features, prediction))
        if len(self.buffer) > self.buffer_max_size:
            self.buffer.pop(0)  # Remove oldest entry
    
    async def forward_async(self, synapse: ConversionSynapse) -> PredictionOutput:
        """Forward with learning capability"""
        prediction = await super().forward_async(synapse)
        self.update_buffer(synapse.features, prediction)
        
        # Check if it's time to retrain
        now = time.time()
        if now - self.last_training > self.training_interval:
            asyncio.create_task(self.continuous_training())
            self.last_training = now
            
        return prediction
        
    async def continuous_training(self):
        """Background task for continuous model updating"""
        if len(self.buffer) < 100:
            return  # Not enough data
        
        logger.info(f"Starting continuous learning with {len(self.buffer)} samples")
        
        # Extract data from buffer
        features = [item[0] for item in self.buffer]
        predictions = [item[1] for item in self.buffer]
        
        # Convert to training format
        X = np.array([list(f.values()) for f in features])
        y_class = np.array([p['conversion_happened'] for p in predictions])
        
        # Finetune model (minimal updates to avoid catastrophic forgetting)
        # ... training code with small learning rate
        
        logger.info("Continuous learning complete")
```

## Conclusion

Building a custom miner for the Bittensor Analytics Framework requires a combination of:

1. **Technical Implementation**: Proper handling of the protocol and model inference
2. **Model Expertise**: Creating high-performing predictive models
3. **Bittensor Integration**: Effective integration with the network mechanisms

By implementing these advanced techniques, you can significantly improve your miner's performance, ultimately leading to higher rewards in the subnet.

Remember that the competition evolves over time, so continuous improvement and adaptation to new challenges are essential for long-term success. 
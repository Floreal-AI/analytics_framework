# Miner Quickstart Guide - Bittensor Analytics Framework

This guide will help you get started as a miner in the Bittensor Analytics Subnet, where you'll deploy AI models to predict task success and timing for conversational agents.

## Introduction

In this subnet, miners predict:
1. Whether an AI agent will successfully complete its task (`conversion_happened`: 0 or 1)
2. How long it will take to complete the task (`time_to_conversion_seconds`)

Your predictions are evaluated by validators who issue conversation features in real-time, and your rewards are based on the accuracy of your predictions.

## Prerequisites

- Python 3.8-3.11
- PyTorch
- Basic understanding of neural networks and binary classification
- Bittensor wallet with sufficient TAO for registering as a miner

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Floreal-AI/analytics_framework.git
   cd analytics_framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Bittensor wallet:
   ```bash
   btcli wallet create --wallet.name miner
   ```

## Understanding the Framework

### Protocol

The communication protocol revolves around the `ConversionSynapse`, which contains:

- **Input**: 40 conversation features like `conversation_duration_seconds`, `has_target_entity`, etc.
- **Output**: Your prediction containing `conversion_happened` (0 or 1) and `time_to_conversion_seconds`
- **Metadata**: Your response time and confidence score

### Default Miner Implementation

The framework provides a default `BinaryClassificationMiner` class with:

- A configurable multi-layer perceptron (MLP) model
- Methods to process inputs and generate predictions
- Error handling for robustness

## Training Your Model

The most critical part of being a successful miner is having a well-trained model. The framework provides tools for this:

1. **Data Loading**:
   ```python
   from conversion_subnet.miner.train import load_data
   
   # Load and split training data
   X_train, X_test, y_train, y_test = load_data("path/to/train.csv")
   ```

2. **Creating and Training a Model**:
   ```python
   from conversion_subnet.miner.miner import BinaryClassificationMiner
   from conversion_subnet.miner.train import train_model
   from conversion_subnet.utils.configuration import config
   
   # Initialize miner with default config
   miner = BinaryClassificationMiner(config)
   
   # Train the model
   history = train_model(
       miner=miner,
       X_train=X_train,
       y_train=y_train,
       X_val=X_test,
       y_val=y_test,
       epochs=20,
       batch_size=64,
       learning_rate=0.001,
       checkpoints_dir="checkpoints",
       device="cuda" if torch.cuda.is_available() else "cpu"
   )
   ```

3. **Saving and Loading Models**:
   ```python
   from conversion_subnet.miner.train import save_model, load_model
   from pathlib import Path
   
   # Save trained model
   save_model(miner, Path("checkpoints"), "my_model.pt")
   
   # Load trained model later
   load_model(miner, Path("checkpoints/my_model.pt"))
   ```

## Advanced Model Building

To improve your ranking and rewards, consider customizing your model beyond the default MLP:

1. **Custom Model Architecture**:
   ```python
   import torch.nn as nn
   
   class CustomModel(nn.Module):
       def __init__(self, input_size=40):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(input_size, 128),
               nn.BatchNorm1d(128),
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(128, 64),
               nn.ReLU(),
               nn.Linear(64, 32),
               nn.ReLU(),
               nn.Linear(32, 1),
               nn.Sigmoid()
           )
           
       def forward(self, x):
           return self.layers(x)
   ```

2. **Feature Engineering**:
   - Normalize or standardize input features
   - Create interaction terms between related features
   - Develop custom regression model for time prediction

3. **Time-to-conversion Prediction Strategy**:
   - Consider separate models for binary classification and regression
   - For example, only predict time when you're confident conversion will happen

## Running Your Miner

1. **Create a Bittensor Config**:
   ```bash
   # Create a subtensor config
   btcli new_subtensor --subtensor.chain_endpoint wss://your-subnet.chain.endpoint --subtensor._network <your-network>
   ```

2. **Register Your Miner**:
   ```bash
   python -m conversion_subnet.miner.run --subtensor.network <network> --wallet.name miner --wallet.hotkey default --logging.debug
   ```

3. **Monitoring Performance**:
   - Use the Bittensor dashboard to check your miner's score and ranking
   - Analyze logs to identify prediction errors or improvement opportunities

## Scoring Mechanism

Understanding how validators score you helps optimize your model:

1. **Classification Reward (55%)**:
   - Correct prediction of `conversion_happened`
   - Class-weighted to handle imbalance

2. **Regression Reward (35%)**:
   - Accuracy of `time_to_conversion_seconds` prediction
   - Only applies when your classification is correct

3. **Diversity Reward (10%)**:
   - Encourages unique, confident predictions
   - Penalizes predictions near 50% probability

4. **Time Reward (20% of total)**:
   - Fast responses (within 60 seconds)
   - Critical for real-time performance

## Best Practices

1. **Model Performance**:
   - Train on diverse datasets
   - Use cross-validation to prevent overfitting
   - Consider ensemble methods for better predictions

2. **Reliability**:
   - Implement error handling for all edge cases
   - Have fallback strategies when predictions fail
   - Ensure fast response times (< 60 seconds)

3. **Continuous Improvement**:
   - Regularly retrain models with new data
   - Monitor your performance metrics
   - Stay updated with subnet changes

## Troubleshooting

1. **Common Issues**:
   - Tensor shape mismatches
   - Out-of-memory errors with large models
   - Connection timeouts to validators

2. **Debug Strategies**:
   - Use `--logging.debug` for detailed logs
   - Test your model on historical data before deploying
   - Verify data preprocessing matches training workflow

## Conclusion

Success as a miner in the Analytics Framework depends on balancing:
- Prediction accuracy (classification and regression)
- Response speed
- Model uniqueness

Start with the default implementation, understand the framework thoroughly, then customize your approach to maximize rewards.

Happy mining!

## Additional Resources

- [Bittensor Documentation](https://docs.bittensor.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Scorer Mechanism Details](https://github.com/Floreal-AI/analytics_framework/blob/main/README.md#scoring) 
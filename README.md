# Bittensor Conversion Subnet: Predicting AI Agent Performance in Real-Time

<div align="center">
  <img src="https://raw.githubusercontent.com/opentensor/bittensor/master/docs/assets/logo.jpg" width="200" alt="Bittensor Logo"/>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/downloads/)

A decentralized framework for evaluating and improving AI agent task performance in real-time conversations. This subnet allows participants to deploy models (miners) that predict whether an AI agent will successfully complete its conversion task and how long it will take.

## Overview

This subnet evaluates AI agents' ability to perform tasks in customer conversations, like convincing users to request quotes or complete transactions. Miners predict:

1. 🎯 `conversion_happened`: Will the agent successfully convince the user? (Binary: 0/1)
2. ⏱️ `time_to_conversion_seconds`: How long will it take? (Seconds if converted, -1.0 if not)

Validators issue conversation challenges based on 40 conversation metrics, score predictions, and set weights to drive token rewards, creating a self-improving ecosystem.

## Features

- 🔄 **Real-time evaluation**: Miners must respond to live conversation data within 60 seconds
- 📊 **Rich feature set**: 40 conversation metrics for precise predictions
- 💰 **Fair reward system**: Balanced scoring across classification, regression, and response time
- 🧠 **Self-improvement loop**: Miners continuously improve with validator feedback
- 🛠️ **Easy deployment**: Simple installation and configuration

## Dataset

### Train Set (train.csv)
A fixed dataset provided for training AI models. Contains conversation sessions with:
- 40 Features: Conversation metrics (e.g., conversation_duration_seconds, has_target_entity)
- 3 Target Variables: conversion_happened (1 if task succeeded, 0 otherwise), time_to_conversion_seconds (time to task completion if succeeded, -1.0 otherwise), time_to_conversion_minutes (seconds / 60)

#### Format:

```
session_id,conversation_duration_seconds,conversation_duration_minutes,hour_of_day,day_of_week,is_business_hours,is_weekend,time_to_first_response_seconds,avg_response_time_seconds,max_response_time_seconds,min_response_time_seconds,avg_agent_response_time_seconds,avg_user_response_time_seconds,response_time_stddev,response_gap_max,messages_per_minute,total_messages,user_messages_count,agent_messages_count,message_ratio,avg_message_length_user,max_message_length_user,min_message_length_user,total_chars_from_user,avg_message_length_agent,max_message_length_agent,min_message_length_agent,total_chars_from_agent,question_count_agent,questions_per_agent_message,question_count_user,questions_per_user_message,sequential_user_messages,sequential_agent_messages,entities_collected_count,has_target_entity,avg_entity_confidence,min_entity_confidence,entity_collection_rate,repeated_questions,message_alternation_rate,conversion_happened,time_to_conversion_seconds,time_to_conversion_minutes
```

#### Example Row:

```
fe093ea6-8fb3-4856-9aee-05673faaae11,92.0000,1.5333,13,2,1,0,7.0000,11.5000,15.0000,7.0000,11.7500,11.6667,2.8723,15.0000,5.8696,9,5,4,1.2500,47.0000,62,25,235,84.0000,120,53,336,3,0.7500,0,0.0000,2,1,5,1,0.9200,0.8000,0.0000,0,0.8750,1,62.0000,1.0333
```

### Test Set (Dynamic, Real-Time Conversations)
A stream of actual conversations arriving sequentially via the subnet. Each conversation provides the same 40 features, but no targets.

#### Format (Input to Miners):

```
session_id,conversation_duration_seconds,conversation_duration_minutes,hour_of_day,day_of_week,is_business_hours,is_weekend,time_to_first_response_seconds,avg_response_time_seconds,max_response_time_seconds,min_response_time_seconds,avg_agent_response_time_seconds,avg_user_response_time_seconds,response_time_stddev,response_gap_max,messages_per_minute,total_messages,user_messages_count,agent_messages_count,message_ratio,avg_message_length_user,max_message_length_user,min_message_length_user,total_chars_from_user,avg_message_length_agent,max_message_length_agent,min_message_length_agent,total_chars_from_agent,question_count_agent,questions_per_agent_message,question_count_user,questions_per_user_message,sequential_user_messages,sequential_agent_messages,entities_collected_count,has_target_entity,avg_entity_confidence,min_entity_confidence,entity_collection_rate,repeated_questions,message_alternation_rate
```

#### Ground Truth (Dynamic, Private):
True targets for test conversations, generated post-conversation based on AI agent task success.

```
session_id,conversion_happened,time_to_conversion_seconds
e29b3b9d-d5c6-4007-9da7-3d449b3bd89c,1,67.5316
e2849720-2b33-4bd9-8690-ba35bafbfec3,0,-1.0000
```

## Quick Start

### Prerequisites

- Python 3.8-3.11
- Linux, macOS, or Windows

### Installation

```bash
# Clone the repository
git clone https://github.com/Floreal-AI/analytics_framework.git
cd analytics_framework

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Running a Validator

```bash
# Start a validator neuron
python -m neurons.validator

# Alternative using the CLI (if installed with pip install -e .)
validator-conversion
```

### Running a Miner

```bash
# Start a miner neuron with default settings
python -m neurons.miner

# Alternative using the CLI (if installed with pip install -e .)
miner-conversion
```

## Understanding the Data

### Conversation Features (Input)

Miners receive 40 conversation metrics, including:

```python
{
    'session_id': 'fe093ea6-8fb3-4856-9aee-05673faaae11',
    'conversation_duration_seconds': 92.0000,
    'has_target_entity': 1,
    'entities_collected_count': 5,
    'message_ratio': 1.2500,
    'total_messages': 9,
    'user_messages_count': 5,
    'agent_messages_count': 4,
    # ... and many more metrics
}
```

### Prediction Output

Miners must return:

```python
{
    'conversion_happened': 1,  # 1 = conversion successful, 0 = unsuccessful
    'time_to_conversion_seconds': 62.0000  # Time to conversion or -1.0 if no conversion
}
```

## How the Reward System Works

The scoring system balances multiple factors to ensure accurate, timely, and diverse predictions:

```python
# Example of how a miner's prediction is scored
def score_prediction(ground_truth, prediction, confidence, response_time):
    # Classification reward (55% of prediction score)
    class_reward = 1.0 if prediction['conversion_happened'] == ground_truth['conversion_happened'] else 0.0
    
    # Apply class weights to handle imbalance
    if class_reward > 0:
        class_reward *= CLASS_WEIGHTS['positive' if ground_truth['conversion_happened'] == 1 else 'negative']
    
    # Regression reward (45% of prediction score)
    reg_reward = 0.0
    if prediction['conversion_happened'] == 1 and ground_truth['conversion_happened'] == 1:
        mae = abs(prediction['time_to_conversion_seconds'] - ground_truth['time_to_conversion_seconds'])
        reg_reward = max(1.0 - mae / BASELINE_MAE, 0.0)  # BASELINE_MAE = 15.0 seconds

    # Combine prediction components
    pred_reward = 0.55 * class_reward + 0.35 * reg_reward + 0.10 * div_reward
    
    # Time reward (20% of total score)
    time_reward = max(1.0 - response_time / 60.0, 0.0)  # 60-second timeout
    
    # Final reward
    return 0.80 * pred_reward + 0.20 * time_reward
```

## Configuration

### Validator Configuration

```bash
# Run with custom settings
python -m neurons.validator --neuron.sample_size 15 --neuron.alpha_prediction 0.9
```

### Miner Configuration

```bash
# Run with custom device
python -m neurons.miner --miner.device "cuda" --neuron.wallet.name "miner_wallet"
```

## Understanding the Scores

Your miner's performance is measured by:

1. **Classification Accuracy**: Can you correctly predict if a conversion will happen?
2. **Regression Accuracy**: Can you precisely estimate the conversion time?
3. **Prediction Boldness**: Are you making confident predictions (not hovering around 0.5)?
4. **Response Speed**: Are you responding quickly to real-time requests?

The final score, used to determine rewards, is an exponential moving average (EMA) of these components.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](contrib/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Bittensor Subnet Competition: Evaluating and Improving AI Agent Performance in Real-Time Conversations


Welcome to the Bittensor Subnet Competition, a decentralized challenge hosted on a Bittensor subnet to evaluate and enhance AI agent task performance in real-time conversational tasks. Participants deploy miners (AI models) to predict whether an AI agent (e.g., a "quote agent") will successfully complete its task—such as convincing a user to request a quote and delivering it—and the time taken to do so. Validators issue conversation challenges, score predictions, and set weights to drive Alpha Token rewards and self-improvement. Conversations arrive sequentially, requiring miners to submit predictions within a 60-second window.

This competition leverages Bittensor’s subnet architecture to foster innovation in AI agent development. By rewarding miners for accurate, timely predictions and validators for fair evaluations, the subnet creates a decentralized ecosystem where agents continuously improve their ability to perform tasks in dynamic conversations across industries like customer support and sales.

## Overview

### Datasets

#### Train Set (train.csv):
A fixed dataset provided by the organizer for training AI models.
Contains N sessions (to be specified upon receipt) with:
40 Features: Conversation metrics (e.g., conversation_duration_seconds, has_target_entity).
3 Target Variables: conversion_happened (1 if task succeeded, 0 otherwise), time_to_conversion_seconds (time to task completion if succeeded, -1.0 otherwise), time_to_conversion_minutes (seconds / 60).

Format:
session_id,conversation_duration_seconds,conversation_duration_minutes,hour_of_day,day_of_week,is_business_hours,is_weekend,time_to_first_response_seconds,avg_response_time_seconds,max_response_time_seconds,min_response_time_seconds,avg_agent_response_time_seconds,avg_user_response_time_seconds,response_time_stddev,response_gap_max,messages_per_minute,total_messages,user_messages_count,agent_messages_count,message_ratio,avg_message_length_user,max_message_length_user,min_message_length_user,total_chars_from_user,avg_message_length_agent,max_message_length_agent,min_message_length_agent,total_chars_from_agent,question_count_agent,questions_per_agent_message,question_count_user,questions_per_user_message,sequential_user_messages,sequential_agent_messages,entities_collected_count,has_target_entity,avg_entity_confidence,min_entity_confidence,entity_collection_rate,repeated_questions,message_alternation_rate,conversion_happened,time_to_conversion_seconds,time_to_conversion_minutes

Example Row (hypothetical):
fe093ea6-8fb3-4856-9aee-05673faaae11,92.0,1.5333333333333334,13,2,1,0,7.0,11.5,15.0,7.0,11.75,11.666666666666666,2.8722813232690143,15.0,5.869565217391304,9,5,4,1.25,47.0,62,25,235,84.0,120,53,336,3,0.75,0,0.0,2,1,5,1,0.9200000000000002,0.8,0,0,0.875,1,62.0,1.0333333333333334
#### Test Set (Dynamic, Real-Time Conversations):
A stream of actual conversations arriving sequentially via the subnet.
Each conversation provides the same 40 features, but no targets.
Miners predict conversion_happened and time_to_conversion_seconds.
Format: Identical to train set features:

session_id,conversation_duration_seconds,conversation_duration_minutes,hour_of_day,day_of_week,is_business_hours,is_weekend,time_to_first_response_seconds,avg_response_time_seconds,max_response_time_seconds,min_response_time_seconds,avg_agent_response_time_seconds,avg_user_response_time_seconds,response_time_stddev,response_gap_max,messages_per_minute,total_messages,user_messages_count,agent_messages_count,message_ratio,avg_message_length_user,max_message_length_user,min_message_length_user,total_chars_from_user,avg_message_length_agent,max_message_length_agent,min_message_length_agent,total_chars_from_agent,question_count_agent,questions_per_agent_message,question_count_user,questions_per_user_message,sequential_user_messages,sequential_agent_messages,entities_collected_count,has_target_entity,avg_entity_confidence,min_entity_confidence,entity_collection_rate,repeated_questions,message_alternation_rate


#### Ground Truth Dynamic, Private):
True targets for test conversations, generated post-conversation based on AI agent task success (e.g., convincing user and providing a quote).

session_id,conversion_happened,time_to_conversion_seconds
e29b3b9d-d5c6-4007-9da7-3d449b3bd89c,1,67.53160202861642
e2849720-2b33-4bd9-8690-ba35bafbfec3,0,-1.0

### Roles

#### Miners:
Deploy AI models to predict task success (conversion_happened) and time (time_to_conversion_seconds).
Compete for Alpha Token rewards based on validator scores, driving self-improvement through model refinement.



#### Validators:
Send conversation features to miners, evaluate predictions against ground truth.

## Objectives
Predict Task Success: Determine if the AI agent will succeed in its task (conversion_happened = 1) or not (conversion_happened = 0).
Predict Task Time: Estimate time_to_conversion_seconds for successful tasks, -1.0 otherwise.
Drive Self-Improvement: Use validator scores to refine models, enhancing task performance.


## Incentive Mechanism

### Challenge
Validators send a ConversionSynapse object containing 40 conversation features (e.g., conversation_duration_seconds, has_target_entity) to miners.
Features are serialized as a JSON dictionary.
Miners must respond within 60 seconds to align with real-time requirements.

### Miner Response
Miners predict:
conversion_happened: Binary (0 or 1).
time_to_conversion_seconds: Positive float if conversion_happened = 1, -1.0 if conversion_happened = 0.
Predictions are attached to the ConversionSynapse and returned to validators.

### Scoring

#### Classification Reward
For single-row scoring, use binary accuracy adjusted by a class-weight penalty to handle imbalance (~62.5% conversions):

\[
\text{Classification Reward} = \begin{cases} 
1 & \text{if predicted = true} \\
0 & \text{if predicted} \neq \text{true}
\end{cases}
\]

Apply a weight adjustment to penalize over-predicting conversions:
- If `conversion_happened = 1` (positive), reward is scaled by \(\frac{\text{Negatives}}{\text{Positives} + \text{Negatives}}\) (~0.375 if 62.5% positives).
- If `conversion_happened = 0` (negative), reward is scaled by \(\frac{\text{Positives}}{\text{Positives} + \text{Negatives}}\) (~0.625).

This approximates balanced accuracy for single rows, encouraging nuanced models without batch dependency. Weights are estimated from historical data (updated periodically by validators, e.g., every 100 conversations).

#### Regression Reward
For conversations where both ground truth and prediction have `conversion_happened = 1`, compute Mean Absolute Error (MAE) to reward precise timing while being robust to outliers (ground truth range 60.0–91.43):

\[
\text{MAE} = |\text{predicted_time} - \text{true_time}|
\]

\[
\text{Regression Score} = \max\left(1 - \frac{\text{MAE}}{15.0}, 0\right)
\]

Baseline MAE (~15.0) is estimated from ground truth sample std (11.74); adjustable with train set data. If no correct conversion prediction, set Regression Score = 0.

#### Diversity Reward
To encourage diverse strategies without batch processing, use a prediction confidence penalty based on the miner’s predicted probability for `conversion_happened` (from model’s `predict_proba`):

\[
\text{Confidence Penalty} = 1 - |\text{predicted_probability} - 0.5|
\]

Miners with high confidence (near 0 or 1) receive lower penalties, rewarding bold, unique predictions. This approximates diversity by penalizing conservative predictions (near 0.5), actionable for single rows.

#### Time Reward
Reward fast responses to prioritize real-time performance:

\[
\text{Time Reward} = \max\left(1 - \frac{\text{response_time}}{60}, 0\right)
\]

#### Prediction Reward
Combine metrics, prioritizing classification and regression:

\[
\text{Prediction Reward} = 0.55 \cdot \text{Classification Reward} + 0.35 \cdot \text{Regression Score} + 0.1 \cdot \text{Confidence Penalty}
\]

- 55% classification reflects task success priority.
- 35% regression ensures timing accuracy, leveraging large train set.
- 10% confidence penalty promotes diversity without batch dependency.

#### Total Reward
Balance prediction and speed:

\[
\text{Total Reward} = 0.8 \cdot \text{Prediction Reward} + 0.2 \cdot \text{Time Reward}
\]

20% time reward emphasizes real-time needs, balanced to avoid sacrificing quality.

#### EMA Scoring
Smooth rewards for stability over sequential conversations:

\[
\text{EMA Score}_t = 0.1 \cdot \text{Total Reward}_t + 0.9 \cdot \text{EMA Score}_{t-1}
\]

\(\beta = 0.1\) balances responsiveness and consistency, suitable for frequent test conversations.

#### Weight Setting
Validators normalize EMA scores to create a weight vector for miners. Weights are submitted to the blockchain, aggregated via Yuma Consensus, and combined with stake to allocate TAO emissions every tempo. Validators align scores to maximize V-Trust, ensuring fairness.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Floreal-AI/analytics_framework.git
   cd quote-prediction-subnet
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Python 3.8–3.11 and Bittensor are installed.

## Running the Subnet

### Validator

Run a validator to generate data and score miner responses:

```bash
python -m ocr_subnet.validator
```

### Miner

Run a miner to predict quote outcomes:

```bash
python -m neurons.miner
```

## Configuration

- **Validator Config**: Adjust `neuron.timeout` and `neuron.alpha_prediction` in the validator script or config file.
- **Miner Enhancements**: Improve the baseline miner by implementing machine learning models (e.g., using `scikit-learn`).

## Dataset

The subnet uses synthetic data based on the following features:

- `interaction_duration`: Mean 117.13s, std 12.36
- `question_count`: Mean 5.53, std 1.08
- `sentiment_score`: Mean 0.71, std 0.09
- ... (see `validator/generate.py` for full list)

**Targets**:

- `quote_delivery_time`: Continuous (mean \~108s, target ≤90s)
- `quote_acceptance`: Binary (mean 80%, target ≥85%)

## Contributing

See `contrib/CONTRIBUTING.md` for guidelines. Submit issues or PRs to improve miners, validators, or documentation.

## License

MIT License. See `LICENSE` for details.

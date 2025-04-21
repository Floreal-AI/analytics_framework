
import torch
import numpy as np
from typing import Dict
from conversion_subnet.protocol import ConversionSynapse

class Validator:
    def __init__(self):
        """
        Initialize the Validator with initial class weights, EMA scores, and ground truth history.
        """
        self.class_weights = {'positive': 0.375, 'negative': 0.625}  # Initial weights based on ~62.5% conversions
        self.ema_scores = {}  # Miner UID to EMA score
        self.ground_truth_history = []  # Store ground truth for weight updates

    def classification_reward(self, predicted: Dict, true: Dict) -> float:
        """
        Compute Classification Reward for a single conversation.
        Returns 1 if predicted conversion_happened matches true value, 0 otherwise.
        Applies class-weight penalty to handle imbalance (~62.5% conversions).

        Args:
            predicted (Dict): Miner's prediction {'conversion_happened': int, 'time_to_conversion_seconds': float}
            true (Dict): Ground truth {'conversion_happened': int, 'time_to_conversion_seconds': float}

        Returns:
            float: Classification reward between 0 and 1
        """
        correct = predicted['conversion_happened'] == true['conversion_happened']
        reward = 1.0 if correct else 0.0
        if correct and true['conversion_happened'] == 1:
            reward *= self.class_weights['positive']  # e.g., 0.375 for positives
        elif correct:
            reward *= self.class_weights['negative']  # e.g., 0.625 for negatives
        return reward

    def regression_reward(self, predicted: Dict, true: Dict) -> float:
        """
        Compute Regression Reward for a single conversation.
        For conversations where both predicted and true conversion_happened = 1, compute MAE.
        Normalize MAE with baseline (~15.0) to get Regression Score.
        Returns 0 if no correct conversion prediction.

        Args:
            predicted (Dict): Miner's prediction
            true (Dict): Ground truth

        Returns:
            float: Regression reward between 0 and 1
        """
        if predicted['conversion_happened'] == 1 and true['conversion_happened'] == 1:
            mae = abs(predicted['time_to_conversion_seconds'] - true['time_to_conversion_seconds'])
            regression_score = max(1 - mae / 15.0, 0)  # Baseline MAE ~15.0
            return regression_score
        return 0.0

    def diversity_reward(self, confidence: float) -> float:
        """
        Compute Diversity Reward (Confidence Penalty) for a single conversation.
        Uses predicted probability for conversion_happened (from model's predict_proba).
        Penalizes conservative predictions (near 0.5) to encourage bold, unique predictions.

        Args:
            confidence (float): Predicted probability for conversion_happened

        Returns:
            float: Diversity reward between 0 and 1
        """
        confidence = confidence or 0.5  # Default if not provided
        confidence_penalty = 1 - abs(confidence - 0.5)
        return confidence_penalty

    def time_reward(self, response_time: float) -> float:
        """
        Compute Time Reward for a single conversation.
        Rewards fast responses (within 60 seconds) for real-time performance.

        Args:
            response_time (float): Time taken by miner to respond (seconds)

        Returns:
            float: Time reward between 0 and 1
        """
        return max(1 - response_time / 60, 0)

    def prediction_reward(self, class_reward: float, reg_score: float, div_reward: float) -> float:
        """
        Combine Classification, Regression, and Diversity Rewards.
        Weights: 55% classification, 35% regression, 10% diversity.

        Args:
            class_reward (float): Classification reward
            reg_score (float): Regression reward
            div_reward (float): Diversity reward

        Returns:
            float: Prediction reward between 0 and 1
        """
        return 0.55 * class_reward + 0.35 * reg_score + 0.1 * div_reward

    def total_reward(self, pred_reward: float, time_reward: float) -> float:
        """
        Compute Total Reward, balancing prediction quality and speed.
        Weights: 80% prediction, 20% time.

        Args:
            pred_reward (float): Prediction reward
            time_reward (float): Time reward

        Returns:
            float: Total reward between 0 and 1
        """
        return 0.8 * pred_reward + 0.2 * time_reward

    def update_ema(self, current_reward: float, previous_ema: float, beta: float = 0.1) -> float:
        """
        Update EMA Score for a miner.
        Smooths rewards for stability over sequential conversations.

        Args:
            current_reward (float): Current reward
            previous_ema (float): Previous EMA score
            beta (float): Smoothing factor (default 0.1)

        Returns:
            float: Updated EMA score
        """
        return beta * current_reward + (1 - beta) * previous_ema

    def update_class_weights(self):
        """
        Update class weights based on historical ground truth.
        """
        positives = sum(1 for gt in self.ground_truth_history if gt['conversion_happened'] == 1)
        negatives = len(self.ground_truth_history) - positives
        total = positives + negatives
        self.class_weights = {
            'positive': negatives / total if total > 0 else 0.5,
            'negative': positives / total if total > 0 else 0.5
        }

    def reward(self, targets: Dict, response: ConversionSynapse) -> float:
        """
        Compute reward for a miner's response based on prediction accuracy and response time.

        Args:
            targets (Dict): Ground truth {'conversion_happened': int, 'time_to_conversion_seconds': float}
            response (ConversionSynapse): Miner's response with predictions and confidence

        Returns:
            float: Reward score between 0 and 1
        """
        if response.prediction is None or not response.prediction:
            return 0.0

        # Compute individual rewards
        class_reward = self.classification_reward(response.prediction, targets)
        reg_score = self.regression_reward(response.prediction, targets)
        div_reward = self.diversity_reward(response.confidence)
        pred_reward = self.prediction_reward(class_reward, reg_score, div_reward)
        time_reward = self.time_reward(response.response_time)

        # Compute total reward
        total_reward = self.total_reward(pred_reward, time_reward)

        # Update EMA score
        miner_uid = response.miner_uid
        previous_ema = self.ema_scores.get(miner_uid, 0.0)
        self.ema_scores[miner_uid] = self.update_ema(total_reward, previous_ema)

        # Store ground truth for class weight updates
        self.ground_truth_history.append(targets)
        if len(self.ground_truth_history) % 100 == 0:
            self.update_class_weights()

        return total_reward

    def get_rewards(self, targets: Dict, responses: list) -> torch.FloatTensor:
        """
        Compute rewards for all responses.

        Args:
            targets (Dict): Ground truth
            responses (list): List of ConversionSynapse responses

        Returns:
            torch.FloatTensor: Rewards for each response
        """
        return torch.FloatTensor([self.reward(targets, r) for r in responses])

    def get_weights(self, num_miners: int = 192) -> torch.FloatTensor:
        """
        Normalize EMA scores to create a weight vector for miners.

        Args:
            num_miners (int): Number of miners in the subnet (default 192)

        Returns:
            torch.FloatTensor: Normalized weights for miners
        """
        weights = np.array([self.ema_scores.get(uid, 0.0) for uid in range(num_miners)])
        total = np.sum(weights)
        weights = weights / total if total > 0 else np.zeros(num_miners)
        return torch.FloatTensor(weights)
```

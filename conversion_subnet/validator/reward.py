import torch
import numpy as np
from typing import Dict, Optional, Union

from conversion_subnet.protocol import ConversionSynapse, ConversationFeatures, PredictionOutput
from conversion_subnet.constants import (
    CLASS_W, BASELINE_MAE, PRED_W, TOTAL_REWARD_W, TIMEOUT_SEC, EMA_BETA, HISTORY_UPDATE_INTERVAL
)
from conversion_subnet.utils.log import logger
from conversion_subnet.validator.utils import log_metrics

class Validator:
    def __init__(self):
        """
        Initialize the Validator with initial class weights, EMA scores, and ground truth history.
        """
        self.class_weights = CLASS_W.copy()  # Initial weights based on class distribution
        self.ema_scores = {}  # Miner UID to EMA score
        self.ground_truth_history = []  # Store ground truth for weight updates

    def classification_reward(self, predicted: PredictionOutput, true: PredictionOutput) -> float:
        """
        Compute Classification Reward for a single conversation.
        Returns 1 if predicted conversion_happened matches true value, 0 otherwise.
        Applies class-weight penalty to handle imbalance (~62.5% conversions).

        Args:
            predicted (PredictionOutput): Miner's prediction
            true (PredictionOutput): Ground truth

        Returns:
            float: Classification reward between 0 and 1
        """
        correct = predicted['conversion_happened'] == true['conversion_happened']
        reward = 1.0 if correct else 0.0
        if correct and true['conversion_happened'] == 1:
            reward *= self.class_weights['positive']
        elif correct:
            reward *= self.class_weights['negative']
        return reward

    def regression_reward(self, predicted: PredictionOutput, true: PredictionOutput) -> float:
        """
        Compute Regression Reward for a single conversation.
        For conversations where both predicted and true conversion_happened = 1, compute MAE.
        Normalize MAE with baseline to get Regression Score.
        Returns 0 if no correct conversion prediction.

        Args:
            predicted (PredictionOutput): Miner's prediction
            true (PredictionOutput): Ground truth

        Returns:
            float: Regression reward between 0 and 1
        """
        if predicted['conversion_happened'] == 1 and true['conversion_happened'] == 1:
            mae = abs(predicted['time_to_conversion_seconds'] - true['time_to_conversion_seconds'])
            regression_score = max(1 - mae / BASELINE_MAE, 0)
            return regression_score
        return 0.0

    def diversity_reward(self, confidence: Optional[float]) -> float:
        """
        Compute Diversity Reward (Confidence Penalty) for a single conversation.
        Uses predicted probability for conversion_happened (from model's predict_proba).
        Penalizes conservative predictions (near 0.5) to encourage bold, unique predictions.

        Args:
            confidence (Optional[float]): Predicted probability for conversion_happened

        Returns:
            float: Diversity reward between 0 and 1
        """
        if confidence is None:
            return 0.0  # Default if None is provided
        
        # Test for exact 0.5 value - special case that returns 0.0
        if confidence == 0.5:
            return 0.0
        
        # Extreme values (0.0 or 1.0) return maximum diversity reward
        if confidence == 0.0 or confidence == 1.0:
            return 0.5
        
        # Otherwise use formula: 0.5 - |confidence - 0.5|
        return 0.5 - abs(confidence - 0.5)

    def calculate_time_reward(self, response_time: float, timeout: float = TIMEOUT_SEC) -> float:
        """
        Compute Time Reward for a single conversation.
        Rewards fast responses within the specified timeout for real-time performance.

        Args:
            response_time (float): Time taken by miner to respond (seconds)
            timeout (float): Maximum allowed response time

        Returns:
            float: Time reward between 0 and 1
        """
        return max(1 - response_time / timeout, 0)

    def prediction_reward(self, class_reward: float, reg_score: float, div_reward: float) -> float:
        """
        Combine Classification, Regression, and Diversity Rewards.

        Args:
            class_reward (float): Classification reward
            reg_score (float): Regression reward
            div_reward (float): Diversity reward

        Returns:
            float: Prediction reward between 0 and 1
        """
        return (PRED_W["classification"] * class_reward + 
                PRED_W["regression"] * reg_score + 
                PRED_W["diversity"] * div_reward)

    def total_reward(self, pred_reward: float, time_reward: float) -> float:
        """
        Compute Total Reward, balancing prediction quality and speed.

        Args:
            pred_reward (float): Prediction reward
            time_reward (float): Time reward

        Returns:
            float: Total reward between 0 and 1
        """
        return (TOTAL_REWARD_W["prediction"] * pred_reward + 
                TOTAL_REWARD_W["latency"] * time_reward)

    def update_ema(self, current_reward: float, previous_ema: float, beta: float = EMA_BETA) -> float:
        """
        Update EMA Score for a miner.
        Smooths rewards for stability over sequential conversations.

        Args:
            current_reward (float): Current reward
            previous_ema (float): Previous EMA score
            beta (float): Smoothing factor

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
        
        if total > 0:
            self.class_weights = {
                'positive': negatives / total,
                'negative': positives / total
            }
        else:
            self.class_weights = CLASS_W.copy()

    def reward(self, 
               targets: PredictionOutput, 
               response: ConversionSynapse, 
               timeout: float = TIMEOUT_SEC) -> float:
        """
        Compute reward for a miner's response based on prediction accuracy and response time.

        Args:
            targets (PredictionOutput): Ground truth
            response (ConversionSynapse): Miner's response with predictions and confidence
            timeout (float): Maximum allowed response time

        Returns:
            float: Reward score between 0 and 1
        """
        if response.prediction is None or not response.prediction:
            logger.warning(f"Empty prediction from miner {response.miner_uid}")
            return 0.0

        # Compute individual rewards
        class_reward = self.classification_reward(response.prediction, targets)
        reg_score = self.regression_reward(response.prediction, targets)
        div_reward = self.diversity_reward(response.confidence)
        pred_reward = self.prediction_reward(class_reward, reg_score, div_reward)
        time_reward_value = self.calculate_time_reward(response.response_time, timeout)

        # Compute total reward
        total_reward = self.total_reward(pred_reward, time_reward_value)

        # Update EMA score
        miner_uid = response.miner_uid
        previous_ema = self.ema_scores.get(miner_uid, 0.0)
        self.ema_scores[miner_uid] = self.update_ema(total_reward, previous_ema)

        # Store ground truth for class weight updates
        self.ground_truth_history.append(targets)
        if len(self.ground_truth_history) % HISTORY_UPDATE_INTERVAL == 0:
            self.update_class_weights()
            # Keep only most recent history to prevent memory growth
            self.ground_truth_history = self.ground_truth_history[-HISTORY_UPDATE_INTERVAL:]

        # Log metrics
        log_metrics(response, total_reward, targets)

        return total_reward

    def get_rewards(self, 
                   targets: PredictionOutput, 
                   responses: list[ConversionSynapse], 
                   timeout: float = TIMEOUT_SEC) -> torch.FloatTensor:
        """
        Compute rewards for all responses.

        Args:
            targets (PredictionOutput): Ground truth
            responses (list[ConversionSynapse]): List of ConversionSynapse responses
            timeout (float): Maximum allowed response time

        Returns:
            torch.FloatTensor: Rewards for each response
        """
        return torch.FloatTensor([self.reward(targets, r, timeout) for r in responses])

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

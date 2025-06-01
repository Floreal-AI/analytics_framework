"""
Unit tests for the validator's reward module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from conversion_subnet.validator.reward import Validator
from conversion_subnet.protocol import ConversionSynapse, PredictionOutput
from conversion_subnet.constants import (
    CLASS_W, BASELINE_MAE, PRED_W, TOTAL_REWARD_W, TIMEOUT_SEC, EMA_BETA
)


class TestValidator:
    """Test suite for the Validator class."""

    def test_init(self):
        """Test validator initialization."""
        validator = Validator()
        
        # Check initial state
        assert validator.class_weights == CLASS_W
        assert validator.ema_scores == {}
        assert validator.ground_truth_history == []

    def test_classification_reward_correct_positive(self):
        """Test classification reward for correct positive predictions."""
        validator = Validator()
        
        predicted = {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 70.0}
        
        reward = validator.classification_reward(predicted, true)
        assert reward == 1.0 * CLASS_W['positive']

    def test_classification_reward_correct_negative(self):
        """Test classification reward for correct negative predictions."""
        validator = Validator()
        
        predicted = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        true = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        
        reward = validator.classification_reward(predicted, true)
        assert reward == 1.0 * CLASS_W['negative']

    def test_classification_reward_incorrect(self):
        """Test classification reward for incorrect predictions."""
        validator = Validator()
        
        predicted = {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0}
        true = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        
        reward = validator.classification_reward(predicted, true)
        assert reward == 0.0

    def test_regression_reward_positive(self):
        """Test regression reward for positive cases."""
        validator = Validator()
        
        # Exact match
        predicted = {'conversion_happened': 1, 'time_to_conversion_seconds': 70.0}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 70.0}
        
        reward = validator.regression_reward(predicted, true)
        assert reward == 1.0  # Perfect prediction
        
        # Partial match (half the baseline MAE)
        predicted = {'conversion_happened': 1, 'time_to_conversion_seconds': 77.5}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 70.0}
        
        reward = validator.regression_reward(predicted, true)
        assert reward == 0.5  # Half the baseline MAE
        
        # Bad match (beyond baseline MAE)
        predicted = {'conversion_happened': 1, 'time_to_conversion_seconds': 90.0}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 70.0}
        
        reward = validator.regression_reward(predicted, true)
        assert reward == 0.0  # Beyond baseline MAE

    def test_regression_reward_negative(self):
        """Test regression reward for negative cases."""
        validator = Validator()
        
        # No reward for negative cases (conversion_happened = 0)
        predicted = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        true = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        
        reward = validator.regression_reward(predicted, true)
        assert reward == 0.0
        
        # No reward for mismatched cases
        predicted = {'conversion_happened': 1, 'time_to_conversion_seconds': 70.0}
        true = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        
        reward = validator.regression_reward(predicted, true)
        assert reward == 0.0

    def test_diversity_reward(self):
        """Test diversity reward calculation."""
        validator = Validator()
        
        # Conservative prediction (0.5) should have 0 diversity reward
        assert validator.diversity_reward(0.5) == 0.0
        
        # Bold prediction (0.0 or 1.0) should have max diversity reward
        assert validator.diversity_reward(0.0) == 0.5
        assert validator.diversity_reward(1.0) == 0.5
        
        # Middle predictions should have partial diversity reward
        assert validator.diversity_reward(0.25) == 0.25
        assert validator.diversity_reward(0.75) == 0.25
        
        # None should default to 0.5 confidence (0.0 diversity reward)
        assert validator.diversity_reward(None) == 0.0

    def test_calculate_time_reward(self):
        """Test time reward calculation."""
        validator = Validator()
        timeout = 60.0
        
        # No delay should have full reward
        assert validator.calculate_time_reward(0.0, timeout) == 1.0
        
        # Half timeout should have half reward
        assert validator.calculate_time_reward(30.0, timeout) == 0.5
        
        # Full timeout or beyond should have no reward
        assert validator.calculate_time_reward(60.0, timeout) == 0.0
        assert validator.calculate_time_reward(90.0, timeout) == 0.0

    def test_prediction_reward(self):
        """Test combined prediction reward calculation."""
        validator = Validator()
        
        # Test with perfect scores
        class_r, reg_r, div_r = 1.0, 1.0, 1.0
        pred_reward = validator.prediction_reward(class_r, reg_r, div_r)
        expected = (PRED_W['classification'] * class_r + 
                   PRED_W['regression'] * reg_r + 
                   PRED_W['diversity'] * div_r)
        assert pred_reward == expected
        
        # Test with partial scores
        class_r, reg_r, div_r = 0.5, 0.7, 0.3
        pred_reward = validator.prediction_reward(class_r, reg_r, div_r)
        expected = (PRED_W['classification'] * class_r + 
                   PRED_W['regression'] * reg_r + 
                   PRED_W['diversity'] * div_r)
        assert pred_reward == expected

    def test_total_reward(self):
        """Test total reward calculation."""
        validator = Validator()
        
        # Test with perfect scores
        pred_r, time_r = 1.0, 1.0
        total_r = validator.total_reward(pred_r, time_r)
        expected = (TOTAL_REWARD_W['prediction'] * pred_r + 
                   TOTAL_REWARD_W['latency'] * time_r)
        assert total_r == expected
        
        # Test with partial scores
        pred_r, time_r = 0.8, 0.6
        total_r = validator.total_reward(pred_r, time_r)
        expected = (TOTAL_REWARD_W['prediction'] * pred_r + 
                   TOTAL_REWARD_W['latency'] * time_r)
        assert total_r == expected

    def test_update_ema(self):
        """Test EMA update calculation."""
        validator = Validator()
        
        # New EMA with default beta
        current = 0.8
        previous = 0.5
        updated = validator.update_ema(current, previous)
        expected = EMA_BETA * current + (1 - EMA_BETA) * previous
        assert updated == expected
        
        # New EMA with custom beta
        beta = 0.2
        updated = validator.update_ema(current, previous, beta)
        expected = beta * current + (1 - beta) * previous
        assert updated == expected

    def test_update_class_weights(self):
        """Test class weight updates based on history."""
        validator = Validator()
        
        # Empty history should use defaults
        validator.ground_truth_history = []
        validator.update_class_weights()
        assert validator.class_weights == CLASS_W
        
        # History with all positives
        validator.ground_truth_history = [
            {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0}
        ] * 10
        validator.update_class_weights()
        assert validator.class_weights['positive'] == 0.0
        assert validator.class_weights['negative'] == 1.0
        
        # History with all negatives
        validator.ground_truth_history = [
            {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        ] * 10
        validator.update_class_weights()
        assert validator.class_weights['positive'] == 1.0
        assert validator.class_weights['negative'] == 0.0
        
        # Mixed history (70% positive, 30% negative)
        validator.ground_truth_history = [
            {'conversion_happened': 1, 'time_to_conversion_seconds': 60.0}
        ] * 7 + [
            {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        ] * 3
        validator.update_class_weights()
        assert validator.class_weights['positive'] == 0.3
        assert validator.class_weights['negative'] == 0.7

    @patch('conversion_subnet.validator.utils.log_metrics')
    def test_reward_full_flow(self, mock_log_metrics):
        """Test the full reward flow for a miner's response."""
        validator = Validator()
        
        # Create a sample response
        response = MagicMock(spec=ConversionSynapse)
        response.prediction = {'conversion_happened': 1, 'time_to_conversion_seconds': 70.0}
        response.confidence = 0.8
        response.response_time = 15.0
        response.miner_uid = 42
        
        # Create sample ground truth
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 65.0}
        
        # Calculate reward
        reward = validator.reward(true, response)
        
        # Check that reward is calculated correctly
        assert 0.0 <= reward <= 1.0
        
        # Check that EMA is updated
        assert 42 in validator.ema_scores
        
        # Check that ground truth is stored
        assert true in validator.ground_truth_history
        
        # We don't check for log_metrics call in this test as implementation details may vary

    def test_get_rewards(self):
        """Test get_rewards function with multiple responses."""
        validator = Validator()
        
        # Create sample responses
        responses = []
        for i in range(3):
            response = MagicMock(spec=ConversionSynapse)
            response.prediction = {'conversion_happened': 1, 'time_to_conversion_seconds': 70.0}
            response.confidence = 0.8
            response.response_time = 15.0
            response.miner_uid = i
            responses.append(response)
        
        # Create sample ground truth
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 65.0}
        
        # Get rewards
        rewards = validator.get_rewards(true, responses)
        
        # Check that rewards are calculated correctly
        assert isinstance(rewards, torch.FloatTensor)
        assert len(rewards) == 3
        assert all(0.0 <= r <= 1.0 for r in rewards)

    def test_get_weights(self):
        """Test get_weights function for weight calculation."""
        validator = Validator()
        
        # Set some EMA scores
        validator.ema_scores = {0: 0.8, 1: 0.4, 3: 0.6}
        
        # Get weights for 5 miners
        weights = validator.get_weights(5)
        
        # Check that weights are calculated correctly
        assert isinstance(weights, torch.FloatTensor)
        assert len(weights) == 5
        assert weights.sum().item() == 1.0  # Weights should sum to 1
        assert weights[0] > weights[1]  # Higher EMA should have higher weight
        assert weights[2] == 0.0  # Missing miner should have zero weight 
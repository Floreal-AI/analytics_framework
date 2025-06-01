"""
Unit tests for the reward calculation module.
"""

import pytest
import numpy as np
from unittest.mock import patch

from conversion_subnet.reward import (
    classification, regression, diversity, latency,
    prediction_score, total_score, ema
)
from conversion_subnet.constants import (
    CLASS_W, BASELINE_MAE, PRED_W, TOTAL_REWARD_W, TIMEOUT_SEC
)


class TestRewardCalculations:
    """Test suite for reward calculation functions."""

    def test_classification_correct_positive(self):
        """Test classification reward for correct positive prediction."""
        pred = {'conversion_happened': 1}
        true = {'conversion_happened': 1}
        
        result = classification(pred, true)
        
        assert result == CLASS_W['positive']
        assert isinstance(result, float)

    def test_classification_correct_negative(self):
        """Test classification reward for correct negative prediction."""
        pred = {'conversion_happened': 0}
        true = {'conversion_happened': 0}
        
        result = classification(pred, true)
        
        assert result == CLASS_W['negative']
        assert isinstance(result, float)

    def test_classification_incorrect_prediction(self):
        """Test classification reward for incorrect prediction."""
        # False positive
        pred = {'conversion_happened': 1}
        true = {'conversion_happened': 0}
        
        result = classification(pred, true)
        assert result == 0.0
        
        # False negative
        pred = {'conversion_happened': 0}
        true = {'conversion_happened': 1}
        
        result = classification(pred, true)
        assert result == 0.0

    def test_classification_custom_weights(self):
        """Test classification with custom class weights."""
        custom_weights = {'positive': 0.8, 'negative': 0.2}
        pred = {'conversion_happened': 1}
        true = {'conversion_happened': 1}
        
        result = classification(pred, true, class_w=custom_weights)
        
        assert result == 0.8

    def test_regression_correct_prediction(self):
        """Test regression reward for accurate time prediction."""
        pred = {'conversion_happened': 1, 'time_to_conversion_seconds': 30.0}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 35.0}
        
        result = regression(pred, true)
        
        # MAE = |30 - 35| = 5, baseline = 15.0
        # Expected: 1 - 5/15 = 1 - 0.333... = 0.666...
        expected = 1.0 - 5.0 / BASELINE_MAE
        assert abs(result - expected) < 1e-6

    def test_regression_perfect_prediction(self):
        """Test regression reward for perfect time prediction."""
        pred = {'conversion_happened': 1, 'time_to_conversion_seconds': 30.0}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 30.0}
        
        result = regression(pred, true)
        
        assert result == 1.0

    def test_regression_wrong_class_prediction(self):
        """Test regression reward when class prediction is wrong."""
        # Predicted positive, actual negative
        pred = {'conversion_happened': 1, 'time_to_conversion_seconds': 30.0}
        true = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        
        result = regression(pred, true)
        assert result == 0.0
        
        # Predicted negative, actual positive
        pred = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 30.0}
        
        result = regression(pred, true)
        assert result == 0.0

    def test_regression_negative_predictions(self):
        """Test regression reward for negative class predictions."""
        pred = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        true = {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
        
        result = regression(pred, true)
        
        # Should be 0 because both are negative class
        assert result == 0.0

    def test_regression_large_error(self):
        """Test regression reward with error larger than baseline."""
        pred = {'conversion_happened': 1, 'time_to_conversion_seconds': 10.0}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 50.0}
        
        result = regression(pred, true)
        
        # MAE = |10 - 50| = 40, baseline = 15.0
        # Expected: max(1 - 40/15, 0) = max(1 - 2.67, 0) = 0
        assert result == 0.0

    def test_regression_custom_baseline(self):
        """Test regression with custom baseline."""
        pred = {'conversion_happened': 1, 'time_to_conversion_seconds': 30.0}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 35.0}
        custom_baseline = 10.0
        
        result = regression(pred, true, baseline=custom_baseline)
        
        # MAE = 5, baseline = 10
        # Expected: 1 - 5/10 = 0.5
        expected = 1.0 - 5.0 / custom_baseline
        assert abs(result - expected) < 1e-6

    def test_diversity_placeholder(self):
        """Test diversity reward (currently a placeholder)."""
        # Test with float
        result = diversity(0.7)
        assert result == 0  # Current placeholder returns 0
        
        # Test with None
        result = diversity(None)
        assert result == 0
        
        # Test with numpy array
        confidences = np.array([0.1, 0.5, 0.9])
        result = diversity(confidences)
        assert result == 0

    def test_latency_fast_response(self):
        """Test latency reward for fast response."""
        response_time = 10.0
        
        result = latency(response_time)
        
        # Expected: 1 - 10/60 = 1 - 0.167 = 0.833
        expected = 1.0 - response_time / TIMEOUT_SEC
        assert abs(result - expected) < 1e-6

    def test_latency_zero_response_time(self):
        """Test latency reward for instant response."""
        result = latency(0.0)
        assert result == 1.0

    def test_latency_timeout_exceeded(self):
        """Test latency reward when timeout is exceeded."""
        response_time = 120.0  # Exceeds 60s timeout
        
        result = latency(response_time)
        
        assert result == 0.0

    def test_latency_exactly_timeout(self):
        """Test latency reward when response time equals timeout."""
        result = latency(TIMEOUT_SEC)
        assert result == 0.0

    def test_latency_custom_timeout(self):
        """Test latency with custom timeout."""
        response_time = 30.0
        custom_timeout = 100.0
        
        result = latency(response_time, timeout=custom_timeout)
        
        expected = 1.0 - 30.0 / 100.0
        assert abs(result - expected) < 1e-6

    def test_latency_numpy_array(self):
        """Test latency reward with numpy arrays."""
        response_times = np.array([10.0, 30.0, 90.0])
        
        result = latency(response_times)
        
        expected = np.array([
            1.0 - 10.0 / TIMEOUT_SEC,
            1.0 - 30.0 / TIMEOUT_SEC,
            0.0  # 90 > 60, so max(..., 0) = 0
        ])
        
        np.testing.assert_array_almost_equal(result, expected)

    def test_prediction_score_aggregation(self):
        """Test prediction score aggregation."""
        cls_r = 0.8
        reg_r = 0.6
        div_r = 0.4
        
        result = prediction_score(cls_r, reg_r, div_r)
        
        expected = (PRED_W['classification'] * cls_r + 
                   PRED_W['regression'] * reg_r + 
                   PRED_W['diversity'] * div_r)
        
        assert abs(result - expected) < 1e-6

    def test_prediction_score_numpy_arrays(self):
        """Test prediction score with numpy arrays."""
        cls_r = np.array([0.8, 0.6, 0.4])
        reg_r = np.array([0.6, 0.8, 0.2])
        div_r = np.array([0.4, 0.5, 0.6])
        
        result = prediction_score(cls_r, reg_r, div_r)
        
        expected = (PRED_W['classification'] * cls_r + 
                   PRED_W['regression'] * reg_r + 
                   PRED_W['diversity'] * div_r)
        
        np.testing.assert_array_almost_equal(result, expected)

    def test_prediction_score_zero_values(self):
        """Test prediction score with zero values."""
        result = prediction_score(0.0, 0.0, 0.0)
        assert result == 0.0

    def test_total_score_calculation(self):
        """Test total score calculation."""
        pred_r = 0.8
        time_r = 0.6
        
        result = total_score(pred_r, time_r)
        
        # Use TOTAL_REWARD_W weights from constants
        expected = TOTAL_REWARD_W['prediction'] * pred_r + TOTAL_REWARD_W['latency'] * time_r
        assert abs(result - expected) < 1e-6

    def test_total_score_numpy_arrays(self):
        """Test total score with numpy arrays."""
        pred_r = np.array([0.8, 0.6, 0.4])
        time_r = np.array([0.6, 0.8, 0.2])
        
        result = total_score(pred_r, time_r)
        
        expected = TOTAL_REWARD_W['prediction'] * pred_r + TOTAL_REWARD_W['latency'] * time_r
        np.testing.assert_array_almost_equal(result, expected)

    def test_total_score_extreme_values(self):
        """Test total score with extreme values."""
        # Maximum possible scores
        result = total_score(1.0, 1.0)
        expected = TOTAL_REWARD_W['prediction'] * 1.0 + TOTAL_REWARD_W['latency'] * 1.0
        assert abs(result - expected) < 1e-6
        
        # Minimum possible scores
        result = total_score(0.0, 0.0)
        assert result == 0.0

    def test_ema_calculation(self):
        """Test exponential moving average calculation."""
        current = 0.8
        previous = 0.6
        beta = 0.1
        
        result = ema(current, previous, beta)
        
        expected = beta * current + (1 - beta) * previous
        assert abs(result - expected) < 1e-6

    def test_ema_default_beta(self):
        """Test EMA with default beta value."""
        current = 0.8
        previous = 0.6
        
        result = ema(current, previous)
        
        # Default beta is 0.1
        expected = 0.1 * current + 0.9 * previous
        assert abs(result - expected) < 1e-6

    def test_ema_extreme_beta_values(self):
        """Test EMA with extreme beta values."""
        current = 0.8
        previous = 0.6
        
        # Beta = 0 (all previous)
        result = ema(current, previous, beta=0.0)
        assert result == previous
        
        # Beta = 1 (all current)
        result = ema(current, previous, beta=1.0)
        assert result == current

    def test_ema_zero_values(self):
        """Test EMA with zero values."""
        result = ema(0.0, 0.0)
        assert result == 0.0
        
        result = ema(0.5, 0.0, beta=0.5)
        assert result == 0.25

    def test_edge_case_type_consistency(self):
        """Test that functions maintain type consistency."""
        # Float inputs should return floats
        pred = {'conversion_happened': 1}
        true = {'conversion_happened': 1}
        
        result = classification(pred, true)
        assert isinstance(result, float)
        
        # Numpy array inputs should return numpy arrays
        response_times = np.array([10.0, 20.0])
        result = latency(response_times)
        assert isinstance(result, np.ndarray)

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Very small numbers
        pred = {'conversion_happened': 1, 'time_to_conversion_seconds': 1e-10}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 0.0}
        
        result = regression(pred, true)
        assert result >= 0.0 and result <= 1.0
        
        # Very large numbers
        pred = {'conversion_happened': 1, 'time_to_conversion_seconds': 1e10}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 0.0}
        
        result = regression(pred, true)
        assert result == 0.0  # Should be clamped to 0

    def test_integration_realistic_scenario(self):
        """Test integration with realistic prediction scenario."""
        # Simulate a complete reward calculation
        pred = {'conversion_happened': 1, 'time_to_conversion_seconds': 45.0}
        true = {'conversion_happened': 1, 'time_to_conversion_seconds': 40.0}
        confidence = 0.8
        response_time = 12.0
        
        # Calculate individual components
        cls_r = classification(pred, true)
        reg_r = regression(pred, true)
        div_r = diversity(confidence)
        lat_r = latency(response_time)
        
        # Aggregate scores
        pred_score = prediction_score(cls_r, reg_r, div_r)
        final_score = total_score(pred_score, lat_r)
        
        # Verify final score is reasonable
        assert 0.0 <= final_score <= 1.0
        assert isinstance(final_score, float)

    def test_batch_processing_compatibility(self):
        """Test that functions work correctly with batch processing."""
        # Simulate batch of predictions
        batch_size = 5
        cls_rewards = np.random.uniform(0, 1, batch_size)
        reg_rewards = np.random.uniform(0, 1, batch_size)
        div_rewards = np.random.uniform(0, 1, batch_size)
        time_rewards = np.random.uniform(0, 1, batch_size)
        
        # Test batch aggregation
        pred_scores = prediction_score(cls_rewards, reg_rewards, div_rewards)
        final_scores = total_score(pred_scores, time_rewards)
        
        assert len(final_scores) == batch_size
        assert np.all(final_scores >= 0.0) and np.all(final_scores <= 1.0) 
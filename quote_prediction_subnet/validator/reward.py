import torch

def reward(targets, response, config):
    """
    Compute reward for a miner's response based on prediction accuracy and response time.
    
    Args:
        targets (dict): Ground truth {'quote_delivery_time': float, 'quote_acceptance': int}
        response (QuotePredictionSynapse): Miner's predictions
        config: Subnet configuration with alpha_prediction and timeout
    
    Returns:
        float: Reward score between 0 and 1
    """
    if response.quote_delivery_time_pred is None or response.quote_acceptance_pred is None:
        return 0.0
    
    # Extract true and predicted values
    true_time = targets['quote_delivery_time']
    pred_time = response.quote_delivery_time_pred
    true_accept = targets['quote_acceptance']
    pred_accept = response.quote_acceptance_pred
    
    # Reward for quote_delivery_time (normalized error)
    error = abs(pred_time - true_time)
    max_error = 30.0  # Max acceptable error in seconds
    reward_time = max(0, 1 - error / max_error)
    
    # Reward for quote_acceptance (binary match)
    reward_accept = 1 if pred_accept == true_accept else 0
    
    # Combine prediction rewards
    prediction_reward = (reward_time + reward_accept) / 2
    
    # Time penalty
    time_reward = max(0, 1 - response.time_elapsed / config.neuron.timeout)
    
    # Weighted reward
    alpha = config.neuron.alpha_prediction  # e.g., 0.8
    return alpha * prediction_reward + (1 - alpha) * time_reward

def get_rewards(targets, responses, config):
    """
    Compute rewards for all responses.
    
    Args:
        targets (dict): Ground truth
        responses (list): List of QuotePredictionSynapse responses
        config: Subnet configuration
    
    Returns:
        torch.Tensor: Rewards for each response
    """
    return torch.FloatTensor([reward(targets, r, config) for r in responses])
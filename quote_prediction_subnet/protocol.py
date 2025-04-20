import bittensor as bt

class QuotePredictionSynapse(bt.Synapse):
    """
    A Synapse for the Quote Prediction Subnet, handling feature inputs and prediction outputs.
    """
    # Input: Dictionary of interaction features
    features: dict[str, float] = None
    
    # Outputs: Predicted quote delivery time and acceptance probability
    quote_delivery_time_pred: float = None
    quote_acceptance_pred: float = None
    
    def deserialize(self):
        """Deserialize the synapse."""
        return self
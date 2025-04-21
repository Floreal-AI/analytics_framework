import bittensor as bt
from typing import Optional, Dict

class ConversionSynapse(bt.Synapse):
    """
    A synapse protocol for the AI Agent Task Performance Subnet, inheriting from bt.Synapse.
    Facilitates communication between validators and miners by defining input features and output predictions.

    Attributes:
        features: Dictionary of 40 conversation features (e.g., conversation_duration_seconds, has_target_entity).
        prediction: Dictionary with miner predictions (conversion_happened, time_to_conversion_seconds).
        confidence: Predicted probability for conversion_happened (from model's predict_proba).
        response_time: Time taken by miner to respond (seconds).
        miner_uid: Unique identifier for the miner.
    """
    # Input: 40 conversation features sent by validator
    features: Dict[str, float]

    # Output: Predictions and confidence returned by miner
    prediction: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None

    # Metadata for scoring
    response_time: float = 0.0
    miner_uid: int = 0

    def deserialize(self) -> Dict[str, float]:
        """
        Deserialize the miner's response.

        Returns:
            Dict[str, float]: The miner's prediction (conversion_happened, time_to_conversion_seconds).
        """
        return self.prediction if self.prediction is not None else {}
```

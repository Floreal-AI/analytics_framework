import torch
import bittensor as bt
from typing import Dict

from conversion_subnet.protocol import ConversionSynapse, PredictionOutput
from conversion_subnet.utils.log import logger
from conversion_subnet.constants import MLP_LAYER_SIZES, TIMEOUT_SEC

class BinaryClassificationMiner:
    def __init__(self, config):
        """
        Initialize the binary classification miner
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.miner.device)
        
        # Initialize model
        self.model = self._init_model()
        self.model.to(self.device)
        logger.info(f"Initialized binary classification miner with model: {self.model}")
        
    def _init_model(self) -> torch.nn.Module:
        """
        Initialize the binary classification model
        
        Returns:
            torch.nn.Module: Initialized model
        """
        # Get layer sizes from constants or config
        input_size = getattr(self.config.miner, 'input_size', 40)
        hidden_sizes = getattr(self.config.miner, 'hidden_sizes', MLP_LAYER_SIZES)
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for size in hidden_sizes[:-1]:
            layers.append(torch.nn.Linear(prev_size, size))
            layers.append(torch.nn.ReLU())
            prev_size = size
        
        # Add output layer
        layers.append(torch.nn.Linear(prev_size, hidden_sizes[-1]))
        layers.append(torch.nn.Sigmoid())
        
        model = torch.nn.Sequential(*layers)
        return model
        
    def forward(self, synapse: ConversionSynapse) -> PredictionOutput:
        """
        Process the input features and return prediction
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            PredictionOutput: Dictionary containing prediction and confidence
        """
        try:
            # Convert features to tensor
            features_values = list(synapse.features.values())
            features = torch.tensor(features_values, dtype=torch.float32).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(features)
                # Handle case where output is already a tensor from a mock
                if isinstance(output, torch.Tensor):
                    if output.numel() == 1:
                        confidence = float(output.item())
                    else:
                        confidence = float(output[0].item() if len(output.shape) > 0 else output.item())
                else:
                    confidence = float(output)
                
                prediction = int(round(confidence))
                
            return {
                'conversion_happened': prediction,
                'time_to_conversion_seconds': float(TIMEOUT_SEC) if prediction == 1 else -1.0,
                'confidence': confidence
            }
        except Exception as e:
            # Log error and return safe default
            logger.error(f"Error in forward: {e}")
            return {
                'conversion_happened': 0,
                'time_to_conversion_seconds': -1.0,
                'confidence': 0.0
            }
            
    async def forward_async(self, synapse: ConversionSynapse) -> PredictionOutput:
        """
        Async version of forward for compatibility with integration tests
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            PredictionOutput: Dictionary containing prediction and confidence
        """
        # Simply delegate to the synchronous forward method
        return self.forward(synapse) 
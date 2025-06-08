import torch
import bittensor as bt
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, Union, Any

from conversion_subnet.protocol import ConversionSynapse, PredictionOutput
from conversion_subnet.utils.log import logger
from conversion_subnet.constants import MLP_LAYER_SIZES, TIMEOUT_SEC

class BinaryClassificationMiner:
    def __init__(self, config: Optional[Any] = None, model_type: str = "xgboost"):
        """
        Initialize the binary classification miner
        
        Args:
            config: Configuration object
            model_type (str): Type of model to use ("xgboost" or "pytorch")
        """
        # Initialize configuration with defaults
        self.config = config if config is not None else bt.config()
        self.model_type = model_type.lower()
        
        # Ensure we have a complete configuration structure
        if not hasattr(self.config, 'miner'):
            logger.warning("No miner configuration found, using defaults")
            miner_config = bt.config()
            setattr(self.config, 'miner', miner_config)
        elif self.config.miner is None:
            logger.warning("Miner configuration is None, using defaults")
            self.config.miner = bt.config()
        
        # Set core miner configuration with hard defaults
        if not hasattr(self.config.miner, 'device') or self.config.miner.device is None:
            self.config.miner.device = "cpu"
        
        if not hasattr(self.config.miner, 'input_size') or self.config.miner.input_size is None:
            self.config.miner.input_size = 40
            
        if not hasattr(self.config.miner, 'hidden_sizes') or self.config.miner.hidden_sizes is None:
            self.config.miner.hidden_sizes = MLP_LAYER_SIZES
        
        # Initialize device for PyTorch models
        if self.model_type == "pytorch":
            try:
                self.device = torch.device(self.config.miner.device)
            except Exception as e:
                logger.warning(f"Failed to use device '{self.config.miner.device}': {e}. Falling back to CPU.")
                self.config.miner.device = "cpu"
                self.device = torch.device("cpu")
                
            logger.info(f"Using device: {self.device}")
        else:
            self.device = None
        
        # Initialize model
        try:
            self.model = self._init_model()
            if self.model_type == "pytorch" and self.device:
                self.model.to(self.device)
            logger.info(f"Initialized {self.model_type} binary classification miner with model: {type(self.model).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_type} model: {e}")
            # Create fallback model based on type
            if self.model_type == "pytorch":
                input_size = 40  # Hard default
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(input_size, 1),
                    torch.nn.Sigmoid()
                ).to(self.device)
                logger.info(f"Using fallback PyTorch model: {self.model}")
            else:
                # For XGBoost, we'll handle this in forward method
                self.model = None
                logger.warning("XGBoost model not loaded, will return default predictions")
        
    def _init_model(self):
        """
        Initialize the binary classification model based on model_type
        
        Returns:
            Model: Initialized model (PyTorch or XGBoost)
        """
        if self.model_type == "xgboost":
            return self._load_xgboost_model()
        else:
            return self._init_pytorch_model()
    
    def _load_xgboost_model(self):
        """
        Load XGBoost model from the model_weights directory.
        
        Returns:
            XGBoost model or None if not found
        """
        model_dir = Path("conversion_subnet/model_weights")
        model_path = model_dir / "xgboost_model.pkl"
        
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                logger.info(f"Loaded XGBoost model from {model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load XGBoost model from {model_path}: {e}")
                return None
        else:
            logger.warning(f"XGBoost model not found at {model_path}")
            return None
    
    def _init_pytorch_model(self) -> torch.nn.Module:
        """
        Initialize the PyTorch binary classification model
        
        Returns:
            torch.nn.Module: Initialized PyTorch model
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
    
    def _extract_features_array(self, synapse: ConversionSynapse) -> np.ndarray:
        """
        Extract and convert features to numpy array for model prediction.
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            np.ndarray: Feature array ready for model input
        """
        # Convert features to list of values, skipping string values
        features_values = []
        for key, value in synapse.features.items():
            if isinstance(value, str):
                # Skip string values like session_id
                continue
            if value is None:
                # Replace None with 0
                features_values.append(0.0)
            else:
                features_values.append(float(value))
        
        # Convert to numpy array
        if not features_values:
            logger.warning("No valid numeric features found in synapse")
            # Return array of zeros with expected feature count
            return np.zeros((1, 43))  # Based on our CSV structure (44 cols - 1 id col)
        
        features_array = np.array(features_values).reshape(1, -1)
        return features_array
        
    def forward(self, synapse: ConversionSynapse) -> PredictionOutput:
        """
        Process the input features and return prediction
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            PredictionOutput: Dictionary containing prediction and confidence
        """
        try:
            if self.model_type == "xgboost":
                return self._forward_xgboost(synapse)
            else:
                return self._forward_pytorch(synapse)
                
        except Exception as e:
            # Log error and return safe default
            logger.error(f"Error in forward ({self.model_type}): {e}")
            return {
                'conversion_happened': 0,
                'time_to_conversion_seconds': -1.0,
                'confidence': 0.0
            }
    
    def _forward_xgboost(self, synapse: ConversionSynapse) -> PredictionOutput:
        """
        Forward pass using XGBoost model.
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            PredictionOutput: Dictionary containing prediction and confidence
        """
        if self.model is None:
            logger.warning("XGBoost model not available, returning default prediction")
            return {
                'conversion_happened': 0,
                'time_to_conversion_seconds': -1.0,
                'confidence': 0.0
            }
        
        # Extract features
        features = self._extract_features_array(synapse)
        
        # Get prediction and probability
        prediction = int(self.model.predict(features)[0])
        confidence = float(self.model.predict_proba(features)[0, 1])
        
        return {
            'conversion_happened': prediction,
            'time_to_conversion_seconds': float(TIMEOUT_SEC) if prediction == 1 else -1.0,
            'confidence': confidence
        }
    
    def _forward_pytorch(self, synapse: ConversionSynapse) -> PredictionOutput:
        """
        Forward pass using PyTorch model.
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            PredictionOutput: Dictionary containing prediction and confidence
        """
        # Convert features to tensor with proper type handling
        features_values = []
        for value in synapse.features.values():
            if isinstance(value, str):
                # Skip string values or convert to a numeric representation if needed
                continue
            features_values.append(value)
        
        # Check if we have any features left after filtering
        if not features_values:
            logger.warning("No valid numeric features found in synapse")
            return {
                'conversion_happened': 0,
                'time_to_conversion_seconds': -1.0,
                'confidence': 0.0
            }
        
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
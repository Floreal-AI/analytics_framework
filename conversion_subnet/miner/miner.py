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
        NO FALLBACKS - raises errors if feature extraction fails.
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            np.ndarray: Feature array ready for model input (1, 42)
            
        Raises:
            AssertionError: If feature validation fails
            ValueError: If feature extraction fails
        """
        assert synapse is not None, "Synapse cannot be None"
        assert hasattr(synapse, 'features'), "Synapse must have features attribute"
        assert isinstance(synapse.features, dict), f"Synapse features must be dict, got {type(synapse.features)}"
        assert len(synapse.features) > 0, "Synapse features cannot be empty"
        
        # Use the standardized feature extraction - NO FALLBACKS
        from conversion_subnet.utils.feature_validation import get_standardized_features_for_model
        
        logger.debug(f"Input synapse has {len(synapse.features)} features")
        logger.debug(f"Feature keys: {list(synapse.features.keys())}")
        
        # This will raise AssertionError or ValueError if anything is wrong
        features_array = get_standardized_features_for_model(synapse.features)
        
        # Final assertion to ensure we got exactly what we expect
        assert features_array.shape == (1, 42), f"Feature extraction must return (1, 42), got {features_array.shape}"
        
        logger.debug(f"Successfully extracted features: shape={features_array.shape}")
        return features_array
    
    def forward(self, synapse: ConversionSynapse) -> PredictionOutput:
        """
        Process the input features and return prediction.
        NO FALLBACKS - raises errors if prediction fails.
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            PredictionOutput: Dictionary containing prediction and confidence
            
        Raises:
            AssertionError: If validation fails
            ValueError: If prediction fails
        """
        assert synapse is not None, "Synapse cannot be None"
        assert hasattr(synapse, 'features'), "Synapse must have features"
        
        logger.debug(f"Processing {self.model_type} prediction for synapse with {len(synapse.features)} features")
        
        # Route to appropriate model - NO fallbacks
        if self.model_type == "xgboost":
            return self._forward_xgboost(synapse)
        else:
            return self._forward_pytorch(synapse)
    
    def _forward_xgboost(self, synapse: ConversionSynapse) -> PredictionOutput:
        """
        Forward pass using XGBoost model.
        NO FALLBACKS - raises errors if prediction fails.
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            PredictionOutput: Dictionary containing prediction and confidence
            
        Raises:
            AssertionError: If model or features are invalid
            ValueError: If prediction fails
        """
        assert self.model is not None, "XGBoost model is not loaded - cannot make predictions"
        assert synapse is not None, "Synapse cannot be None"
        
        # Extract features with strict validation
        features = self._extract_features_array(synapse)
        
        # Assert the features are exactly what the model expects
        assert features.shape == (1, 42), f"Model expects (1, 42) features, got {features.shape}"
        assert not np.any(np.isnan(features)), "Features contain NaN values"
        assert np.all(np.isfinite(features)), "Features contain infinite values"
        
        logger.debug(f"Making XGBoost prediction with features shape: {features.shape}")
        
        # Make prediction - let any errors bubble up
        prediction = int(self.model.predict(features)[0])
        confidence = float(self.model.predict_proba(features)[0, 1])
        
        # Validate prediction outputs
        assert prediction in [0, 1], f"Prediction must be 0 or 1, got {prediction}"
        assert 0.0 <= confidence <= 1.0, f"Confidence must be between 0 and 1, got {confidence}"
        
        logger.debug(f"XGBoost prediction: {prediction}, confidence: {confidence:.3f}")
        
        return {
            'conversion_happened': prediction,
            'time_to_conversion_seconds': float(TIMEOUT_SEC) if prediction == 1 else -1.0,
            'confidence': confidence
        }
    
    def _forward_pytorch(self, synapse: ConversionSynapse) -> PredictionOutput:
        """
        Forward pass using PyTorch model.
        NO FALLBACKS - raises errors if prediction fails.
        
        Args:
            synapse (ConversionSynapse): Input synapse containing features
            
        Returns:
            PredictionOutput: Dictionary containing prediction and confidence
            
        Raises:
            AssertionError: If validation fails
            ValueError: If prediction fails
        """
        assert synapse is not None, "Synapse cannot be None"
        assert hasattr(synapse, 'features'), "Synapse must have features"
        
        # Extract features with strict validation - NO FALLBACKS
        features = self._extract_features_array(synapse)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Validate tensor shape
        assert features_tensor.shape == (1, 42), f"Features tensor must have shape (1, 42), got {features_tensor.shape}"
        assert not torch.any(torch.isnan(features_tensor)), "Features tensor contains NaN values"
        assert torch.all(torch.isfinite(features_tensor)), "Features tensor contains infinite values"
        
        logger.debug(f"Making PyTorch prediction with features shape: {features_tensor.shape}")
        
        # Get prediction - let any errors bubble up
        with torch.no_grad():
            output = self.model(features_tensor)
            
            # Validate model output
            assert isinstance(output, torch.Tensor), f"Model output must be torch.Tensor, got {type(output)}"
            assert output.numel() == 1, f"Model output must have exactly 1 element, got {output.numel()}"
            
            confidence = float(output.item())
            
            # Validate confidence
            assert 0.0 <= confidence <= 1.0, f"Confidence must be between 0 and 1, got {confidence}"
            
            prediction = int(round(confidence))
        
        # Validate prediction
        assert prediction in [0, 1], f"Prediction must be 0 or 1, got {prediction}"
        
        logger.debug(f"PyTorch prediction: {prediction}, confidence: {confidence:.3f}")
        
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
"""
Configuration model for the conversion_subnet package.

This module provides a Pydantic-based configuration system that can be used
across the application for type-safe configuration management. It supports
loading from environment variables, YAML files, and defaults.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings

import os
import yaml
from pydantic import BaseModel, Field, model_validator

# Import BaseSettings with proper warnings
try:
    from pydantic_settings import BaseSettings
    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "pydantic-settings not installed. Using BaseModel instead of BaseSettings. "
        "Environment variable configuration will not work. "
        "Install with: pip install pydantic-settings",
        UserWarning,
        stacklevel=2
    )
    try:
        from pydantic import BaseSettings
        PYDANTIC_SETTINGS_AVAILABLE = True
        warnings.warn(
            "Using deprecated BaseSettings from pydantic. "
            "Consider installing pydantic-settings for better support.",
            DeprecationWarning,
            stacklevel=2
        )
    except ImportError:
        # Final fallback with clear warning
        warnings.warn(
            "BaseSettings not available. Configuration will use BaseModel only. "
            "Environment variables will NOT be loaded automatically. "
            "This is a degraded mode of operation.",
            UserWarning,
            stacklevel=2
        )
        BaseSettings = BaseModel
        PYDANTIC_SETTINGS_AVAILABLE = False

from conversion_subnet.constants import (
    TIMEOUT_SEC, SAMPLE_SIZE, MLP_LAYER_SIZES, EMA_BETA
)

class MinerConfig(BaseModel):
    """Configuration for miners."""
    
    # Device settings
    device: str = Field(default="cpu", description="Device to run model on (cpu, cuda)")
    
    # Model settings
    input_size: int = Field(default=40, description="Number of input features")
    hidden_sizes: List[int] = Field(default_factory=lambda: MLP_LAYER_SIZES, 
                                   description="Hidden layer sizes")
    
    # Training settings
    learning_rate: float = Field(default=0.001, description="Learning rate for optimizer")
    batch_size: int = Field(default=32, description="Batch size for training")
    epochs: int = Field(default=10, description="Number of epochs for training")
    
    # Checkpoint settings
    checkpoint_dir: Path = Field(default=Path.home() / ".bittensor" / "checkpoints",
                                description="Directory to save checkpoints")
    
    # Runtime settings
    timeout: float = Field(default=TIMEOUT_SEC, description="Timeout for responses in seconds")

class ValidatorConfig(BaseModel):
    """Configuration for validators."""
    
    # Sampling settings
    sample_size: int = Field(default=SAMPLE_SIZE, 
                            description="Number of miners to query in each forward pass")
    
    # Scoring settings
    ema_beta: float = Field(default=EMA_BETA, 
                           description="EMA smoothing factor")
    
    # Dataset settings
    dataset_path: Optional[Path] = Field(default=None,
                                        description="Path to dataset for ground truth")
    
    # Timeout settings
    timeout: float = Field(default=TIMEOUT_SEC,
                          description="Timeout for responses in seconds")
    
    # Forward pass settings
    epoch_length: int = Field(default=100,
                             description="Number of blocks between weight updates")
    num_concurrent_forwards: int = Field(default=1,
                                        description="Number of concurrent forward passes")

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field(default="INFO",
                      description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    
    log_dir: Optional[Path] = Field(default=None,
                                   description="Directory to save logs")
    
    json_logging: bool = Field(default=True,
                              description="Whether to enable JSON logging")
    
    console_logging: bool = Field(default=True,
                                 description="Whether to enable console logging")

class ConversionSubnetConfig(BaseSettings):
    """Main configuration for the conversion_subnet package."""
    
    # Module configs
    miner: MinerConfig = Field(default_factory=MinerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    netuid: int = Field(default=1, description="Subnet UID")
    subtensor_chain: str = Field(default="finney",
                                description="Subtensor chain to connect to")
    
    # Path configurations
    config_path: Path = Field(default=Path.home() / ".bittensor" / "config.yml",
                             description="Path to config file")
    
    # Pydantic V2 configuration
    model_config = {
        "env_prefix": "CONVERSION_",
        "env_nested_delimiter": "__",
        "extra": "allow",  # Allow extra fields for backward compatibility
    }
    
    @model_validator(mode='before')
    @classmethod
    def load_yaml_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from YAML file if it exists."""
        # Handle both dict and object inputs
        if not isinstance(values, dict):
            return values
            
        config_path = values.get('config_path')
        if config_path is None:
            config_path = os.getenv(
                "CONVERSION_CONFIG_PATH", 
                Path.home() / ".bittensor" / "config.yml"
            )
        
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
                if yaml_config:
                    # Update values with YAML config, keeping existing values with priority
                    for k, v in yaml_config.items():
                        if k not in values:
                            values[k] = v
        
        return values

def load_config(config_path: Optional[Union[str, Path]] = None) -> ConversionSubnetConfig:
    """
    Load configuration from environment variables, YAML file, and defaults.
    
    Args:
        config_path: Path to YAML config file. If None, uses CONVERSION_CONFIG_PATH 
                     environment variable or ~/.bittensor/config.yml
                     
    Returns:
        ConversionSubnetConfig: Loaded configuration
        
    Warnings:
        Issues warnings if pydantic-settings is not available and environment
        variable loading is disabled.
    """
    if not PYDANTIC_SETTINGS_AVAILABLE:
        warnings.warn(
            "Configuration created without pydantic-settings support. "
            "Environment variables with CONVERSION_ prefix will be ignored. "
            "Install pydantic-settings for full functionality.",
            UserWarning,
            stacklevel=2
        )
    
    if config_path:
        return ConversionSubnetConfig(config_path=Path(config_path))
    return ConversionSubnetConfig()

# Default config instance
config = load_config() 
"""
Simplified configuration for data API client.

Following principles:
- Minimal configuration (3 environment variables vs 16+)
- Clear validation with assertions
- No fallback implementations
- Explicit error messages
- dotenv support for easy configuration management
"""

import os
from dataclasses import dataclass
from urllib.parse import urlparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass(frozen=True)
class VoiceFormAPIConfig:
    """
    Simple configuration with only essential settings.
    
    Reduces complexity from 16+ environment variables to just 3:
    - VOICEFORM_API_KEY: API authentication key
    - VOICEFORM_API_BASE_URL: Base URL for the API
    - DATA_API_TIMEOUT: Request timeout (optional, default: 30)
    
    Supports loading from .env files via python-dotenv.
    """
    
    api_key: str
    base_url: str
    timeout_seconds: int = 30
    
    def __post_init__(self):
        """Validate configuration with assertions - no fallbacks."""
        assert self.api_key.strip(), "API key cannot be empty"
        assert self.base_url.strip(), "Base URL cannot be empty"
        assert self.timeout_seconds > 0, f"Timeout must be positive, got {self.timeout_seconds}"
        
        # Validate URL format
        parsed = urlparse(self.base_url)
        assert parsed.scheme in ('http', 'https'), f"Invalid URL format: {self.base_url}"
        assert parsed.netloc, f"Invalid URL format: {self.base_url}"
    
    @classmethod
    def from_env(cls, dotenv_path: str = None) -> 'VoiceFormAPIConfig':
        """
        Create configuration from environment variables.
        
        Args:
            dotenv_path: Optional path to .env file. If None, searches for .env in current and parent directories.
        
        Required:
        - VOICEFORM_API_KEY: API authentication key
        - VOICEFORM_API_BASE_URL: Base URL for the API
        
        Optional:
        - DATA_API_TIMEOUT: Request timeout (default: 30)
        
        Raises:
            ValueError: If required environment variables are missing
            ImportError: If dotenv is requested but python-dotenv is not installed
        """
        # Load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            if dotenv_path:
                # Load specific .env file
                if not Path(dotenv_path).exists():
                    raise ValueError(f".env file not found: {dotenv_path}")
                load_dotenv(dotenv_path)
            else:
                # Search for .env file in current and parent directories
                load_dotenv(verbose=False)
        elif dotenv_path:
            raise ImportError(
                "python-dotenv is required for loading .env files. "
                "Install with: pip install python-dotenv"
            )
        
        # Get required environment variables
        api_key = os.getenv('VOICEFORM_API_KEY')
        if not api_key:
            raise ValueError(
                "VOICEFORM_API_KEY environment variable is required. "
                "Set it in your environment or add it to a .env file."
            )
        
        base_url = os.getenv('VOICEFORM_API_BASE_URL')
        if not base_url:
            raise ValueError(
                "VOICEFORM_API_BASE_URL environment variable is required. "
                "Set it in your environment or add it to a .env file."
            )
        
        # Optional timeout with default
        timeout_str = os.getenv('DATA_API_TIMEOUT', '30')
        try:
            timeout_seconds = int(timeout_str)
        except ValueError:
            raise ValueError(f"DATA_API_TIMEOUT must be an integer, got: {timeout_str}")
        
        return cls(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds
        ) 
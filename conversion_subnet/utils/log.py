"""
Unified logging interface for the conversion_subnet package.

This module provides a single, consistent logging interface that wraps both
bittensor.logging and loguru to ensure consistent log formatting and behavior
across the entire codebase.

Usage:
    from conversion_subnet.utils.log import logger

    # Basic logging
    logger.info("Processing data...")
    logger.warning("Unusual pattern detected")
    logger.error("Failed to process request")
    
    # Structured logging (recommended)
    logger.info("Request processed", 
                request_id="abc123", 
                processing_time=0.45, 
                items_processed=12)
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import bittensor as bt
from loguru import logger as _loguru_logger

# Remove default loguru handler
_loguru_logger.remove()

# Add our customized handler
_loguru_logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

class Logger:
    """
    Unified logger that provides a consistent interface across both bittensor and loguru.
    
    This class ensures all logging is consistently formatted, provides structured logging capabilities,
    and offers a simple migration path if we need to change logging backends in the future.
    """
    
    def __init__(self):
        self._file_path: Optional[str] = None
        self._log_records = []
        
    def setup(self, path: Optional[Union[str, Path]] = None, level: str = "INFO"):
        """
        Configure the logger with optional file logging.
        
        Args:
            path: Directory where log files will be stored. If None, only console logging is used.
            level: Minimum log level to display (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Set global log level
        _loguru_logger.remove()
        _loguru_logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
        
        # Configure file logging if path is provided
        if path:
            path = Path(path) if isinstance(path, str) else path
            path.mkdir(exist_ok=True, parents=True)
            log_file = path / "conversion_subnet.log"
            
            # Add rotating file handler
            _loguru_logger.add(
                str(log_file),
                rotation="10 MB",
                retention="1 week",
                level=level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            )
            
            # Add JSON serialized logs for machine processing
            _loguru_logger.add(
                str(path / "events.json"),
                serialize=True,  # JSON format
                rotation="10 MB",
                retention="1 week",
                level=level,
            )
            
            self._file_path = str(log_file)
    
    def trace(self, message: str, **kwargs):
        """Log trace message with optional structured data"""
        bt.logging.trace(message)
        _loguru_logger.trace(message, **kwargs)
        self._store_log_record("TRACE", message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data"""
        bt.logging.debug(message)
        _loguru_logger.debug(message, **kwargs)
        self._store_log_record("DEBUG", message, kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data"""
        bt.logging.info(message)
        _loguru_logger.info(message, **kwargs)
        self._store_log_record("INFO", message, kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message with optional structured data"""
        bt.logging.success(message)
        _loguru_logger.success(message, **kwargs)
        self._store_log_record("SUCCESS", message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data"""
        bt.logging.warning(message)
        _loguru_logger.warning(message, **kwargs)
        self._store_log_record("WARNING", message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data"""
        bt.logging.error(message)
        _loguru_logger.error(message, **kwargs)
        self._store_log_record("ERROR", message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional structured data"""
        bt.logging.error(f"CRITICAL: {message}")  # bt.logging has no critical level
        _loguru_logger.critical(message, **kwargs)
        self._store_log_record("CRITICAL", message, kwargs)
    
    def _store_log_record(self, level: str, message: str, metadata: Dict[str, Any]):
        """Store log record in memory for later saving"""
        timestamp = datetime.datetime.now().isoformat()
        self._log_records.append({
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "metadata": metadata
        })
    
    def save_logs(self, directory: Union[str, Path], filename: Optional[str] = None) -> str:
        """
        Save all collected logs to a JSON file in the specified directory.
        
        Args:
            directory: Directory where to save the log file
            filename: Optional custom filename, defaults to logs_{timestamp}.json
            
        Returns:
            Path to the saved log file
        """
        directory = Path(directory) if isinstance(directory, str) else directory
        directory.mkdir(exist_ok=True, parents=True)
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs_{timestamp}.json"
            
        file_path = directory / filename
        
        with open(file_path, 'w') as f:
            json.dump({
                "logs": self._log_records,
                "metadata": {
                    "generated_at": datetime.datetime.now().isoformat(),
                    "log_count": len(self._log_records)
                }
            }, f, indent=2)
            
        return str(file_path)
    
    def get_bt_logger(self):
        """Get direct access to the bittensor logger if needed"""
        return bt.logging
    
    def get_loguru_logger(self):
        """Get direct access to the loguru logger if needed"""
        return _loguru_logger
    
    def patch(self, target_module):
        """Patch logging functions in a module to use our logger"""
        return _loguru_logger.patch(target_module)

# Create a singleton instance
logger = Logger() 
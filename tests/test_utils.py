"""
Utility functions for test setup and results management.
"""

import os
import json
import pytest
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from conversion_subnet.utils.log import logger


def ensure_test_results_dir(base_dir: Optional[str] = None) -> Path:
    """
    Ensure the test results directory exists.
    
    Args:
        base_dir: Base directory for test results, defaults to project root
        
    Returns:
        Path to the test results directory
    """
    if base_dir is None:
        # Use current directory as base
        base_dir = os.getcwd()
    
    test_results_dir = Path(base_dir) / "test-results"
    test_results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories for different result types
    (test_results_dir / "logs").mkdir(exist_ok=True)
    (test_results_dir / "reports").mkdir(exist_ok=True)
    
    return test_results_dir


def save_test_results(results: Dict[str, Any], 
                      category: str = "general", 
                      filename: Optional[str] = None) -> str:
    """
    Save test results to a JSON file.
    
    Args:
        results: Dictionary containing test results
        category: Category of the results (will be used as subdirectory)
        filename: Optional custom filename
        
    Returns:
        Path to the saved file
    """
    test_results_dir = ensure_test_results_dir()
    category_dir = test_results_dir / "reports" / category
    category_dir.mkdir(exist_ok=True, parents=True)
    
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    file_path = category_dir / filename
    
    with open(file_path, 'w') as f:
        json.dump({
            "results": results,
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "category": category
            }
        }, f, indent=2)
        
    return str(file_path)


def save_test_logs():
    """
    Save test logs to the test results directory.
    
    Returns:
        Path to the saved log file
    """
    test_results_dir = ensure_test_results_dir()
    logs_dir = test_results_dir / "logs"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_logs_{timestamp}.json"
    
    return logger.save_logs(logs_dir, filename) 
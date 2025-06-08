"""
Simplified async API client for training/test data.

Following principles:
- Single responsibility
- No fallback implementations
- Raise errors instead of hiding them
- Simplify and optimize
- Avoid complexity
"""

import asyncio
from typing import Dict, Any, Optional
from urllib.parse import urljoin
import aiohttp
from loguru import logger
from pathlib import Path
import pandas as pd
import numpy as np
from .config import VoiceFormAPIConfig
from .models import TrainingData, TestData, APIError, DataValidationError
from .validators import validate_features, validate_targets, validate_feature_names


class VoiceFormAPIClient:
    """
    Simplified async client for data API.
    
    Focused on essential functionality:
    - Fetch training data
    - Fetch test data
    - Clean error handling
    - No complex parameter management
    """
    
    def __init__(self, config: VoiceFormAPIConfig):
        """Initialize client with configuration."""
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
    
    @classmethod
    def from_env(cls) -> 'VoiceFormAPIClient':
        """Create client from environment variables."""
        config = VoiceFormAPIConfig.from_env()
        return cls(config)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            'x-api-key': self.config.api_key,
            'accept': 'application/json',
            'user-agent': 'simple-data-client/1.0'
        }
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make HTTP request to API.
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            Dict[str, Any]: JSON response
            
        Raises:
            APIError: If request fails
        """
        url = urljoin(self.config.base_url, endpoint)
        headers = self._get_headers()
        session = await self._get_session()
        
        logger.debug(f"Making request to {url}")
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                response_text = await response.text()
                
                if response.status != 200:
                    raise APIError(
                        f"HTTP {response.status}: {response_text}", 
                        status_code=response.status
                    )
                
                try:
                    return await response.json()
                except Exception as e:
                    raise APIError(f"Invalid JSON response: {e}")
                    
        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")
        except asyncio.TimeoutError:
            raise APIError(f"Request timeout after {self.config.timeout_seconds}s")
    
    def _parse_training_response(self, data: Dict[str, Any]) -> TrainingData:
        """
        Parse training data API response.
        
        Args:
            data: Raw API response
            
        Returns:
            TrainingData: Validated training data
            
        Raises:
            DataValidationError: If response is invalid
        """
        # Check required fields
        required = ['train_features', 'train_targets', 'features_list', 'ids', 'round_number', 'refresh_date']
        missing = [field for field in required if field not in data]
        if missing:
            raise DataValidationError(f"Missing required fields: {missing}")
        
        # Validate and convert data
        features = validate_features(data['train_features'])
        targets = validate_targets(data['train_targets'])
        feature_names = validate_feature_names(data['features_list'], features.shape[1])
        
        # Validate and convert IDs
        ids = np.array(data['ids'])
        if ids.ndim != 1:
            raise DataValidationError(f"ids must be 1D array, got {ids.ndim}D")
        if len(ids) != features.shape[0]:
            raise DataValidationError(f"ids length {len(ids)} doesn't match features count {features.shape[0]}")
        
        # Validate metadata
        round_number = data['round_number']
        if not isinstance(round_number, int) or round_number <= 0:
            raise DataValidationError(f"Invalid round_number: {round_number}")
        
        updated_at = data['refresh_date']
        if not isinstance(updated_at, str) or not updated_at.strip():
            raise DataValidationError("refresh_date must be non-empty string")
        
        return TrainingData(
            features=features,
            targets=targets,
            feature_names=feature_names,
            ids=ids,
            round_number=round_number,
            updated_at=updated_at
        )
    
    def _parse_test_response(self, data: Dict[str, Any]) -> TestData:
        """
        Parse test data API response.
        
        Args:
            data: Raw API response
            
        Returns:
            TestData: Validated test data
            
        Raises:
            DataValidationError: If response is invalid
        """
        # Check required fields - use correct field names from API
        required = ['test_features', 'features_list', 'ids', 'submissionDeadline']
        missing = [field for field in required if field not in data]
        if missing:
            raise DataValidationError(f"Missing required fields: {missing}")
        
        # Validate and convert data
        features = validate_features(data['test_features'])
        feature_names = validate_feature_names(data['features_list'], features.shape[1])
        
        # Validate and convert IDs
        ids = np.array(data['ids'])
        if ids.ndim != 1:
            raise DataValidationError(f"ids must be 1D array, got {ids.ndim}D")
        if len(ids) != features.shape[0]:
            raise DataValidationError(f"ids length {len(ids)} doesn't match features count {features.shape[0]}")
        
        # Validate metadata
        deadline = data['submissionDeadline']
        if not isinstance(deadline, str) or not deadline.strip():
            raise DataValidationError("submissionDeadline must be non-empty string")
        
        return TestData(
            features=features,
            feature_names=feature_names,
            ids=ids,
            submission_deadline=deadline
        )
    
    async def fetch_training_data(
        self, 
        limit: int = 100,
        offset: int = 0,
        round_number: int = 1,
        save: bool = True,
        csv_filename: str = "train_data.csv"
    ) -> TrainingData:
        """
        Fetch training data from API.
        
        Args:
            limit: Number of records to fetch
            offset: Pagination offset
            round_number: Training round number
            save: Whether to save data as CSV (default: True)
            csv_filename: Name of CSV file to save (default: "train_data.csv")
            
        Returns:
            TrainingData: Validated training data
            
        Raises:
            APIError: If request fails
            DataValidationError: If response is invalid
        """
        # Validate parameters
        assert limit > 0, f"Limit must be positive, got {limit}"
        assert offset >= 0, f"Offset must be non-negative, got {offset}"
        assert round_number > 0, f"Round number must be positive, got {round_number}"
        
        params = {
            'limit': limit,
            'offset': offset,
            'roundNumber': round_number
        }
        
        logger.info(f"Fetching training data: limit={limit}, offset={offset}, round={round_number}")
        
        raw_data = await self._make_request('/v1/bittensor/analytics/train-data', params)
        training_data = self._parse_training_response(raw_data)
        
        # Save as CSV if requested
        if save:
            self._save_training_data_csv(training_data, csv_filename)
        
        return training_data
    
    def _save_training_data_csv(self, training_data: TrainingData, filename: str = "train_data.csv") -> None:
        """
        Save training data as CSV file.
        
        Args:
            training_data: TrainingData object to save
            filename: Name of the CSV file (default: "train_data.csv")
        """
        try:
            # Create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Create DataFrame with features and target, using IDs as index
            df = pd.DataFrame(training_data.features, columns=training_data.feature_names, index=training_data.ids)
            df['target'] = training_data.targets
            
            # Save to CSV with IDs as index
            csv_path = data_dir / filename
            df.to_csv(csv_path, index=True, index_label='id')
            
            logger.info(f"Training data saved to {csv_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
        except Exception as e:
            logger.warning(f"Failed to save training data as CSV: {e}")
            # Don't raise error - saving is optional, don't break the main functionality
    
    async def fetch_test_data(
        self,
        limit: int = 100,
        offset: int = 0,
        save: bool = False,
        csv_filename: str = "test_data.csv"
    ) -> TestData:
        """
        Fetch test data from API.
        
        Args:
            limit: Number of records to fetch
            offset: Pagination offset
            save: Whether to save data as CSV (default: False)
            csv_filename: Name of CSV file to save (default: "test_data.csv")
            
        Returns:
            TestData: Validated test data
            
        Raises:
            APIError: If request fails
            DataValidationError: If response is invalid
        """
        # Validate parameters
        assert limit > 0, f"Limit must be positive, got {limit}"
        assert offset >= 0, f"Offset must be non-negative, got {offset}"
        
        params = {
            'limit': limit,
            'offset': offset
        }
        
        logger.info(f"Fetching test data: limit={limit}, offset={offset}")
        
        raw_data = await self._make_request('/v1/bittensor/analytics/test-data', params)
        test_data = self._parse_test_response(raw_data)
        
        # Save as CSV if requested
        if save:
            self._save_test_data_csv(test_data, csv_filename)
        
        return test_data
    
    def _save_test_data_csv(self, test_data: TestData, filename: str = "test_data.csv") -> None:
        """
        Save test data as CSV file.
        
        Args:
            test_data: TestData object to save
            filename: Name of the CSV file (default: "test_data.csv")
        """
        try:
            # Create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Create DataFrame with features only (no targets for test data), using IDs as index
            df = pd.DataFrame(test_data.features, columns=test_data.feature_names, index=test_data.ids)
            
            # Save to CSV with IDs as index
            csv_path = data_dir / filename
            df.to_csv(csv_path, index=True, index_label='id')
            
            logger.info(f"Test data saved to {csv_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
        except Exception as e:
            logger.warning(f"Failed to save test data as CSV: {e}")
            # Don't raise error - saving is optional, don't break the main functionality
    
    async def fetch_both(
        self,
        train_limit: int = 100,
        train_offset: int = 0,
        round_number: int = 1,
        test_limit: int = 100,
        test_offset: int = 0,
        save_training: bool = True,
        save_test: bool = False,
        train_csv_filename: str = "train_data.csv",
        test_csv_filename: str = "test_data.csv"
    ) -> tuple[TrainingData, TestData]:
        """
        Fetch both training and test data concurrently.
        
        Args:
            train_limit: Training data limit
            train_offset: Training data offset
            round_number: Training round number
            test_limit: Test data limit
            test_offset: Test data offset
            save_training: Whether to save training data as CSV (default: True)
            save_test: Whether to save test data as CSV (default: False)
            train_csv_filename: Training CSV filename (default: "train_data.csv")
            test_csv_filename: Test CSV filename (default: "test_data.csv")
            
        Returns:
            tuple[TrainingData, TestData]: Both datasets
        """
        logger.info("Fetching both training and test data concurrently")
        
        # Execute both requests concurrently
        train_task = self.fetch_training_data(train_limit, train_offset, round_number, save_training, train_csv_filename)
        test_task = self.fetch_test_data(test_limit, test_offset, save_test, test_csv_filename)
        
        training_data, test_data = await asyncio.gather(train_task, test_task)
        
        # Validate that feature names match
        if training_data.feature_names != test_data.feature_names:
            raise DataValidationError(
                "Training and test data have different feature names"
            )
        
        return training_data, test_data
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("HTTP session closed") 
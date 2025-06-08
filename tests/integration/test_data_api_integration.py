#!/usr/bin/env python3
"""
Data API integration tests using real API calls.

This test module performs actual API calls to validate end-to-end functionality.
These tests require a valid .env file with API credentials.

Run with: python -m pytest tests/integration/test_data_api_integration.py -v
"""

import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from conversion_subnet.data_api.core import (
    VoiceFormAPIClient,
    VoiceFormAPIConfig,
    TrainingData,
    TestData,
    APIError
)


class TestDataAPIIntegration:
    """Integration tests for real API functionality."""
    
    @pytest.fixture
    def config_from_env(self):
        """Load configuration from .env file."""
        env_path = Path(__file__).parent.parent / ".env"
        if not env_path.exists():
            pytest.skip(f".env file not found: {env_path}")
        
        try:
            config = VoiceFormAPIConfig.from_env(str(env_path))
            return config
        except Exception as e:
            pytest.skip(f"Failed to load .env configuration: {e}")
    
    @pytest_asyncio.fixture
    async def client(self, config_from_env):
        """Create API client from .env configuration."""
        client = VoiceFormAPIClient(config_from_env)
        yield client
        await client.close()
    
    @pytest.mark.asyncio
    async def test_real_training_data_fetch(self, client):
        """Test fetching real training data from API."""
        try:
            training_data = await client.fetch_training_data(
                limit=5,           # Small sample for testing
                offset=0,          
                round_number=1,    
                save=False         
            )
            
            # Validate return type and basic structure
            assert isinstance(training_data, TrainingData)
            assert training_data.features.shape[0] <= 5
            assert training_data.features.shape[0] == training_data.targets.shape[0]
            assert len(training_data.feature_names) == training_data.features.shape[1]
            
            # Validate data quality
            assert not np.isnan(training_data.features).any()
            assert not np.isinf(training_data.features).any()
            
            print(f"✅ Real training data fetch successful: {training_data.features.shape}")
            
        except APIError as e:
            if e.status_code == 404:
                pytest.skip("API endpoint not available (404)")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_real_test_data_fetch(self, client):
        """Test fetching real test data from API."""
        try:
            test_data = await client.fetch_test_data(
                limit=5,           
                offset=0,          
                save=False         
            )
            
            # Validate return type and basic structure
            assert isinstance(test_data, TestData)
            assert test_data.features.shape[0] <= 5
            assert len(test_data.feature_names) == test_data.features.shape[1]
            
            # Validate data quality
            assert not np.isnan(test_data.features).any()
            assert not np.isinf(test_data.features).any()
            
            print(f"✅ Real test data fetch successful: {test_data.features.shape}")
            
        except APIError as e:
            if e.status_code == 404:
                pytest.skip("API endpoint not available (404)")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_data_compatibility(self, client):
        """Test that training and test data have compatible feature sets."""
        try:
            training_data = await client.fetch_training_data(limit=2, save=False)
            test_data = await client.fetch_test_data(limit=2, save=False)
            
            # Feature compatibility checks
            assert training_data.feature_names == test_data.feature_names
            assert training_data.features.shape[1] == test_data.features.shape[1]
            
            print("✅ Training and test data are compatible")
            
        except APIError as e:
            if e.status_code == 404:
                pytest.skip("API endpoint not available (404)")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_csv_saving(self, client):
        """Test CSV saving functionality with real data."""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                
                # Test training data CSV saving
                training_data = await client.fetch_training_data(
                    limit=3, 
                    save=True, 
                    csv_dir=str(tmp_path)
                )
                
                # Find generated CSV file
                csv_files = list(tmp_path.glob("*training*.csv"))
                assert len(csv_files) > 0, "Training CSV file should be generated"
                
                # Validate CSV content
                csv_file = csv_files[0]
                df = pd.read_csv(csv_file)
                assert len(df) == training_data.features.shape[0]
                assert len(df.columns) >= training_data.features.shape[1]  # Features + targets + ids
                
                print(f"✅ CSV saving successful: {csv_file.name}")
                
        except APIError as e:
            if e.status_code == 404:
                pytest.skip("API endpoint not available (404)")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_concurrent_data_fetch(self, client):
        """Test fetching both training and test data concurrently."""
        try:
            import asyncio
            
            # Fetch both datasets concurrently
            training_task = client.fetch_training_data(limit=2, save=False)
            test_task = client.fetch_test_data(limit=2, save=False)
            
            training_data, test_data = await asyncio.gather(training_task, test_task)
            
            # Both should succeed
            assert isinstance(training_data, TrainingData)
            assert isinstance(test_data, TestData)
            assert training_data.feature_names == test_data.feature_names
            
            print("✅ Concurrent fetch successful")
            
        except APIError as e:
            if e.status_code == 404:
                pytest.skip("API endpoint not available (404)")
            else:
                raise 
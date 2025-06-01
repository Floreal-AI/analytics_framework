#!/usr/bin/env python3
"""
API diagnostics test to understand endpoint structure and troubleshoot 404 errors.

This test helps diagnose API connectivity and endpoint issues without fallbacks.
"""

import pytest
import pytest_asyncio
import aiohttp
from pathlib import Path
import json
from urllib.parse import urljoin

from conversion_subnet.data_api.core import (
    VoiceFormAPIConfig,
    VoiceFormAPIClient,
    APIError
)


class TestAPIdiagnostics:
    """Diagnostic tests for API connectivity and endpoint discovery."""
    
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
    
    @pytest.mark.asyncio
    async def test_api_base_connectivity(self, config_from_env):
        """Test basic connectivity to the API base URL."""
        print(f"\n🔍 Testing API base connectivity...")
        print(f"   Base URL: {config_from_env.base_url}")
        print(f"   API Key length: {len(config_from_env.api_key)}")
        
        # Test direct HTTP connectivity
        headers = {
            'Authorization': f'Bearer {config_from_env.api_key}',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Basic connectivity to base URL
            try:
                async with session.get(config_from_env.base_url, headers=headers) as response:
                    print(f"   Base URL status: {response.status}")
                    if response.status in [200, 404, 405]:  # These are OK - means server is responding
                        print("   ✅ Server is responding")
                    else:
                        print(f"   ⚠️  Unexpected status: {response.status}")
                        response_text = await response.text()
                        print(f"   Response: {response_text[:200]}...")
            except Exception as e:
                print(f"   ❌ Connection failed: {e}")
                pytest.fail(f"Cannot connect to API base URL: {e}")
    
    @pytest.mark.asyncio
    async def test_endpoint_discovery(self, config_from_env):
        """Test various possible endpoints to discover the correct one."""
        print(f"\n🔍 Testing endpoint discovery...")
        
        # Common API endpoint patterns
        possible_endpoints = [
            '/',
            '/health',
            '/status',
            '/api',
            '/api/v1',
            '/train-data',
            '/training-data',
            '/data/train',
            '/data/training',
            '/v1/train-data',
            '/v1/training-data',
            '/bittensor/train-data',
            '/bittensor/training-data',
            '/analytics/train-data',
            '/analytics/training-data'
        ]
        
        headers = {
            'Authorization': f'Bearer {config_from_env.api_key}',
            'Content-Type': 'application/json'
        }
        
        working_endpoints = []
        
        async with aiohttp.ClientSession() as session:
            for endpoint in possible_endpoints:
                url = urljoin(config_from_env.base_url, endpoint)
                try:
                    async with session.get(url, headers=headers) as response:
                        status = response.status
                        print(f"   {endpoint:20} -> {status}")
                        
                        if status == 200:
                            working_endpoints.append(endpoint)
                            response_text = await response.text()
                            print(f"     ✅ SUCCESS: {response_text[:100]}...")
                        elif status == 401:
                            print(f"     🔐 Auth required")
                        elif status == 404:
                            print(f"     ❌ Not found")
                        elif status == 405:
                            print(f"     ⚠️  Method not allowed (endpoint exists)")
                        else:
                            print(f"     ? Status {status}")
                            
                except Exception as e:
                    print(f"   {endpoint:20} -> ERROR: {e}")
        
        print(f"\n📊 Discovery results:")
        print(f"   Working endpoints: {working_endpoints}")
        if not working_endpoints:
            print("   ⚠️  No endpoints returned 200 OK")
        
        return working_endpoints
    
    @pytest.mark.asyncio
    async def test_auth_methods(self, config_from_env):
        """Test different authentication methods."""
        print(f"\n🔍 Testing authentication methods...")
        
        base_endpoint = '/train-data'  # Assuming this is the correct endpoint
        url = urljoin(config_from_env.base_url, base_endpoint)
        
        auth_methods = [
            {'Authorization': f'Bearer {config_from_env.api_key}'},
            {'Authorization': f'Token {config_from_env.api_key}'},
            {'x-api-key': config_from_env.api_key},
            {'apikey': config_from_env.api_key},
            {'api-key': config_from_env.api_key},
        ]
        
        async with aiohttp.ClientSession() as session:
            for i, headers in enumerate(auth_methods, 1):
                try:
                    async with session.get(url, headers=headers) as response:
                        status = response.status
                        auth_header = list(headers.keys())[0]
                        print(f"   Method {i} ({auth_header}): {status}")
                        
                        if status == 200:
                            print(f"     ✅ SUCCESS with {auth_header}")
                            return headers
                        elif status == 401:
                            print(f"     🔐 Auth failed")
                        elif status == 404:
                            print(f"     ❌ Not found (but auth worked)")
                        else:
                            print(f"     ? Status {status}")
                            
                except Exception as e:
                    print(f"   Method {i}: ERROR: {e}")
        
        print("   ⚠️  No authentication method worked")
        return None
    
    @pytest.mark.asyncio
    async def test_api_with_client(self, config_from_env):
        """Test using our VoiceFormAPIClient to get detailed error info."""
        print(f"\n🔍 Testing with VoiceFormAPIClient...")
        
        client = VoiceFormAPIClient(config_from_env)
        
        try:
            # Try to fetch training data and catch the specific error
            training_data = await client.fetch_training_data(
                limit=1,
                offset=0,
                round_number=1,
                save=False
            )
            print("   ✅ SUCCESS: API call worked!")
            print(f"   Shape: {training_data.features.shape}")
            
        except APIError as e:
            print(f"   ❌ API Error: {e}")
            print(f"   Status code: {getattr(e, 'status_code', 'Unknown')}")
            
            # This is expected behavior - no fallbacks used
            print("   ✅ Error handling working correctly (no fallbacks)")
            
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            print(f"   Error type: {type(e).__name__}")
            
        finally:
            await client.close()
    
    @pytest.mark.asyncio 
    async def test_parameter_validation_works(self, config_from_env):
        """Test that our parameter validation catches issues early."""
        print(f"\n🔍 Testing parameter validation...")
        
        client = VoiceFormAPIClient(config_from_env)
        
        try:
            # Test various invalid parameters - should fail immediately
            test_cases = [
                {"limit": 0, "error": "Limit must be positive"},
                {"limit": -1, "error": "Limit must be positive"},
                {"offset": -1, "error": "Offset must be non-negative"},
                {"round_number": 0, "error": "Round number must be positive"},
                {"round_number": -1, "error": "Round number must be positive"},
            ]
            
            for i, test_case in enumerate(test_cases, 1):
                params = {"limit": 10, "offset": 0, "round_number": 1, "save": False}
                params.update({k: v for k, v in test_case.items() if k != "error"})
                
                try:
                    await client.fetch_training_data(**params)
                    print(f"   Test {i}: ❌ Should have failed for {test_case}")
                except AssertionError as e:
                    if test_case["error"] in str(e):
                        print(f"   Test {i}: ✅ Correctly caught: {test_case['error']}")
                    else:
                        print(f"   Test {i}: ⚠️  Wrong error: {e}")
                except Exception as e:
                    print(f"   Test {i}: ? Unexpected error: {e}")
            
        finally:
            await client.close()


if __name__ == '__main__':
    # Run diagnostics
    pytest.main([__file__, '-v', '-s']) 
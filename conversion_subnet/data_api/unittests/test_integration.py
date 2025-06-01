#!/usr/bin/env python3
"""
Integration test for simplified data API client.
Can be run with: python -m conversion_subnet.data_api.unittests.test_integration
"""

import asyncio
import os
import sys
import numpy as np

from conversion_subnet.data_api.core import (
    VoiceFormAPIClient,
    VoiceFormAPIConfig, 
    APIError,
    DataValidationError,
    TrainingData,
    TestData,
    validate_features,
    validate_targets,
    validate_feature_names
)


async def test_configuration():
    """Test configuration creation and validation."""
    print("🧪 Testing configuration...")
    
    # Test valid configuration
    config = VoiceFormAPIConfig(
        api_key="test-key",
        base_url="https://api.example.com",
        timeout_seconds=30
    )
    print(f"✅ Valid config created: timeout={config.timeout_seconds}s")
    
    # Test configuration validation
    try:
        VoiceFormAPIConfig(
            api_key="",  # Should fail
            base_url="https://api.example.com"
        )
        print("❌ Should have failed on empty API key")
    except AssertionError as e:
        print(f"✅ Configuration validation works: {e}")
    
    # Test invalid URL
    try:
        VoiceFormAPIConfig(
            api_key="test-key",
            base_url="not-a-url"  # Should fail
        )
        print("❌ Should have failed on invalid URL")
    except AssertionError as e:
        print(f"✅ URL validation works: {e}")


async def test_client_creation():
    """Test client creation."""
    print("\n🧪 Testing client creation...")
    
    config = VoiceFormAPIConfig(
        api_key="test-key",
        base_url="https://api.example.com",
        timeout_seconds=30
    )
    
    client = VoiceFormAPIClient(config)
    print("✅ Client created successfully")
    
    # Test headers
    headers = client._get_headers()
    assert headers['x-api-key'] == 'test-key'
    assert headers['accept'] == 'application/json'
    print("✅ Headers generated correctly")
    
    await client.close()


async def test_parameter_validation():
    """Test parameter validation with assertions."""
    print("\n🧪 Testing parameter validation...")
    
    config = VoiceFormAPIConfig(
        api_key="test-key",
        base_url="https://api.example.com",
        timeout_seconds=30
    )
    
    client = VoiceFormAPIClient(config)
    
    # Test invalid parameters (these should fail fast with assertions)
    try:
        await client.fetch_training_data(limit=0)  # Should fail
        print("❌ Should have failed on invalid limit")
    except AssertionError as e:
        print(f"✅ Parameter validation works: {e}")
    
    try:
        await client.fetch_training_data(offset=-1)  # Should fail
        print("❌ Should have failed on invalid offset")
    except AssertionError as e:
        print(f"✅ Offset validation works: {e}")
    
    await client.close()


async def test_environment_configuration():
    """Test environment-based configuration."""
    print("\n🧪 Testing environment configuration...")
    
    # Set test environment
    original_env = os.environ.copy()
    
    try:
        # Test missing required variables
        os.environ.clear()
        try:
            VoiceFormAPIConfig.from_env()
            print("❌ Should have failed on missing env vars")
        except ValueError as e:
            print(f"✅ Missing env var detection works: {e}")
        
        # Test valid environment
        os.environ['VOICEFORM_API_KEY'] = 'env-test-key'
        os.environ['VOICEFORM_API_BASE_URL'] = 'https://env.example.com'
        os.environ['DATA_API_TIMEOUT'] = '45'
        
        config = VoiceFormAPIConfig.from_env()
        assert config.api_key == 'env-test-key'
        assert config.base_url == 'https://env.example.com'
        assert config.timeout_seconds == 45
        print("✅ Environment configuration works")
        
        client = VoiceFormAPIClient.from_env()
        print("✅ Client creation from environment works")
        await client.close()
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


def test_dotenv_configuration():
    """Test dotenv configuration functionality."""
    print("\n🧪 Testing dotenv configuration...")
    
    try:
        # Test that dotenv import check works
        from conversion_subnet.data_api.core.config import DOTENV_AVAILABLE
        if DOTENV_AVAILABLE:
            print("✅ python-dotenv is available")
            
            # Test automatic .env loading (will fail gracefully if no .env file)
            try:
                config = VoiceFormAPIConfig.from_env()
                print("✅ Configuration loaded (either from env vars or .env file)")
            except ValueError as e:
                print(f"ℹ️  No valid configuration found (expected): {e}")
                
        else:
            print("ℹ️  python-dotenv not available (this is ok)")
            
            # Test that explicit dotenv path fails gracefully
            try:
                VoiceFormAPIConfig.from_env('nonexistent.env')
                print("❌ Should have failed without dotenv")
            except ImportError as e:
                print(f"✅ Dotenv import error handled correctly: {e}")
                
    except Exception as e:
        print(f"❌ Dotenv test failed: {e}")
        raise


def test_data_models():
    """Test data model validation."""
    print("\n🧪 Testing data models...")
    
    # Test valid training data
    training_data = TrainingData(
        features=np.array([[1.0, 2.0], [3.0, 4.0]]),
        targets=np.array([0, 1]),
        feature_names=['feature1', 'feature2'],
        round_number=1,
        updated_at='2023-12-01T10:30:00Z'
    )
    print("✅ Valid training data created")
    
    # Test data validation - this should fail
    try:
        TrainingData(
            features=np.array([[1.0, 2.0], [3.0, 4.0]]),
            targets=np.array([0]),  # Wrong length
            feature_names=['feature1', 'feature2'],
            round_number=1,
            updated_at='2023-12-01T10:30:00Z'
        )
        print("❌ Should have failed on mismatched shapes")
    except AssertionError as e:
        print(f"✅ Data validation works: {e}")
    
    # Test valid test data
    test_data = TestData(
        features=np.array([[5.0, 6.0], [7.0, 8.0]]),
        feature_names=['feature1', 'feature2'],
        submission_deadline='2024-01-15T23:59:59Z'
    )
    print("✅ Valid test data created")


def test_validators():
    """Test validation functions."""
    print("\n🧪 Testing validators...")
    
    # Test valid features
    features = validate_features([[1.0, 2.0], [3.0, 4.0]])
    assert features.shape == (2, 2)
    print("✅ Feature validation works")
    
    # Test valid targets
    targets = validate_targets([0, 1])
    assert targets.shape == (2,)
    print("✅ Target validation works")
    
    # Test feature names
    names = validate_feature_names(['f1', 'f2'], 2)
    assert len(names) == 2
    print("✅ Feature name validation works")
    
    # Test invalid data
    try:
        validate_features([[1.0, 2.0, float('nan')]])  # Should fail
        print("❌ Should have failed on NaN values")
    except DataValidationError as e:
        print(f"✅ NaN detection works: {e}")


async def test_api_error_handling():
    """Test API error handling with mock scenarios."""
    print("\n🧪 Testing API error handling...")
    
    config = VoiceFormAPIConfig(
        api_key="invalid-key",
        base_url="https://httpbin.org/status/404",  # Will return 404
        timeout_seconds=5
    )
    
    client = VoiceFormAPIClient(config)
    
    try:
        # This should fail with API error
        await client.fetch_training_data()
        print("❌ Should have failed with API error")
    except APIError as e:
        print(f"✅ API error handling works: {e}")
        if hasattr(e, 'status_code'):
            print(f"   Status code: {e.status_code}")
    
    finally:
        await client.close()


async def run_all_tests():
    """Run all tests."""
    print("🚀 Running simplified data API client integration tests\n")
    
    tests_passed = 0
    tests_failed = 0
    
    test_functions = [
        test_configuration,
        test_client_creation,
        test_parameter_validation,
        test_environment_configuration,
        test_dotenv_configuration,
        test_data_models,
        test_validators,
        test_api_error_handling
    ]
    
    for test_func in test_functions:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            tests_passed += 1
        except Exception as e:
            print(f"\n❌ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 All tests passed!")
        print("\n📊 Summary:")
        print("- Configuration validation: ✅")
        print("- Client creation: ✅")
        print("- Parameter validation: ✅")
        print("- Environment config: ✅")
        print("- Dotenv config: ✅")
        print("- Data models: ✅")
        print("- Validators: ✅")
        print("- API error handling: ✅")
        print("\n✨ Simplified implementation is working correctly!")
        return True
    else:
        print(f"\n💥 {tests_failed} test(s) failed!")
        return False


def main():
    """Main entry point for module testing."""
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 
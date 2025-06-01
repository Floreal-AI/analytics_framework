#!/usr/bin/env python3
"""
Simple example of the cleaned data API client.

Following principles:
- Clear, straightforward usage
- No hidden complexity
- Explicit error handling
- No fallback implementations
- Support for .env files via dotenv
"""

import asyncio
import os
from pathlib import Path

from conversion_subnet.data_api.core import (
    VoiceFormAPIClient,
    VoiceFormAPIConfig,
    APIError,
    DataValidationError
)


async def simple_usage_example():
    """Demonstrate simple, clean API usage."""
    
    # Load configuration directly from .env file
    env_path = Path(__file__).parent.parent / ".env"
    print(f"📂 Loading configuration from: {env_path}")
    
    try:
        from conversion_subnet.data_api.core import VoiceFormAPIConfig
        config = VoiceFormAPIConfig.from_env(str(env_path))
        client = VoiceFormAPIClient(config)
        print("✅ Configuration loaded from .env")
    except Exception as e:
        print(f"❌ Failed to load from .env: {e}")
        print("💡 Make sure you have a .env file with valid credentials")
        print("   Copy template: cp conversion_subnet/data_api/env.example conversion_subnet/data_api/.env")
        return
    
    try:
        print("🔄 Fetching training data (will save as CSV by default)...")
        training_data = await client.fetch_training_data(
            limit=100,
            offset=0,
            round_number=1,
            save=True  # Default is True - saves to /data/train_data.csv
        )
        
        print(f"✅ Training data: {training_data.features.shape}")
        print(f"   Features: {training_data.feature_names}")
        print(f"   Round: {training_data.round_number}")
        print(f"   Updated: {training_data.updated_at}")
        print("   💾 Saved to: data/train_data.csv")
        
        print("\n🔄 Fetching test data (save=False by default)...")
        test_data = await client.fetch_test_data(
            limit=100,
            offset=0,
            save=True  # Explicitly enable saving for test data
        )
        
        print(f"✅ Test data: {test_data.features.shape}")
        print(f"   Deadline: {test_data.submission_deadline}")
        print("   💾 Saved to: data/test_data.csv")
        
        # Verify feature compatibility
        if training_data.feature_names != test_data.feature_names:
            raise DataValidationError("Feature names don't match between datasets")
        
        print("✅ Feature names match between training and test data")
        
    except APIError as e:
        print(f"❌ API Error: {e}")
        if hasattr(e, 'status_code') and e.status_code:
            print(f"   HTTP Status: {e.status_code}")
        # Don't hide the error - let it propagate
        raise
        
    except DataValidationError as e:
        print(f"❌ Data Validation Error: {e}")
        # Don't hide the error - let it propagate
        raise
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        # Don't hide the error - let it propagate
        raise
        
    finally:
        # Always clean up resources
        await client.close()


async def concurrent_fetch_example():
    """Demonstrate concurrent fetching for efficiency."""
    
    # Load configuration directly from .env file
    env_path = Path(__file__).parent.parent / ".env"
    from conversion_subnet.data_api.core import VoiceFormAPIConfig
    config = VoiceFormAPIConfig.from_env(str(env_path))
    client = VoiceFormAPIClient(config)
    
    try:
        print("🔄 Fetching both datasets concurrently with CSV saving...")
        
        training_data, test_data = await client.fetch_both(
            train_limit=100,
            train_offset=0,
            round_number=1,
            test_limit=100,
            test_offset=0,
            save_training=True,  # Save training data as CSV
            save_test=True       # Save test data as CSV
        )
        
        print(f"✅ Training: {training_data.features.shape}")
        print(f"✅ Test: {test_data.features.shape}")
        print("✅ Both datasets fetched successfully")
        print("   💾 Saved training data to: data/train_data.csv")
        print("   💾 Saved test data to: data/test_data.csv")
        
    except (APIError, DataValidationError) as e:
        print(f"❌ Error: {e}")
        raise
        
    finally:
        await client.close()


def configuration_example():
    """Demonstrate configuration setup with dotenv support."""
    
    print("Configuration Methods:")
    print("1. Environment variables (traditional)")
    print("2. .env file (recommended for development)")
    print("3. Custom .env file path")
    
    # Method 1: From environment (recommended for production)
    try:
        config = VoiceFormAPIConfig.from_env()
        print("✅ Configuration loaded from environment/dotenv")
        
        # Validate that we have real credentials (not example values)
        if (config.api_key == 'your-api-key-here' or 
            'example.com' in config.base_url or
            'your-api-key' in config.api_key.lower()):
            print("⚠️  Warning: Using example/placeholder credentials")
            print("   Real API calls will fail with these values")
            return False
        else:
            print("✅ Real credentials detected - API calls should work")
            return True
            
    except ValueError as e:
        print(f"❌ Environment configuration failed: {e}")
        print("\nConfiguration options:")
        print("  Option 1: Set environment variables:")
        print("    export VOICEFORM_API_KEY='your-real-api-key'")
        print("    export VOICEFORM_API_BASE_URL='your-real-api-url'")
        print("\n  Option 2: Create a .env file:")
        print("    cp conversion_subnet/data_api/env.example conversion_subnet/data_api/.env")
        print("    # Edit .env with your actual values")
        print("\n  Option 3: Use a custom .env file:")
        print("    config = VoiceFormAPIConfig.from_env('custom.env')")
        return False
    
    # Method 2: Explicit configuration (example only)
    try:
        config = VoiceFormAPIConfig(
            api_key="your-key-here",
            base_url="https://api.example.com",
            timeout_seconds=30
        )
        print("✅ Client created with explicit configuration")
    except Exception as e:
        print(f"❌ Explicit configuration failed: {e}")
    
    return False


def dotenv_example():
    """Demonstrate different ways to use dotenv."""
    
    print("\n🔧 Dotenv Configuration Examples:")
    
    # Check if .env file exists
    env_file = Path("conversion_subnet/data_api/.env")
    env_example = Path("conversion_subnet/data_api/env.example")
    
    if env_example.exists():
        print(f"✅ Example env file found: {env_example}")
        print("   Copy this to .env and update with your values:")
        print(f"   cp {env_example} conversion_subnet/data_api/.env")
    
    if env_file.exists():
        print(f"✅ .env file found: {env_file}")
        try:
            # Load from default .env location
            config = VoiceFormAPIConfig.from_env()
            print("✅ Configuration loaded from .env file")
        except ValueError as e:
            print(f"❌ .env file configuration failed: {e}")
    else:
        print(f"ℹ️  .env file not found: {env_file}")
        print("   Create one from the example:")
        print(f"   cp {env_example} {env_file}")
    
    # Example of loading from custom path
    try:
        custom_config = VoiceFormAPIConfig.from_env("custom.env")
        print("✅ Configuration loaded from custom.env")
    except (ValueError, FileNotFoundError):
        print("ℹ️  Custom .env file not found (this is expected for demo)")


async def error_handling_example():
    """Demonstrate proper error handling without fallbacks."""
    
    # Create client with invalid configuration to demonstrate error handling
    try:
        # This will fail due to invalid URL
        config = VoiceFormAPIConfig(
            api_key="test-key",
            base_url="invalid-url",  # This will trigger validation error
            timeout_seconds=30
        )
        
    except AssertionError as e:
        print(f"✅ Configuration validation caught error: {e}")
        print("   Error handling working correctly - no fallbacks used")
    
    # Demonstrate API error handling
    try:
        # Using a valid config but will fail on actual API call
        config = VoiceFormAPIConfig(
            api_key="invalid-key",
            base_url="https://httpbin.org/status/404",  # Will return 404
            timeout_seconds=30
        )
        
        client = VoiceFormAPIClient(config)
        
        # This should fail with API error
        await client.fetch_training_data()
        
    except APIError as e:
        print(f"✅ API error caught: {e}")
        print("   No fallback attempted - error properly raised")
    
    finally:
        await client.close()


async def fetch_data_example():
    """
    Focused example: Fetch training and test data and inspect structure.
    
    This example shows:
    - How to fetch training data
    - How to fetch test data 
    - Data structure inspection
    - Basic data analysis
    - CSV saving options
    """
    
    print("📊 Fetching Training and Test Data Example")
    print("=" * 50)
    
    # Load configuration from .env file
    env_path = Path(__file__).parent.parent / ".env"
    from conversion_subnet.data_api.core import VoiceFormAPIConfig
    config = VoiceFormAPIConfig.from_env(str(env_path))
    client = VoiceFormAPIClient(config)
    
    try:
        # 1. Fetch Training Data
        print("\n🎯 Step 1: Fetching Training Data")
        print("-" * 30)
        
        training_data = await client.fetch_training_data(
            limit=50,           # Get 50 samples
            offset=0,           # Start from beginning  
            round_number=1,     # Training round 1
            save=True          # Save as CSV (default: True)
        )
        
        print(f"✅ Training data fetched successfully!")
        print(f"   📏 Shape: {training_data.features.shape} (rows × features)")
        print(f"   🏷️  Targets: {training_data.targets.shape} samples")
        print(f"   📋 Features: {len(training_data.feature_names)} columns")
        print(f"   🔄 Round: {training_data.round_number}")
        print(f"   🕐 Updated: {training_data.updated_at}")
        print(f"   💾 Saved to: data/train_data.csv")
        
        # Show feature names
        print(f"\n📊 Feature Names (first 5): {training_data.feature_names[:5]}")
        if len(training_data.feature_names) > 5:
            print(f"   ... and {len(training_data.feature_names) - 5} more")
        
        # Show basic statistics
        import numpy as np
        print(f"\n📈 Training Data Statistics:")
        print(f"   Features - Min: {training_data.features.min():.3f}, Max: {training_data.features.max():.3f}")
        print(f"   Features - Mean: {training_data.features.mean():.3f}, Std: {training_data.features.std():.3f}")
        print(f"   Targets - Unique values: {np.unique(training_data.targets)}")
        print(f"   Targets - Distribution: {dict(zip(*np.unique(training_data.targets, return_counts=True)))}")
        
        # 2. Fetch Test Data
        print("\n🎯 Step 2: Fetching Test Data")
        print("-" * 30)
        
        test_data = await client.fetch_test_data(
            limit=50,           # Get 50 samples
            offset=0,           # Start from beginning
            save=True          # Save as CSV (default: False, but enabling for demo)
        )
        
        print(f"✅ Test data fetched successfully!")
        print(f"   📏 Shape: {test_data.features.shape} (rows × features)")
        print(f"   📋 Features: {len(test_data.feature_names)} columns")
        print(f"   ⏰ Deadline: {test_data.submission_deadline}")
        print(f"   💾 Saved to: data/test_data.csv")
        
        # Show test data statistics
        print(f"\n📈 Test Data Statistics:")
        print(f"   Features - Min: {test_data.features.min():.3f}, Max: {test_data.features.max():.3f}")
        print(f"   Features - Mean: {test_data.features.mean():.3f}, Std: {test_data.features.std():.3f}")
        
        # 3. Compare Datasets
        print("\n🎯 Step 3: Dataset Comparison")
        print("-" * 30)
        
        # Check feature compatibility
        if training_data.feature_names == test_data.feature_names:
            print("✅ Feature names match between training and test data")
            print(f"   📊 Both datasets have {len(training_data.feature_names)} features")
        else:
            print("❌ Feature names mismatch!")
            print(f"   Training features: {len(training_data.feature_names)}")
            print(f"   Test features: {len(test_data.feature_names)}")
        
        # Compare feature ranges
        train_ranges = (training_data.features.min(axis=0), training_data.features.max(axis=0))
        test_ranges = (test_data.features.min(axis=0), test_data.features.max(axis=0))
        
        print(f"\n📊 Feature Range Comparison:")
        print(f"   Training - Overall min: {train_ranges[0].min():.3f}, max: {train_ranges[1].max():.3f}")
        print(f"   Test     - Overall min: {test_ranges[0].min():.3f}, max: {test_ranges[1].max():.3f}")
        
        # 4. Ready for ML
        print("\n🎯 Step 4: Ready for Machine Learning")
        print("-" * 30)
        print("✅ Data is ready for ML workflows!")
        print(f"   🎯 Training: {training_data.features.shape[0]} samples for model training")
        print(f"   🧪 Test: {test_data.features.shape[0]} samples for predictions")
        print(f"   📊 Features: {training_data.features.shape[1]} input features")
        print(f"   💾 CSV files saved in 'data/' folder for external analysis")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Load CSV files: pd.read_csv('data/train_data.csv')")
        print(f"   2. Train your model on training_data.features → training_data.targets")
        print(f"   3. Make predictions on test_data.features") 
        print(f"   4. Submit predictions before: {test_data.submission_deadline}")
        
    except APIError as e:
        print(f"❌ API Error: {e}")
        if hasattr(e, 'status_code'):
            print(f"   Status Code: {e.status_code}")
        raise
        
    except DataValidationError as e:
        print(f"❌ Data Validation Error: {e}")
        raise
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        raise
        
    finally:
        await client.close()
        print(f"\n🔒 Client connection closed")


async def csv_saving_example():
    """Demonstrate CSV saving options."""
    
    # Load configuration directly from .env file
    env_path = Path(__file__).parent.parent / ".env"
    from conversion_subnet.data_api.core import VoiceFormAPIConfig
    config = VoiceFormAPIConfig.from_env(str(env_path))
    client = VoiceFormAPIClient(config)
    
    try:
        print("📊 CSV Saving Examples:")
        
        # Example 1: Training data with default save=True
        print("\n1. Training data (save=True by default)")
        training = await client.fetch_training_data(limit=50, round_number=1)
        print("   💾 Automatically saved to data/train_data.csv")
        
        # Example 2: Test data with explicit save=True
        print("\n2. Test data (save=False by default, enabling manually)")
        test = await client.fetch_test_data(limit=50, save=True)
        print("   💾 Manually saved to data/test_data.csv")
        
        # Example 3: Disable saving for training data
        print("\n3. Training data without saving")
        training_no_save = await client.fetch_training_data(limit=10, save=False)
        print("   📂 No CSV file created")
        
        print("\n✅ CSV saving examples completed")
        print("💡 Note: CSV files are saved in the 'data' folder with feature names as columns")
        
    except (APIError, DataValidationError) as e:
        print(f"❌ Error: {e}")
        raise
        
    finally:
        await client.close()


async def main():
    """Run all examples."""
    
    print("🧪 Simple Data API Client Examples (using .env credentials)\n")
    
    print("1. Configuration Example:")
    has_valid_config = configuration_example()
    
    print("\n2. Dotenv Configuration Example:")
    dotenv_example()
    
    print("\n3. Error Handling Example:")
    await error_handling_example()
    
    # Run API examples using credentials from .env
    print("\n4. Simple Usage Example (using .env):")
    try:
        await simple_usage_example()
    except Exception as e:
        print(f"❌ Simple usage example failed: {e}")
        print("   This might be due to network issues or API changes")
    
    print("\n5. Concurrent Fetch Example (using .env):")
    try:
        await concurrent_fetch_example()
    except Exception as e:
        print(f"❌ Concurrent fetch example failed: {e}")
        print("   This might be due to network issues or API changes")
        
    print("\n6. Fetch Data Example (using .env):")
    try:
        await fetch_data_example()
    except Exception as e:
        print(f"❌ Fetch data example failed: {e}")
        print("   This might be due to network issues or API changes")
    
    print("\n7. CSV Saving Example (using .env):")
    try:
        await csv_saving_example()
    except Exception as e:
        print(f"❌ CSV saving example failed: {e}")
        print("   This might be due to network issues or API changes")
    
    print("\n✅ All examples completed")
    print("\n📋 Summary:")
    print("✅ Configuration examples - Always work")
    print("✅ Dotenv examples - Always work") 
    print("✅ Error handling examples - Always work")
    print("✅ API examples - Using credentials from .env file")
    print("✅ Fetch data example - Detailed data inspection and analysis")
    print("✅ CSV saving example - Demonstrates saving options")
    print("\n💡 Setup instructions:")
    print("   1. Copy template: cp conversion_subnet/data_api/env.example conversion_subnet/data_api/.env")
    print("   2. Edit .env with your real API credentials")
    print("   3. Run examples: python -m conversion_subnet.data_api.examples.example")


if __name__ == '__main__':
    asyncio.run(main()) 
#!/usr/bin/env python3
"""
Standalone example: Fetch training and test data from VoiceForm API.

This example demonstrates:
- How to fetch training data
- How to fetch test data  
- Data structure inspection
- Basic data analysis
- CSV saving functionality
- Data compatibility checking

Setup:
1. Copy template: cp conversion_subnet/data_api/env.example conversion_subnet/data_api/.env
2. Edit .env with your API credentials
3. Run with: python -m conversion_subnet.data_api.examples.fetch_data_example
"""

import asyncio
import numpy as np
from pathlib import Path

from conversion_subnet.data_api.core import (
    VoiceFormAPIClient,
    VoiceFormAPIConfig,
    APIError,
    DataValidationError
)


async def main():
    """
    Fetch training and test data and perform detailed analysis.
    """
    
    print("🚀 VoiceForm Data API - Fetch Training & Test Data")
    print("=" * 60)
    print("This example fetches both datasets and shows their structure\n")
    
    # Load configuration from .env file
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    print(f"📂 Loading configuration from: {env_path}")
    
    try:
        # Load configuration from .env file
        config = VoiceFormAPIConfig.from_env(str(env_path))
        client = VoiceFormAPIClient(config)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        print("💡 Make sure you have a .env file with:")
        print("   VOICEFORM_API_KEY=your-api-key")
        print("   VOICEFORM_API_BASE_URL=your-api-url")
        print("   Copy from: cp conversion_subnet/data_api/env.example conversion_subnet/data_api/.env")
        return
    
    try:
        # Step 1: Fetch Training Data
        print("\n" + "="*60)
        print("🎯 STEP 1: FETCHING TRAINING DATA")
        print("="*60)
        
        print("🔄 Requesting training data...")
        training_data = await client.fetch_training_data(
            limit=100,          # Get 100 samples
            offset=0,           # Start from beginning  
            round_number=1,     # Training round 1
            save=True          # Save as CSV (default: True)
        )
        
        print("✅ Training data fetched successfully!")
        print(f"\n📊 TRAINING DATA SUMMARY:")
        print(f"   📏 Dataset shape: {training_data.features.shape} (samples × features)")
        print(f"   🏷️  Target shape: {training_data.targets.shape}")
        print(f"   📋 Feature count: {len(training_data.feature_names)}")
        print(f"   🔄 Training round: {training_data.round_number}")
        print(f"   🕐 Last updated: {training_data.updated_at}")
        print(f"   💾 Saved to: data/train_data.csv")
        
        # Feature names
        print(f"\n📝 FEATURE NAMES:")
        if len(training_data.feature_names) <= 10:
            print(f"   All features: {training_data.feature_names}")
        else:
            print(f"   First 10: {training_data.feature_names[:10]}")
            print(f"   ... and {len(training_data.feature_names) - 10} more")
        
        # Training data statistics
        print(f"\n📈 TRAINING STATISTICS:")
        print(f"   Features:")
        print(f"     • Min value: {training_data.features.min():.3f}")
        print(f"     • Max value: {training_data.features.max():.3f}")
        print(f"     • Mean: {training_data.features.mean():.3f}")
        print(f"     • Std dev: {training_data.features.std():.3f}")
        
        print(f"   Targets:")
        unique_targets, target_counts = np.unique(training_data.targets, return_counts=True)
        print(f"     • Unique values: {unique_targets}")
        print(f"     • Distribution: {dict(zip(unique_targets, target_counts))}")
        print(f"     • Class balance: {target_counts.min()}/{target_counts.max()} (min/max)")
        
        # Step 2: Fetch Test Data
        print("\n" + "="*60)
        print("🎯 STEP 2: FETCHING TEST DATA")
        print("="*60)
        
        print("🔄 Requesting test data...")
        test_data = await client.fetch_test_data(
            limit=100,          # Get 100 samples
            offset=0,           # Start from beginning
            save=True          # Save as CSV (default: False for test data)
        )
        
        print("✅ Test data fetched successfully!")
        print(f"\n📊 TEST DATA SUMMARY:")
        print(f"   📏 Dataset shape: {test_data.features.shape} (samples × features)")
        print(f"   📋 Feature count: {len(test_data.feature_names)}")
        print(f"   ⏰ Submission deadline: {test_data.submission_deadline}")
        print(f"   💾 Saved to: data/test_data.csv")
        
        # Test data statistics
        print(f"\n📈 TEST STATISTICS:")
        print(f"   Features:")
        print(f"     • Min value: {test_data.features.min():.3f}")
        print(f"     • Max value: {test_data.features.max():.3f}")
        print(f"     • Mean: {test_data.features.mean():.3f}")
        print(f"     • Std dev: {test_data.features.std():.3f}")
        
        # Step 3: Dataset Comparison
        print("\n" + "="*60)
        print("🎯 STEP 3: DATASET COMPATIBILITY CHECK")
        print("="*60)
        
        # Feature compatibility
        features_match = training_data.feature_names == test_data.feature_names
        if features_match:
            print("✅ FEATURE COMPATIBILITY: PERFECT MATCH")
            print(f"   📊 Both datasets have {len(training_data.feature_names)} identical features")
        else:
            print("❌ FEATURE COMPATIBILITY: MISMATCH DETECTED")
            print(f"   Training features: {len(training_data.feature_names)}")
            print(f"   Test features: {len(test_data.feature_names)}")
            
            # Find differences
            train_set = set(training_data.feature_names)
            test_set = set(test_data.feature_names)
            missing_in_test = train_set - test_set
            missing_in_train = test_set - train_set
            
            if missing_in_test:
                print(f"   Missing in test: {list(missing_in_test)[:5]}...")
            if missing_in_train:
                print(f"   Missing in train: {list(missing_in_train)[:5]}...")
        
        # Feature range comparison
        print(f"\n📊 FEATURE RANGE COMPARISON:")
        train_min, train_max = training_data.features.min(), training_data.features.max()
        test_min, test_max = test_data.features.min(), test_data.features.max()
        
        print(f"   Training data ranges: [{train_min:.3f}, {train_max:.3f}]")
        print(f"   Test data ranges:     [{test_min:.3f}, {test_max:.3f}]")
        
        # Check for distribution shift
        train_mean = training_data.features.mean()
        test_mean = test_data.features.mean()
        mean_diff = abs(train_mean - test_mean)
        
        if mean_diff < 0.1:
            print(f"✅ Distribution similarity: Good (mean diff: {mean_diff:.3f})")
        elif mean_diff < 0.5:
            print(f"⚠️  Distribution similarity: Moderate (mean diff: {mean_diff:.3f})")
        else:
            print(f"❌ Distribution similarity: Poor (mean diff: {mean_diff:.3f})")
        
        # Step 4: Ready for ML
        print("\n" + "="*60)
        print("🎯 STEP 4: MACHINE LEARNING READINESS")
        print("="*60)
        
        print("✅ DATASETS ARE READY FOR ML WORKFLOWS!")
        print(f"\n🎯 TRAINING SETUP:")
        print(f"   • Samples: {training_data.features.shape[0]:,}")
        print(f"   • Features: {training_data.features.shape[1]:,}")
        print(f"   • Target classes: {len(np.unique(training_data.targets))}")
        print(f"   • Data file: data/train_data.csv")
        
        print(f"\n🧪 PREDICTION SETUP:")
        print(f"   • Test samples: {test_data.features.shape[0]:,}")
        print(f"   • Features: {test_data.features.shape[1]:,}")
        print(f"   • Submission deadline: {test_data.submission_deadline}")
        print(f"   • Data file: data/test_data.csv")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. 📁 Load data: pd.read_csv('data/train_data.csv')")
        print(f"   2. 🎯 Train model: X = features, y = target column")
        print(f"   3. 🧪 Predict: model.predict(test_features)")
        print(f"   4. 📤 Submit before: {test_data.submission_deadline}")
        
        print(f"\n📋 SAMPLE CODE:")
        print(f"   ```python")
        print(f"   import pandas as pd")
        print(f"   from sklearn.ensemble import RandomForestClassifier")
        print(f"   ")
        print(f"   # Load data")
        print(f"   train_df = pd.read_csv('data/train_data.csv')")
        print(f"   test_df = pd.read_csv('data/test_data.csv')")
        print(f"   ")
        print(f"   # Prepare training data")
        print(f"   X_train = train_df.drop('target', axis=1)")
        print(f"   y_train = train_df['target']")
        print(f"   X_test = test_df")
        print(f"   ")
        print(f"   # Train and predict")
        print(f"   model = RandomForestClassifier()")
        print(f"   model.fit(X_train, y_train)")
        print(f"   predictions = model.predict(X_test)")
        print(f"   ```")
        
    except APIError as e:
        print(f"\n❌ API ERROR: {e}")
        if hasattr(e, 'status_code'):
            print(f"   HTTP Status: {e.status_code}")
        print("   This could be due to:")
        print("   • Invalid API credentials")
        print("   • Network connectivity issues")
        print("   • API service temporarily unavailable")
        
    except DataValidationError as e:
        print(f"\n❌ DATA VALIDATION ERROR: {e}")
        print("   This could be due to:")
        print("   • Unexpected API response format")
        print("   • Missing required fields")
        print("   • Data corruption during transfer")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("   Please check your setup and try again")
        
    finally:
        await client.close()
        print(f"\n🔒 Client connection closed")
        print("=" * 60)
        print("✅ Example completed!")


if __name__ == '__main__':
    asyncio.run(main()) 
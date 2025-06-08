#!/usr/bin/env python3
"""
Standalone example: Fetch training and test data from VoiceForm API.

This example demonstrates:
- How to fetch training data with IDs
- How to fetch test data with IDs
- Data structure inspection including ID handling
- Basic data analysis with indexed data
- Parameterized CSV saving functionality
- Data compatibility checking with ID validation
- Advanced CSV naming patterns

Setup:
1. Copy template: cp conversion_subnet/data_api/env.example conversion_subnet/data_api/.env
2. Edit .env with your API credentials
3. Run with: python -m conversion_subnet.data_api.examples.fetch_data_example
"""

import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime

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
    
    print("🚀 VoiceForm Data API - Fetch Training & Test Data with IDs")
    print("=" * 70)
    print("This example fetches both datasets and shows their structure including ID handling\n")
    
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
        # Step 1: Fetch Training Data with Custom CSV Name
        print("\n" + "="*70)
        print("🎯 STEP 1: FETCHING TRAINING DATA WITH CUSTOM CSV NAME")
        print("="*70)
        
        # Use timestamp-based filename for training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_train_filename = f"training_round1_{timestamp}.csv"
        
        print("🔄 Requesting training data...")
        print(f"📁 Custom CSV filename: {custom_train_filename}")
        
        training_data = await client.fetch_training_data(
            limit=100,                          # Get 100 samples
            offset=0,                           # Start from beginning  
            round_number=1,                     # Training round 1
            save=True,                          # Save as CSV (default: True)
            csv_filename=custom_train_filename  # Custom filename [Optional]
        )
        
        print("✅ Training data fetched successfully!")
        print(f"\n📊 TRAINING DATA SUMMARY:")
        print(f"   📏 Dataset shape: {training_data.features.shape} (samples × features)")
        print(f"   🏷️  Target shape: {training_data.targets.shape}")
        print(f"   🆔 IDs shape: {training_data.ids.shape}")
        print(f"   📋 Feature count: {len(training_data.feature_names)}")
        print(f"   🔄 Training round: {training_data.round_number}")
        print(f"   🕐 Last updated: {training_data.updated_at}")
        print(f"   💾 Saved to: data/{custom_train_filename}")
        
        # ID Information
        print(f"\n🆔 ID INFORMATION:")
        print(f"   • ID count: {len(training_data.ids)}")
        print(f"   • ID type: {training_data.ids.dtype}")
        print(f"   • Unique IDs: {len(np.unique(training_data.ids))}")
        print(f"   • Sample IDs: {list(training_data.ids[:5])}...")
        
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
        
        # Step 2: Fetch Test Data with Custom CSV Name
        print("\n" + "="*70)
        print("🎯 STEP 2: FETCHING TEST DATA WITH CUSTOM CSV NAME")
        print("="*70)
        
        # Use descriptive filename for test data
        custom_test_filename = "evaluation_set_submission.csv"
        
        print("🔄 Requesting test data...")
        print(f"📁 Custom CSV filename: {custom_test_filename}")
        
        test_data = await client.fetch_test_data(
            limit=100,                         # Get 100 samples
            offset=0,                          # Start from beginning
            save=True,                         # Save as CSV
            csv_filename=custom_test_filename  # Custom filename [Optional]
        )
        
        print("✅ Test data fetched successfully!")
        print(f"\n📊 TEST DATA SUMMARY:")
        print(f"   📏 Dataset shape: {test_data.features.shape} (samples × features)")
        print(f"   🆔 IDs shape: {test_data.ids.shape}")
        print(f"   📋 Feature count: {len(test_data.feature_names)}")
        print(f"   ⏰ Submission deadline: {test_data.submission_deadline}")
        print(f"   💾 Saved to: data/{custom_test_filename}")
        
        # Test ID Information
        print(f"\n🆔 TEST ID INFORMATION:")
        print(f"   • ID count: {len(test_data.ids)}")
        print(f"   • ID type: {test_data.ids.dtype}")
        print(f"   • Unique IDs: {len(np.unique(test_data.ids))}")
        print(f"   • Sample IDs: {list(test_data.ids[:5])}...")
        
        # Test data statistics
        print(f"\n📈 TEST STATISTICS:")
        print(f"   Features:")
        print(f"     • Min value: {test_data.features.min():.3f}")
        print(f"     • Max value: {test_data.features.max():.3f}")
        print(f"     • Mean: {test_data.features.mean():.3f}")
        print(f"     • Std dev: {test_data.features.std():.3f}")
        
        # Step 3: Dataset Compatibility Check (including IDs)
        print("\n" + "="*70)
        print("🎯 STEP 3: DATASET COMPATIBILITY CHECK (INCLUDING IDS)")
        print("="*70)
        
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
        
        # ID compatibility check
        print(f"\n🆔 ID COMPATIBILITY CHECK:")
        train_ids_set = set(training_data.ids)
        test_ids_set = set(test_data.ids)
        id_overlap = train_ids_set.intersection(test_ids_set)
        
        if len(id_overlap) == 0:
            print("✅ ID SEPARATION: PERFECT (no overlap between train/test IDs)")
            print(f"   Training IDs: {len(train_ids_set)} unique")
            print(f"   Test IDs: {len(test_ids_set)} unique")
            print(f"   Overlap: 0 (ideal for ML)")
        else:
            print("⚠️  ID OVERLAP DETECTED: Some IDs appear in both datasets")
            print(f"   Training IDs: {len(train_ids_set)} unique")
            print(f"   Test IDs: {len(test_ids_set)} unique")
            print(f"   Overlapping IDs: {len(id_overlap)}")
            print(f"   Sample overlaps: {list(id_overlap)[:5]}...")
        
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
        
        # Step 4: Demonstrate fetch_both with custom naming
        print("\n" + "="*70)
        print("🎯 STEP 4: CONCURRENT FETCH WITH CUSTOM NAMING")
        print("="*70)
        
        print("🔄 Fetching both datasets concurrently with custom names...")
        
        # Custom filenames for batch operation
        batch_train_filename = f"batch_train_{timestamp}.csv"
        batch_test_filename = f"batch_test_{timestamp}.csv"
        
        batch_training_data, batch_test_data = await client.fetch_both(
            train_limit=50,                              # Smaller sample for demo
            train_offset=0,
            round_number=1,
            test_limit=50,
            test_offset=0,
            save_training=True,
            save_test=True,
            train_csv_filename=batch_train_filename,     # Custom training filename
            test_csv_filename=batch_test_filename        # Custom test filename
        )
        
        print("✅ Concurrent fetch completed!")
        print(f"   📁 Training saved to: data/{batch_train_filename}")
        print(f"   📁 Test saved to: data/{batch_test_filename}")
        print(f"   🎯 Training samples: {batch_training_data.features.shape[0]}")
        print(f"   🧪 Test samples: {batch_test_data.features.shape[0]}")
        
        # Step 5: Machine Learning Readiness with ID-indexed CSVs
        print("\n" + "="*70)
        print("🎯 STEP 5: MACHINE LEARNING READINESS WITH ID-INDEXED CSVS")
        print("="*70)
        
        print("✅ DATASETS ARE READY FOR ML WORKFLOWS WITH ID TRACKING!")
        print(f"\n🎯 TRAINING SETUP:")
        print(f"   • Samples: {training_data.features.shape[0]:,}")
        print(f"   • Features: {training_data.features.shape[1]:,}")
        print(f"   • Target classes: {len(np.unique(training_data.targets))}")
        print(f"   • Unique IDs: {len(np.unique(training_data.ids))}")
        print(f"   • Data file: data/{custom_train_filename}")
        
        print(f"\n🧪 PREDICTION SETUP:")
        print(f"   • Test samples: {test_data.features.shape[0]:,}")
        print(f"   • Features: {test_data.features.shape[1]:,}")
        print(f"   • Unique IDs: {len(np.unique(test_data.ids))}")
        print(f"   • Submission deadline: {test_data.submission_deadline}")
        print(f"   • Data file: data/{custom_test_filename}")
        
        print(f"\n💡 NEXT STEPS WITH ID TRACKING:")
        print(f"   1. 📁 Load data: pd.read_csv('data/{custom_train_filename}', index_col='id')")
        print(f"   2. 🎯 Train model: X = features, y = target column")
        print(f"   3. 🧪 Predict: model.predict(test_features)")
        print(f"   4. 📤 Submit with IDs: predictions indexed by test IDs")
        print(f"   5. ⏰ Submit before: {test_data.submission_deadline}")
        
        print(f"\n📋 SAMPLE CODE WITH ID HANDLING:")
        print(f"   ```python")
        print(f"   import pandas as pd")
        print(f"   from sklearn.ensemble import RandomForestClassifier")
        print(f"   ")
        print(f"   # Load data with ID index")
        print(f"   train_df = pd.read_csv('data/{custom_train_filename}', index_col='id')")
        print(f"   test_df = pd.read_csv('data/{custom_test_filename}', index_col='id')")
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
        print(f"   ")
        print(f"   # Create submission with ID tracking")
        print(f"   submission = pd.DataFrame({{")
        print(f"       'id': test_df.index,")
        print(f"       'prediction': predictions")
        print(f"   }})")
        print(f"   submission.to_csv('submission.csv', index=False)")
        print(f"   ```")
        
        # Step 6: Advanced CSV Naming Examples
        print("\n" + "="*70)
        print("🎯 STEP 6: ADVANCED CSV NAMING PATTERNS")
        print("="*70)
        
        print("💡 CSV NAMING EXAMPLES:")
        print(f"\n🕐 Timestamp-based:")
        print(f"   csv_filename='train_{{datetime.now().strftime(\"%Y%m%d_%H%M\")}}.csv'")
        print(f"   csv_filename='test_{{datetime.now().strftime(\"%Y%m%d_%H%M\")}}.csv'")
        
        print(f"\n🎯 Experiment-based:")
        print(f"   csv_filename='train_experiment_001.csv'")
        print(f"   csv_filename='test_baseline_model.csv'")
        
        print(f"\n🌍 Environment-based:")
        print(f"   csv_filename='train_data_dev.csv'")
        print(f"   csv_filename='train_data_prod.csv'")
        
        print(f"\n🔄 Round-based:")
        print(f"   csv_filename=f'train_round_{{round_number}}.csv'")
        print(f"   csv_filename=f'test_round_{{round_number}}.csv'")
        
        print(f"\n🏆 Competition-based:")
        print(f"   csv_filename='final_submission.csv'")
        print(f"   csv_filename='validation_set.csv'")
        
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
        print("   • Missing required fields (including 'ids')")
        print("   • Data corruption during transfer")
        print("   • ID validation failures")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("   Please check your setup and try again")
        
    finally:
        await client.close()
        print(f"\n🔒 Client connection closed")
        print("=" * 70)
        print("✅ Enhanced example with IDs and parameterized CSV completed!")


if __name__ == '__main__':
    asyncio.run(main()) 
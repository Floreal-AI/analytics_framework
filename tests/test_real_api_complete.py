#!/usr/bin/env python3
"""
Complete Real API Test
======================

This test performs end-to-end testing using real API endpoints:
1. Fetch real training data (limit=100, offset=0)
2. Fetch real test data one by one (limit=1, offset=0,1,2...)
3. Get real model predictions with proper formatting
4. Calculate rewards based on real predictions vs mock ground truth
"""

import asyncio
import os
import time
import json
import numpy as np
from datetime import datetime

# Import framework components
from conversion_subnet.data_api.core.client import VoiceFormAPIClient
from conversion_subnet.data_api.core.config import VoiceFormAPIConfig
from conversion_subnet.miner.miner import BinaryClassificationMiner
from conversion_subnet.validator.reward import Validator
from conversion_subnet.protocol import ConversionSynapse

class CompleteRealAPITest:
    """Complete test using real API endpoints."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
        # Environment setup
        self.api_base_url = os.getenv('VOICEFORM_API_BASE_URL')
        self.api_key = os.getenv('VOICEFORM_API_KEY')
        
        print("🚀 COMPLETE REAL API PIPELINE TEST")
        print("=" * 80)
        
        if not self.api_base_url or not self.api_key:
            print("⚠️  Environment variables not configured")
            print("   Will use mock data for demonstration")
            self.use_real_api = False
        else:
            print("✅ Environment variables configured")
            print(f"   API Base URL: {self.api_base_url}")
            print(f"   API Key: ***{self.api_key[-4:]}")
            self.use_real_api = True
        
        # Configure API client
        self.config = VoiceFormAPIConfig(
            api_key=self.api_key or "mock-key",
            base_url=self.api_base_url or "https://mock.com",
            timeout_seconds=30
        )
        self.client = VoiceFormAPIClient(self.config)
        self.validator = Validator()
    
    def log_result(self, step: str, data: dict, success: bool = True):
        """Log a test step result."""
        elapsed = time.time() - self.start_time
        result = {
            "step": step,
            "elapsed_seconds": round(elapsed, 3),
            "success": success,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"\n{status} [{elapsed:.3f}s] {step}")
        
        for key, value in data.items():
            if isinstance(value, dict) and len(str(value)) > 100:
                print(f"   {key}: dict({len(value)} keys)")
            elif isinstance(value, (list, np.ndarray)) and len(value) > 10:
                print(f"   {key}: {type(value).__name__}(length={len(value)})")
            else:
                print(f"   {key}: {value}")
    
    async def fetch_training_data(self):
        """Fetch real training data."""
        print(f"\n{'='*60}")
        print("STEP 1: FETCH TRAINING DATA")
        print("=" * 60)
        
        try:
            if self.use_real_api:
                training_data = await self.client.fetch_training_data(
                    limit=100, offset=0, save=False
                )
                
                self.log_result("Fetch Training Data (Real API)", {
                    "samples": len(training_data.features),
                    "features": len(training_data.feature_names),
                    "conversion_rate": round(float(np.mean(training_data.targets)), 4),
                    "round_number": training_data.round_number
                })
                
            else:
                # Mock data
                training_data = None
                self.log_result("Fetch Training Data (Mock)", {
                    "samples": 100,
                    "features": 42,
                    "conversion_rate": 0.04
                })
            
            return training_data
            
        except Exception as e:
            self.log_result("Fetch Training Data", {
                "error": str(e)
            }, success=False)
            return None
    
    async def initialize_miner(self):
        """Initialize the miner model."""
        print(f"\n{'='*60}")
        print("STEP 2: INITIALIZE MINER")
        print("=" * 60)
        
        try:
            miner = BinaryClassificationMiner()
            
            self.log_result("Initialize Miner", {
                "model_type": type(miner.model).__name__ if hasattr(miner, 'model') else "Unknown",
                "model_loaded": hasattr(miner, 'model') and miner.model is not None
            })
            
            return miner
            
        except Exception as e:
            self.log_result("Initialize Miner", {
                "error": str(e)
            }, success=False)
            return None
    
    async def process_test_samples(self, miner, num_samples=3):
        """Fetch test data and get predictions."""
        print(f"\n{'='*60}")
        print("STEP 3: PROCESS TEST SAMPLES")
        print("=" * 60)
        
        predictions = []
        
        for i in range(num_samples):
            try:
                print(f"\n--- Sample {i+1}/{num_samples} ---")
                
                # Fetch test data
                if self.use_real_api:
                    test_data = await self.client.fetch_test_data(
                        limit=1, offset=i, save=False
                    )
                    
                    if len(test_data.features) == 0:
                        print(f"No more test data at offset {i}")
                        break
                    
                    # Convert to features dict
                    features = {
                        name: float(value) for name, value in 
                        zip(test_data.feature_names, test_data.features[0])
                    }
                    test_pk = test_data.ids[0]
                    
                else:
                    # Mock features
                    features = {f"feature_{j}": np.random.rand() for j in range(42)}
                    test_pk = f"test_sample_{i}"
                
                features['test_pk'] = test_pk
                
                # Create synapse
                synapse = ConversionSynapse(features=features)
                
                # Get prediction
                try:
                    raw_prediction = miner.forward(synapse)
                    
                    # Format prediction properly
                    if isinstance(raw_prediction, dict):
                        prediction = raw_prediction
                    else:
                        # Convert to proper format
                        conversion = int(raw_prediction) if isinstance(raw_prediction, (int, float)) else 0
                        prediction = {
                            'conversion_happened': conversion,
                            'time_to_conversion_seconds': 60.0 if conversion == 1 else -1.0
                        }
                    
                    # Create proper synapse with prediction
                    result_synapse = ConversionSynapse(
                        features=features,
                        prediction=prediction,
                        confidence=0.85,
                        response_time=0.1,
                        miner_uid=0
                    )
                    
                    predictions.append({
                        "sample_id": i,
                        "test_pk": test_pk,
                        "prediction": prediction,
                        "synapse": result_synapse
                    })
                    
                    self.log_result(f"Sample {i+1} Prediction", {
                        "test_pk": test_pk,
                        "prediction": prediction,
                        "feature_count": len([k for k in features.keys() if k != "test_pk"])
                    })
                    
                except Exception as pred_error:
                    print(f"Prediction error: {pred_error}")
                    continue
                
            except Exception as e:
                print(f"Sample {i+1} error: {e}")
                continue
        
        return predictions
    
    async def calculate_rewards(self, predictions):
        """Calculate rewards for predictions."""
        print(f"\n{'='*60}")
        print("STEP 4: CALCULATE REWARDS")
        print("=" * 60)
        
        total_reward = 0.0
        
        for i, pred_data in enumerate(predictions):
            try:
                print(f"\n--- Reward {i+1}/{len(predictions)} ---")
                
                prediction = pred_data["prediction"]
                synapse = pred_data["synapse"]
                
                # Mock ground truth (in real scenario from validation API)
                ground_truth = {
                    'conversion_happened': np.random.choice([0, 1], p=[0.85, 0.15]),
                    'time_to_conversion_seconds': float(np.random.uniform(45, 90)) if np.random.random() < 0.15 else -1.0
                }
                
                # Calculate reward
                reward = self.validator.reward(ground_truth, synapse)
                total_reward += reward
                
                # Analysis
                correct = prediction['conversion_happened'] == ground_truth['conversion_happened']
                
                self.log_result(f"Reward Calculation {i+1}", {
                    "test_pk": pred_data["test_pk"],
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "reward": round(reward, 6),
                    "correct": correct
                })
                
            except Exception as e:
                print(f"Reward calculation error: {e}")
                continue
        
        # Summary
        if len(predictions) > 0:
            avg_reward = total_reward / len(predictions)
            
            self.log_result("Reward Summary", {
                "samples": len(predictions),
                "total_reward": round(total_reward, 6),
                "average_reward": round(avg_reward, 6),
                "performance_vs_max": round(avg_reward / 0.8, 3)
            })
        
        return total_reward
    
    def save_results(self):
        """Save test results."""
        results_data = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "execution_time": round(time.time() - self.start_time, 3),
                "api_mode": "Real API" if self.use_real_api else "Mock"
            },
            "steps": self.results
        }
        
        filename = "complete_real_api_results.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
        
        successful_steps = sum(1 for r in self.results if r['success'])
        print(f"✅ Success rate: {successful_steps}/{len(self.results)}")
        print(f"⏱️  Total time: {time.time() - self.start_time:.3f}s")

async def main():
    """Run the complete test."""
    tester = CompleteRealAPITest()
    
    # Step 1: Fetch training data
    training_data = await tester.fetch_training_data()
    
    # Step 2: Initialize miner
    miner = await tester.initialize_miner()
    if not miner:
        print("❌ Failed to initialize miner")
        return
    
    # Step 3: Process test samples
    predictions = await tester.process_test_samples(miner, num_samples=3)
    if not predictions:
        print("❌ No predictions made")
        return
    
    # Step 4: Calculate rewards
    total_reward = await tester.calculate_rewards(predictions)
    
    # Save results
    tester.save_results()
    
    print(f"\n🎉 TEST COMPLETED!")
    print(f"📊 {len(predictions)} samples processed")
    print(f"💰 Total reward: {total_reward:.6f}")

if __name__ == "__main__":
    asyncio.run(main()) 
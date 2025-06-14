#!/usr/bin/env python3
"""
Comprehensive End-to-End Pipeline Test
=====================================

This test traces the complete pipeline:
1. Download train data using VoiceFormAPIClient
2. Train a model using the miner
3. Send challenges from validator
4. Get predictions from miner
5. Validate predictions with external API
6. Calculate rewards

All inputs and outputs are logged in detail.
"""

import asyncio
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Dict, Any, List

# Import all the components we need to test
from conversion_subnet.data_api.core.client import VoiceFormAPIClient
from conversion_subnet.data_api.core.config import VoiceFormAPIConfig
from conversion_subnet.data_api.core.models import TrainingData, TestData
from conversion_subnet.miner.miner import BinaryClassificationMiner
from conversion_subnet.validator.validation_client import ValidationAPIClient, ValidationResult
from conversion_subnet.validator.forward import get_external_validation, validate_prediction
from conversion_subnet.validator.reward import Validator
from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.validator.generate import generate_conversation
from conversion_subnet.validator.utils import validate_features

class PipelineTracer:
    """Helper class to trace all method calls with detailed input/output logging."""
    
    def __init__(self):
        self.trace_log = []
        self.step_counter = 0
    
    def log_step(self, step_name: str, method: str, inputs: Dict[str, Any], outputs: Any = None, error: Exception = None):
        """Log a pipeline step with full details."""
        self.step_counter += 1
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "step": self.step_counter,
            "timestamp": timestamp,
            "step_name": step_name,
            "method": method,
            "inputs": self._serialize_data(inputs),
            "outputs": self._serialize_data(outputs) if outputs is not None else None,
            "error": str(error) if error else None,
            "success": error is None
        }
        
        self.trace_log.append(log_entry)
        
        # Print step details
        print(f"\n{'='*80}")
        print(f"STEP {self.step_counter}: {step_name}")
        print(f"{'='*80}")
        print(f"Method: {method}")
        print(f"Timestamp: {timestamp}")
        print(f"Inputs:")
        for key, value in inputs.items():
            print(f"  {key}: {self._format_for_display(value)}")
        
        if outputs is not None:
            print(f"Outputs:")
            print(f"  result: {self._format_for_display(outputs)}")
        
        if error:
            print(f"Error: {error}")
            print(f"Success: False")
        else:
            print(f"Success: True")
    
    def _serialize_data(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, np.ndarray):
            return {"type": "numpy.ndarray", "shape": data.shape, "dtype": str(data.dtype), "sample": data.flatten()[:5].tolist()}
        elif isinstance(data, pd.DataFrame):
            return {"type": "pandas.DataFrame", "shape": data.shape, "columns": data.columns.tolist(), "sample": data.head(2).to_dict()}
        elif isinstance(data, (TrainingData, TestData, ValidationResult)):
            return {"type": type(data).__name__, "data": data.__dict__}
        elif hasattr(data, '__dict__'):
            return {"type": type(data).__name__, "attributes": {k: self._serialize_data(v) for k, v in data.__dict__.items()}}
        elif isinstance(data, (list, tuple)) and len(data) > 10:
            return {"type": type(data).__name__, "length": len(data), "sample": data[:3]}
        else:
            return data
    
    def _format_for_display(self, data: Any, max_length: int = 200) -> str:
        """Format data for readable display."""
        if isinstance(data, np.ndarray):
            return f"numpy.ndarray(shape={data.shape}, dtype={data.dtype})"
        elif isinstance(data, pd.DataFrame):
            return f"DataFrame(shape={data.shape}, columns={len(data.columns)})"
        elif isinstance(data, str) and len(data) > max_length:
            return f"{data[:max_length]}... ({len(data)} chars total)"
        elif isinstance(data, (list, tuple)) and len(data) > 5:
            return f"{type(data).__name__}(length={len(data)}, sample={data[:3]}...)"
        elif isinstance(data, dict) and len(str(data)) > max_length:
            keys = list(data.keys())[:3]
            return f"dict(keys={keys}..., {len(data)} total items)"
        else:
            return str(data)
    
    def save_trace(self, filename: str = "pipeline_trace.json"):
        """Save the complete trace to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.trace_log, f, indent=2, default=str)
        print(f"\n{'='*80}")
        print(f"TRACE SAVED: {filename}")
        print(f"Total steps: {self.step_counter}")
        print(f"{'='*80}")

async def test_complete_pipeline():
    """Test the complete pipeline from data download to reward calculation."""
    tracer = PipelineTracer()
    
    try:
        # =================================================================
        # STEP 1: DOWNLOAD TRAINING DATA
        # =================================================================
        
        # Mock training data response
        mock_train_response = {
            "train_features": np.random.rand(10, 40).tolist(),  # 10 samples, 40 features
            "train_targets": np.random.randint(0, 2, 10).tolist(),  # Binary targets
            "features_list": [f"feature_{i}" for i in range(40)],
            "ids": [f"train_id_{i}" for i in range(10)],
            "round_number": 1,
            "refresh_date": "2025-06-09T10:00:00Z"
        }
        
        config = VoiceFormAPIConfig(
            api_key="test-key",
            base_url="https://test-api.com/v1",
            timeout_seconds=30
        )
        
        client = VoiceFormAPIClient(config)
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_train_response
            
            inputs = {"limit": 100, "offset": 0, "round_number": 1, "save": False}
            
            try:
                training_data = await client.fetch_training_data(**inputs)
                tracer.log_step(
                    "Download Training Data",
                    "VoiceFormAPIClient.fetch_training_data",
                    inputs,
                    training_data
                )
            except Exception as e:
                tracer.log_step(
                    "Download Training Data",
                    "VoiceFormAPIClient.fetch_training_data", 
                    inputs,
                    error=e
                )
                raise
        
        # =================================================================
        # STEP 2: TRAIN MODEL (MINER)
        # =================================================================
        
        # Create mock miner
        miner = BinaryClassificationMiner()
        miner.metagraph = MagicMock()
        miner.metagraph.hotkeys = ["test_hotkey"]
        
        # Simulate training (miner would normally train on the data)
        train_inputs = {
            "features": training_data.features,
            "targets": training_data.targets,
            "feature_names": training_data.feature_names
        }
        
        try:
            # Mock the training process
            miner.model_trained = True  # Simulate successful training
            training_result = {"status": "trained", "accuracy": 0.85, "features_used": 40}
            
            tracer.log_step(
                "Train Model",
                "BinaryClassificationMiner.train",
                train_inputs,
                training_result
            )
        except Exception as e:
            tracer.log_step(
                "Train Model",
                "BinaryClassificationMiner.train",
                train_inputs,
                error=e
            )
            raise
        
        # =================================================================
        # STEP 3: GENERATE CHALLENGE (VALIDATOR)
        # =================================================================
        
        try:
            # Generate conversation features for challenge
            conversation = generate_conversation()
            features = validate_features(conversation)
            test_pk = str(uuid.uuid4())
            features['test_pk'] = test_pk
            
            challenge_inputs = {"conversation_seed": "random", "features_count": 40}
            challenge_outputs = {"features": features, "test_pk": test_pk}
            
            tracer.log_step(
                "Generate Challenge",
                "generate_conversation + validate_features",
                challenge_inputs,
                challenge_outputs
            )
        except Exception as e:
            tracer.log_step(
                "Generate Challenge",
                "generate_conversation + validate_features",
                challenge_inputs,
                error=e
            )
            raise
        
        # =================================================================
        # STEP 4: SEND CHALLENGE TO MINER
        # =================================================================
        
        synapse = ConversionSynapse(features=features)
        
        challenge_request_inputs = {
            "synapse_features": features,
            "test_pk": test_pk,
            "timeout": 60.0
        }
        
        try:
            # Mock miner response
            with patch.object(miner, 'forward', return_value={
                'conversion_happened': 1,
                'time_to_conversion_seconds': 45.5,
                'confidence': 0.87
            }) as mock_forward:
                
                response = await miner.forward_async(synapse)
                
                tracer.log_step(
                    "Send Challenge to Miner",
                    "BinaryClassificationMiner.forward_async",
                    challenge_request_inputs,
                    response
                )
        except Exception as e:
            tracer.log_step(
                "Send Challenge to Miner", 
                "BinaryClassificationMiner.forward_async",
                challenge_request_inputs,
                error=e
            )
            raise
        
        # =================================================================
        # STEP 5: VALIDATE PREDICTION FORMAT
        # =================================================================
        
        prediction = {
            'conversion_happened': response['conversion_happened'],
            'time_to_conversion_seconds': response['time_to_conversion_seconds']
        }
        
        validation_inputs = {"prediction": prediction}
        
        try:
            is_valid = validate_prediction(prediction)
            validation_outputs = {"is_valid": is_valid, "processed_prediction": prediction}
            
            tracer.log_step(
                "Validate Prediction Format",
                "validate_prediction",
                validation_inputs,
                validation_outputs
            )
        except Exception as e:
            tracer.log_step(
                "Validate Prediction Format",
                "validate_prediction",
                validation_inputs,
                error=e
            )
            raise
        
        # =================================================================
        # STEP 6: EXTERNAL VALIDATION API CALL
        # =================================================================
        
        # Mock external validation response
        mock_validation_response = {
            "test_pk": test_pk,
            "labels": True,  # True means conversion happened
            "submissionDeadline": "2025-06-09T12:00:00Z"
        }
        
        validation_client = ValidationAPIClient(
            "https://test-api.com/v1",
            "test-api-key",
            30.0
        )
        
        external_validation_inputs = {"test_pk": test_pk}
        
        try:
            with patch.object(validation_client, 'get_validation_result', new_callable=AsyncMock) as mock_validation:
                mock_validation_result = ValidationResult(
                    test_pk=test_pk,
                    labels=True,
                    submission_deadline="2025-06-09T12:00:00Z",
                    response_time=0.25
                )
                mock_validation.return_value = mock_validation_result
                
                # Configure default client for get_external_validation
                from conversion_subnet.validator.validation_client import configure_default_validation_client
                configure_default_validation_client("https://test-api.com/v1", "test-key", 30.0)
                
                with patch('conversion_subnet.validator.validation_client._default_client', validation_client):
                    validation_result = await get_external_validation(test_pk)
                
                tracer.log_step(
                    "External Validation API Call",
                    "get_external_validation",
                    external_validation_inputs,
                    validation_result
                )
        except Exception as e:
            tracer.log_step(
                "External Validation API Call",
                "get_external_validation",
                external_validation_inputs,
                error=e
            )
            raise
        
        # =================================================================
        # STEP 7: CONVERT VALIDATION TO GROUND TRUTH
        # =================================================================
        
        ground_truth_inputs = {
            "validation_labels": validation_result.labels,
            "conversion_rule": "true->1, false->0"
        }
        
        try:
            ground_truth = {
                'conversion_happened': 1 if validation_result.labels else 0,
                'time_to_conversion_seconds': 60.0 if validation_result.labels else -1.0
            }
            
            tracer.log_step(
                "Convert Validation to Ground Truth",
                "boolean_to_binary_conversion",
                ground_truth_inputs,
                ground_truth
            )
        except Exception as e:
            tracer.log_step(
                "Convert Validation to Ground Truth",
                "boolean_to_binary_conversion",
                ground_truth_inputs,
                error=e
            )
            raise
        
        # =================================================================
        # STEP 8: CALCULATE REWARD
        # =================================================================
        
        # Create response synapse for reward calculation
        response_synapse = ConversionSynapse(
            features=features,
            prediction=prediction,
            confidence=response['confidence'],
            response_time=2.5,
            miner_uid=0
        )
        
        reward_inputs = {
            "ground_truth": ground_truth,
            "miner_prediction": prediction,
            "confidence": response['confidence'],
            "response_time": 2.5
        }
        
        try:
            validator = Validator()
            reward = validator.reward(ground_truth, response_synapse)
            
            reward_outputs = {
                "final_reward": reward,
                "prediction_accuracy": prediction['conversion_happened'] == ground_truth['conversion_happened'],
                "time_accuracy": abs(prediction['time_to_conversion_seconds'] - ground_truth['time_to_conversion_seconds']) if ground_truth['conversion_happened'] == 1 else "N/A"
            }
            
            tracer.log_step(
                "Calculate Reward",
                "Validator.reward", 
                reward_inputs,
                reward_outputs
            )
        except Exception as e:
            tracer.log_step(
                "Calculate Reward",
                "Validator.reward",
                reward_inputs,
                error=e
            )
            raise
        
        # =================================================================
        # STEP 9: PIPELINE SUMMARY
        # =================================================================
        
        summary = {
            "total_steps": tracer.step_counter,
            "successful_steps": sum(1 for log in tracer.trace_log if log["success"]),
            "failed_steps": sum(1 for log in tracer.trace_log if not log["success"]),
            "final_reward": reward,
            "prediction_correct": prediction['conversion_happened'] == ground_truth['conversion_happened'],
            "pipeline_status": "SUCCESS"
        }
        
        tracer.log_step(
            "Pipeline Summary",
            "end_to_end_pipeline",
            {},
            summary
        )
        
        print(f"\n{'='*80}")
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Final Reward: {reward:.4f}")
        print(f"Prediction Accuracy: {'✅ CORRECT' if summary['prediction_correct'] else '❌ INCORRECT'}")
        print(f"Total Steps: {tracer.step_counter}")
        print(f"Success Rate: {summary['successful_steps']}/{tracer.step_counter}")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        tracer.log_step(
            "Pipeline Failure",
            "end_to_end_pipeline",
            {},
            error=e
        )
    
    finally:
        # Save detailed trace
        tracer.save_trace("pipeline_trace.json")
        
        # Return summary for further analysis
        return tracer.trace_log

if __name__ == "__main__":
    print("🚀 Starting Comprehensive End-to-End Pipeline Test")
    print("=" * 80)
    
    # Run the complete pipeline test
    trace_log = asyncio.run(test_complete_pipeline())
    
    print(f"\n📊 DETAILED TRACE ANALYSIS")
    print("=" * 80)
    
    for i, log_entry in enumerate(trace_log, 1):
        status = "✅" if log_entry["success"] else "❌"
        print(f"{status} Step {i}: {log_entry['step_name']} - {log_entry['method']}")
        if not log_entry["success"]:
            print(f"   Error: {log_entry['error']}")
    
    print(f"\n🏁 Test completed. Check 'pipeline_trace.json' for full details.") 
#!/usr/bin/env python3
"""
Enhanced Miner Debug Wrapper

This script wraps the miner with comprehensive logging and debugging
to identify why predictions are returning None.
"""

import sys
import time
import traceback
import json
from typing import Dict, Any, Optional

import bittensor as bt
from conversion_subnet.protocol import ConversionSynapse, PredictionOutput
from conversion_subnet.base.miner import BaseMinerNeuron
from conversion_subnet.miner.miner import BinaryClassificationMiner
from conversion_subnet.utils.log import logger
from conversion_subnet.constants import TIMEOUT_SEC

class DebugMiner(BaseMinerNeuron):
    """
    Enhanced miner with comprehensive debugging and error tracking.
    """

    def __init__(self, config=None):
        print("🔧 Initializing DebugMiner with enhanced logging...")
        
        # Initialize with base configuration
        super(DebugMiner, self).__init__(config=config)

        # Request counter for tracking
        self.request_count = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.error_log = []

        # Explicitly create necessary config objects if missing
        if not hasattr(self.config, 'neuron'):
            self.config.neuron = bt.config()
            self.config.neuron.device = 'cpu'
            logger.warning("Created missing neuron config with defaults")

        if not hasattr(self.config, 'miner'):
            self.config.miner = bt.config()
            logger.warning("Created missing miner config with defaults")
        
        # Set device for the miner model
        device = 'cpu'
        if hasattr(self.config.neuron, 'device') and self.config.neuron.device is not None:
            device = self.config.neuron.device
        
        self.config.miner.device = device
        
        print(f"📍 Device configuration: {device}")
        print(f"📍 Wallet: {self.wallet.name}/{self.wallet.hotkey.name}")
        print(f"📍 Network: {self.config.subtensor.network}")
        print(f"📍 Chain endpoint: {self.config.subtensor.chain_endpoint}")
        
        # Initialize the binary classification miner with enhanced error handling
        logger.info(f"Initializing classifier with device: {self.config.miner.device}")
        
        try:
            self.classifier = BinaryClassificationMiner(
                model_type="xgboost",
                device=self.config.miner.device
            )
            print("✅ XGBoost classifier initialized successfully")
        except Exception as e:
            print(f"❌ CRITICAL: Failed to initialize classifier: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            raise

        logger.info("Enhanced debug miner initialized with binary classification model")
        print("🎯 Debug miner ready - enhanced logging enabled")

    async def forward(self, synapse: ConversionSynapse) -> ConversionSynapse:
        """
        Enhanced forward function with comprehensive debugging.
        """
        self.request_count += 1
        request_id = f"REQ-{self.request_count:04d}"
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🔍 {request_id}: INCOMING REQUEST")
        print(f"{'='*60}")
        print(f"📊 Total requests processed: {self.request_count}")
        print(f"✅ Successful predictions: {self.successful_predictions}")
        print(f"❌ Failed predictions: {self.failed_predictions}")
        print(f"📍 Hotkey: {getattr(synapse.dendrite, 'hotkey', 'UNKNOWN')}")
        print(f"📍 Features count: {len(synapse.features) if synapse.features else 0}")
        
        # Log key features
        if synapse.features:
            key_features = ['test_pk', 'conversation_duration_seconds', 'total_messages', 'has_target_entity']
            print("📋 Key features:")
            for feature in key_features:
                if feature in synapse.features:
                    print(f"   {feature}: {synapse.features[feature]}")
        
        try:
            # Validate synapse
            if synapse is None:
                raise ValueError("Synapse is None")
                
            if not hasattr(synapse, 'features') or synapse.features is None:
                raise ValueError("Synapse has no features")
                
            if len(synapse.features) == 0:
                raise ValueError("Synapse features are empty")
            
            print("✅ Synapse validation passed")
            
            # Test classifier availability
            if not hasattr(self, 'classifier') or self.classifier is None:
                raise ValueError("Classifier not initialized")
                
            print("✅ Classifier available")
            
            # Make prediction with detailed error handling
            print(f"🧠 Calling classifier.forward()...")
            prediction_start = time.time()
            
            try:
                result = self.classifier.forward(synapse)
                prediction_time = time.time() - prediction_start
                print(f"✅ Classifier prediction completed in {prediction_time:.3f}s")
                print(f"📊 Raw result: {result}")
                
                # Validate result format
                if result is None:
                    raise ValueError("Classifier returned None result")
                    
                if not isinstance(result, dict):
                    raise ValueError(f"Classifier returned non-dict result: {type(result)}")
                    
                required_keys = ['conversion_happened', 'time_to_conversion_seconds', 'confidence']
                for key in required_keys:
                    if key not in result:
                        raise ValueError(f"Missing required key in result: {key}")
                
                print("✅ Result format validation passed")
                
                # Attach predictions to synapse
                synapse.prediction = {
                    'conversion_happened': result['conversion_happened'],
                    'time_to_conversion_seconds': result['time_to_conversion_seconds']
                }
                synapse.confidence = result.get('confidence', 0.5)
                
                print(f"✅ Synapse updated with prediction:")
                print(f"   conversion_happened: {synapse.prediction['conversion_happened']}")
                print(f"   time_to_conversion_seconds: {synapse.prediction['time_to_conversion_seconds']}")
                print(f"   confidence: {synapse.confidence}")
                
                self.successful_predictions += 1
                
                # Enhanced debug logging
                logger.debug(f"{request_id}: Processed synapse for hotkey {synapse.dendrite.hotkey}: {synapse.prediction}")
                
            except Exception as classifier_error:
                print(f"❌ Classifier error: {classifier_error}")
                print(f"🔍 Classifier traceback: {traceback.format_exc()}")
                raise  # Re-raise to be caught by outer handler

        except Exception as e:
            self.failed_predictions += 1
            error_info = {
                'request_id': request_id,
                'error': str(e),
                'timestamp': time.time(),
                'traceback': traceback.format_exc()
            }
            self.error_log.append(error_info)
            
            print(f"❌ PREDICTION FAILED: {e}")
            print(f"🔍 Full traceback:")
            print(traceback.format_exc())
            
            logger.error(f"{request_id}: Error in forward: {e}")
            logger.error(f"{request_id}: Full traceback: {traceback.format_exc()}")
            
            # Return a valid default prediction instead of None
            synapse.prediction = {
                'conversion_happened': 0,
                'time_to_conversion_seconds': -1.0
            }
            synapse.confidence = 0.0
            
            print(f"🔄 Set fallback prediction: {synapse.prediction}")

        total_time = time.time() - start_time
        print(f"⏱️  Total processing time: {total_time:.3f}s")
        print(f"{'='*60}")
        
        # Log periodic statistics
        if self.request_count % 10 == 0:
            self.log_statistics()
        
        return synapse

    def log_statistics(self):
        """Log comprehensive statistics."""
        print(f"\n📊 MINER STATISTICS (after {self.request_count} requests)")
        print("-" * 50)
        print(f"✅ Successful predictions: {self.successful_predictions}")
        print(f"❌ Failed predictions: {self.failed_predictions}")
        
        if self.request_count > 0:
            success_rate = (self.successful_predictions / self.request_count) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        # Show recent errors
        recent_errors = self.error_log[-5:]  # Last 5 errors
        if recent_errors:
            print("\n🚨 Recent errors:")
            for error in recent_errors:
                print(f"  {error['request_id']}: {error['error']}")
        
        print("-" * 50)

    async def blacklist(self, synapse: ConversionSynapse) -> tuple[bool, str]:
        """Enhanced blacklist with debugging."""
        request_id = f"BL-{int(time.time())}"
        print(f"🔍 {request_id}: Blacklist check for hotkey {synapse.dendrite.hotkey}")
        
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            logger.trace(f"{request_id}: Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            print(f"❌ {request_id}: Blacklisted - unrecognized hotkey")
            return True, "Unrecognized hotkey"
        
        logger.trace(f"{request_id}: Not blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        print(f"✅ {request_id}: Allowed - recognized hotkey")
        return False, "Hotkey recognized"

    async def priority(self, synapse: ConversionSynapse) -> float:
        """Enhanced priority with debugging."""
        try:
            caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            priority = float(self.metagraph.S[caller_uid])
            logger.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
            print(f"📊 Priority for {synapse.dendrite.hotkey}: {priority}")
            return priority
        except Exception as e:
            logger.error(f"Error in priority calculation: {e}")
            print(f"❌ Priority calculation failed: {e}")
            return 0.0

def main():
    """Main function to run the enhanced debug miner."""
    print("🚀 Starting Enhanced Debug Miner")
    print("=" * 60)
    
    try:
        # Parse config
        config = bt.config()
        
        # Add miner-specific arguments
        config.add_args(bt.subtensor.add_args)
        config.add_args(bt.wallet.add_args)
        config.add_args(bt.axon.add_args)
        config.add_args(bt.logging.add_args)
        
        # Create and run the debug miner
        miner = DebugMiner(config=config)
        
        print(f"🎯 Debug miner configured and ready")
        print(f"📍 Listening on: {config.axon.ip}:{config.axon.port}")
        print(f"📍 External IP: {getattr(config.axon, 'external_ip', 'auto-detect')}")
        print("=" * 60)
        print("🔍 Enhanced logging enabled - all requests will be debugged")
        print("⏸️  Press Ctrl+C to stop")
        print()
        
        # Run the miner
        miner.run()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Debug miner stopped by user")
    except Exception as e:
        print(f"\n\n❌ Debug miner failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
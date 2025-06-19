#!/usr/bin/env python3
"""
Simple test to verify local miner-validator communication.
This bypasses the metagraph registration issues and tests direct communication.
"""

import asyncio
import time
from typing import Dict, Any

import bittensor as bt
from conversion_subnet.protocol import ConversionSynapse

async def test_direct_miner_communication():
    """Test direct communication with local miner."""
    
    print("🔍 Testing direct miner communication...")
    
    # Create test features (complete set)
    test_features = {
        'test_pk': 'direct_test_' + str(int(time.time())),
        'session_id': 'direct_session',
        'conversation_duration_seconds': 120.0,
        'conversation_duration_minutes': 2.0,
        'hour_of_day': 14,
        'day_of_week': 2,
        'is_business_hours': 1,
        'is_weekend': 0,
        'time_to_first_response_seconds': 10.0,
        'avg_response_time_seconds': 15.0,
        'max_response_time_seconds': 30.0,
        'min_response_time_seconds': 5.0,
        'avg_agent_response_time_seconds': 12.0,
        'avg_user_response_time_seconds': 18.0,
        'response_time_stddev': 3.0,
        'response_gap_max': 45.0,
        'messages_per_minute': 3.0,
        'total_messages': 15,
        'user_messages_count': 8,
        'agent_messages_count': 7,
        'message_ratio': 1.14,
        'avg_message_length_user': 45.0,
        'max_message_length_user': 80,
        'min_message_length_user': 15,
        'total_chars_from_user': 360,
        'avg_message_length_agent': 55.0,
        'max_message_length_agent': 90,
        'min_message_length_agent': 25,
        'total_chars_from_agent': 385,
        'question_count_agent': 2,
        'questions_per_agent_message': 0.3,
        'question_count_user': 3,
        'questions_per_user_message': 0.4,
        'sequential_user_messages': 1,
        'sequential_agent_messages': 1,
        'entities_collected_count': 3,
        'has_target_entity': 1,
        'avg_entity_confidence': 0.8,
        'min_entity_confidence': 0.6,
        'entity_collection_rate': 0.8,
        'repeated_questions': 0,
        'message_alternation_rate': 0.9,
        'time_to_conversion_seconds': 60.0,
        'time_to_conversion_minutes': 1.0
    }
    
    print(f"📊 Created test features: {len(test_features)} features")
    
    # Create synapse
    synapse = ConversionSynapse(features=test_features)
    print(f"✅ Created synapse with test_pk: {synapse.features['test_pk']}")
    
    # Create axon info for localhost miner
    axon_info = bt.AxonInfo(
        version=4,
        ip="127.0.0.1",  # Use localhost IP
        port=8091,
        ip_type=4,
        hotkey="5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL",
        coldkey="5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
    )
    
    print(f"🎯 Testing connection to {axon_info.ip}:{axon_info.port}")
    
    try:
        # Create wallet for dendrite (use existing validator wallet)
        wallet = bt.wallet(name="amitn_validator", hotkey="default")
        dendrite = bt.dendrite(wallet=wallet)
        
        # Query the miner directly (disable verification for testing)
        start_time = time.time()
        responses = await dendrite([axon_info], synapse, deserialize=False, timeout=30.0)
        end_time = time.time()
        
        print(f"⏱️  Query completed in {end_time - start_time:.2f}s")
        
        if responses and len(responses) > 0:
            response = responses[0]
            print(f"📦 Response type: {type(response)}")
            
            if hasattr(response, 'prediction') and response.prediction is not None:
                print(f"🎉 SUCCESS: Received prediction!")
                print(f"   Prediction: {response.prediction}")
                print(f"   Confidence: {getattr(response, 'confidence', 'N/A')}")
                return True
            else:
                print(f"❌ FAILED: No prediction in response")
                print(f"   Response: {response}")
                return False
        else:
            print(f"❌ FAILED: No response received")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting direct miner communication test...")
    result = asyncio.run(test_direct_miner_communication())
    
    if result:
        print("\n✅ SUCCESS: Direct communication works!")
        print("   The issue is likely with metagraph IP registration.")
    else:
        print("\n❌ FAILED: Direct communication failed.")
        print("   Check if miner is running and accessible.") 
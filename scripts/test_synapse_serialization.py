#!/usr/bin/env python3
"""
Test ConversionSynapse serialization/deserialization to debug feature loss.
"""

import json
from conversion_subnet.protocol import ConversionSynapse

def test_synapse_serialization():
    """Test if ConversionSynapse properly handles feature serialization."""
    
    print("🔍 Testing ConversionSynapse serialization...")
    
    # Create test features (same as validator sends)
    test_features = {
        'test_pk': 'serialization_test',
        'session_id': 'test_session',
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
    
    print(f"📊 Original features: {len(test_features)} features")
    
    # Test 1: Create synapse
    try:
        synapse = ConversionSynapse(features=test_features)
        print(f"✅ Synapse created successfully")
        print(f"📦 Synapse features count: {len(synapse.features)}")
        
        # Check if all features are preserved
        missing_features = set(test_features.keys()) - set(synapse.features.keys())
        extra_features = set(synapse.features.keys()) - set(test_features.keys())
        
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
        if extra_features:
            print(f"⚠️  Extra features: {extra_features}")
        if not missing_features and not extra_features:
            print(f"✅ All features preserved in synapse")
            
    except Exception as e:
        print(f"❌ Failed to create synapse: {e}")
        return False
    
    # Test 2: JSON serialization (simulates network transmission)
    try:
        # Convert to dict (simulates Bittensor serialization)
        synapse_dict = synapse.dict()
        print(f"📤 Serialized to dict: {len(synapse_dict.get('features', {}))} features")
        
        # Convert to JSON (simulates network transmission)
        json_str = json.dumps(synapse_dict)
        print(f"📡 JSON serialization successful: {len(json_str)} bytes")
        
        # Deserialize from JSON
        restored_dict = json.loads(json_str)
        print(f"📥 JSON deserialization successful")
        
        # Create new synapse from restored dict
        restored_synapse = ConversionSynapse(**restored_dict)
        print(f"🔄 Restored synapse: {len(restored_synapse.features)} features")
        
        # Check if features survived the round trip
        if len(restored_synapse.features) == len(test_features):
            print(f"✅ All features survived serialization round trip")
            return True
        else:
            print(f"❌ Feature loss during serialization: {len(test_features)} → {len(restored_synapse.features)}")
            missing = set(test_features.keys()) - set(restored_synapse.features.keys())
            print(f"❌ Lost features: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting ConversionSynapse serialization test...")
    success = test_synapse_serialization()
    
    if success:
        print("\n✅ SUCCESS: Serialization works correctly")
        print("   The issue is likely in Bittensor's network transmission")
    else:
        print("\n❌ FAILED: Serialization has issues")
        print("   This explains the feature loss problem") 
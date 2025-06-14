#!/usr/bin/env python3

"""
Test script for the challenge generation system
"""

try:
    print("Testing challenge generation...")
    
    from conversion_subnet.validator.send_challenge import ConversationChallengeGenerator
    print("✓ Imported ConversationChallengeGenerator successfully")
    
    gen = ConversationChallengeGenerator()
    print("✓ Generator created successfully")
    
    # Test single challenge generation
    challenge = gen.generate_challenge_conversation('high_engagement_conversion')
    print("✓ Single challenge generated successfully")
    print(f"  Challenge keys: {list(challenge.keys())}")
    print(f"  Target value: {challenge.get('target')}")
    print(f"  Total messages: {challenge.get('totalMessages')}")
    print(f"  Message ratio: {challenge.get('messageRatio')}")
    
    # Test batch generation
    challenges = gen.generate_batch_challenges(2)
    print(f"✓ Generated {len(challenges)} challenges in batch")
    
    # Test synapse creation
    if challenges:
        synapse = gen.create_synapse_from_features(challenges[0], miner_uid=1)
        print("✓ Synapse created successfully")
        print(f"  Synapse features count: {len(synapse.features)}")
    
    print("\n🎉 All tests passed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 
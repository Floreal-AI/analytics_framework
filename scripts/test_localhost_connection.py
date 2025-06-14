#!/usr/bin/env python3
"""
Test script to verify localhost connectivity between validator and miner.
Run this after starting both miner and validator to test the connection.
"""

import asyncio
import socket
import time
import requests
from conversion_subnet.protocol import ConversionSynapse
import bittensor as bt


def test_port_connectivity(host="127.0.0.1", port=8091, timeout=5):
    """Test if port is open and accepting connections."""
    print(f"Testing connection to {host}:{port}...")
    
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            print(f"✅ Port {port} is open and accepting connections!")
            return True
    except socket.error as e:
        print(f"❌ Cannot connect to {host}:{port}: {e}")
        return False


def test_http_endpoint(host="127.0.0.1", port=8091):
    """Test if the miner's HTTP endpoint is responding."""
    print(f"Testing HTTP endpoint at http://{host}:{port}...")
    
    try:
        # Try a simple GET request to see if the axon is serving
        response = requests.get(f"http://{host}:{port}/", timeout=10)
        print(f"✅ HTTP endpoint responding with status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP endpoint not responding: {e}")
        return False


async def test_dendrite_connection():
    """Test creating a dendrite connection to the miner."""
    print("Testing dendrite connection...")
    
    try:
        # Create a basic config for testing
        config = bt.config()
        
        # Create a wallet (this won't work without actual wallet files, but that's ok for connection test)
        wallet = bt.wallet(config=config)
        
        # Create dendrite
        dendrite = bt.dendrite(wallet=wallet)
        
        # Create a test synapse
        synapse = ConversionSynapse()
        synapse.features = {
            'test_pk': 'localhost-test',
            'conversation_duration_seconds': 120.0,
            'conversation_duration_minutes': 2.0,
            'hour_of_day': 14,
            'day_of_week': 1,
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
            'conversation_intensity': 0.75,
            'response_consistency': 0.85,
            'engagement_level': 0.8,
            'conversation_depth': 0.7,
            'interaction_quality': 0.9,
            'sentiment_consistency': 0.6,
            'topic_coherence': 0.8,
            'conversation_flow': 0.85,
            'user_satisfaction_signals': 0.7,
            'agent_effectiveness': 0.8,
            'time_to_conversion_seconds': 300.0,
            'time_to_conversion_minutes': 5.0
        }
        
        print("✅ Dendrite and synapse created successfully!")
        print("Note: Actual dendrite call would require proper wallet configuration")
        return True
        
    except Exception as e:
        print(f"❌ Dendrite connection test failed: {e}")
        return False


def main():
    """Run all localhost connectivity tests."""
    print("=" * 60)
    print("LOCALHOST CONNECTIVITY TEST")
    print("=" * 60)
    print()
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Port connectivity
    if test_port_connectivity():
        tests_passed += 1
    print()
    
    # Test 2: HTTP endpoint
    if test_http_endpoint():
        tests_passed += 1
    print()
    
    # Test 3: Dendrite setup
    if asyncio.run(test_dendrite_connection()):
        tests_passed += 1
    print()
    
    # Summary
    print("=" * 60)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your localhost setup should work.")
        print()
        print("Next steps:")
        print("1. Start miner:     ./scripts/run_miner_localhost.sh")
        print("2. Start validator: ./scripts/run_validator_localhost.sh")
        print("3. Monitor logs for successful communication")
    else:
        print("❌ Some tests failed. Check your setup:")
        print("1. Is the miner running with --axon.external_ip 127.0.0.1?")
        print("2. Is port 8091 available?")
        print("3. Are there any firewall restrictions?")


if __name__ == "__main__":
    main() 
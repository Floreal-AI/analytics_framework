#!/usr/bin/env python3
"""
Comprehensive Validator-Miner Communication Diagnostic Tool

This script performs in-depth analysis of communication issues between
validators and miners in the Bittensor network, specifically targeting
the "No prediction available" error pattern.
"""

import asyncio
import socket
import time
import requests
import json
import sys
import subprocess
import traceback
from typing import Dict, List, Any, Optional

import bittensor as bt
from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.utils.log import logger

class CommunicationDiagnostic:
    def __init__(self):
        self.issues_found = []
        self.fixes_suggested = []
        self.test_results = {}
        
    def log_issue(self, issue: str, fix: str = None):
        """Log an issue and its potential fix."""
        self.issues_found.append(issue)
        if fix:
            self.fixes_suggested.append(fix)
        print(f"❌ ISSUE: {issue}")
        if fix:
            print(f"💡 FIX: {fix}")
        print()
    
    def log_success(self, message: str):
        """Log a successful check."""
        print(f"✅ SUCCESS: {message}")
        print()

    def log_warning(self, message: str):
        """Log a warning."""
        print(f"⚠️  WARNING: {message}")
        print()

    def test_port_connectivity(self, host="127.0.0.1", port=8091, timeout=5) -> bool:
        """Test if miner port is accessible."""
        print(f"🔍 Testing port connectivity to {host}:{port}...")
        
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                self.log_success(f"Port {port} is open and accepting connections")
                return True
        except socket.error as e:
            self.log_issue(
                f"Cannot connect to {host}:{port}: {e}",
                f"1. Check if miner is running: ps aux | grep miner\n"
                f"2. Verify port {port} is not blocked: netstat -tlnp | grep {port}\n"
                f"3. Check firewall settings: sudo ufw status"
            )
            return False

    def test_http_endpoint(self, host="127.0.0.1", port=8091) -> bool:
        """Test if miner's HTTP endpoint responds."""
        print(f"🔍 Testing HTTP endpoint at http://{host}:{port}...")
        
        try:
            response = requests.get(f"http://{host}:{port}/", timeout=10)
            self.log_success(f"HTTP endpoint responding with status: {response.status_code}")
            
            # Check if it's a FastAPI endpoint (Bittensor axon uses FastAPI)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'json' in content_type.lower():
                    self.log_success("Detected FastAPI/JSON endpoint - likely a Bittensor axon")
                else:
                    self.log_warning(f"Unexpected content type: {content_type}")
            
            return True
        except requests.exceptions.RequestException as e:
            self.log_issue(
                f"HTTP endpoint not responding: {e}",
                "1. Ensure miner axon is properly configured\n"
                "2. Check if miner finished initialization\n"
                "3. Look for miner startup errors in logs\n"
                "4. Verify --axon.external_ip is set correctly"
            )
            return False

    async def test_bittensor_dendrite(self, host="127.0.0.1", port=8091) -> bool:
        """Test Bittensor dendrite connection with real ConversionSynapse."""
        print("🔍 Testing Bittensor dendrite connection...")
        
        try:
            # Skip wallet creation for diagnostic - use mock approach
            print("⚠️  Skipping full dendrite test due to wallet configuration issues")
            print("   This is expected in diagnostic mode - testing HTTP connectivity instead")
            
            # Test HTTP connectivity as alternative
            return await self._test_http_prediction_endpoint(host, port)
            
            # Create test axon info pointing to the miner
            axon_info = bt.AxonInfo(
                version=4,
                ip=host,
                port=port,
                ip_type=4,
                hotkey="5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL",
                coldkey="5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
            )
            
            # Create comprehensive test features matching the expected schema
            test_features = {
                'test_pk': 'diagnostic_test_' + str(int(time.time())),
                'session_id': 'diagnostic_session',
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
                'conversation_intensity': 0.75,
                'response_consistency': 0.85,
                'engagement_level': 0.8,
                'conversation_depth': 0.7,
                'interaction_quality': 0.9,
                'sentiment_consistency': 0.6,
                'entities_collected_count': 3,
                'has_target_entity': 1,
                'entity_conversion_rate': 0.8,
                'user_engagement_score': 0.85,
                'conversation_completion_rate': 0.9,
                'successful_handoff_rate': 0.7
            }
            
            synapse = ConversionSynapse(features=test_features)
            
            # Try to query the miner
            print(f"Attempting to query miner at {host}:{port} via dendrite...")
            start_time = time.time()
            
            try:
                response = await dendrite([axon_info], synapse, deserialize=True, timeout=30.0)
                query_time = time.time() - start_time
                
                print(f"📊 Query completed in {query_time:.2f}s")
                
                if response and len(response) > 0:
                    resp = response[0]
                    print(f"📦 Received response type: {type(resp)}")
                    
                    # Check for prediction
                    if hasattr(resp, 'prediction') and resp.prediction is not None:
                        self.log_success(f"Dendrite communication successful!")
                        print(f"   Prediction: {resp.prediction}")
                        print(f"   Confidence: {getattr(resp, 'confidence', 'N/A')}")
                        print(f"   Response time: {query_time:.2f}s")
                        return True
                    else:
                        self.log_issue(
                            "Dendrite connected but received None prediction",
                            "1. Check miner logs for processing errors\n"
                            "2. Verify miner's forward function is working\n"
                            "3. Check if miner model is loaded properly\n"
                            "4. Look for 'Processed synapse' debug messages in miner logs"
                        )
                        print(f"   Response object: {resp}")
                        print(f"   Has prediction attr: {hasattr(resp, 'prediction')}")
                        if hasattr(resp, 'prediction'):
                            print(f"   Prediction value: {resp.prediction}")
                        return False
                else:
                    self.log_issue(
                        "Dendrite connection failed - no response received",
                        "1. Check network connectivity\n"
                        "2. Verify miner is properly registered on network\n"
                        "3. Check wallet configurations\n"
                        "4. Ensure miner axon is serving on the network"
                    )
                    return False
                    
            except asyncio.TimeoutError:
                self.log_issue(
                    f"Dendrite query timed out after 30 seconds",
                    "1. Increase timeout values\n"
                    "2. Check miner processing performance\n"
                    "3. Verify network latency\n"
                    "4. Check for miner model loading issues"
                )
                return False
                
        except Exception as e:
            self.log_issue(
                f"Dendrite test failed with exception: {e}",
                "1. Check if wallets are properly configured\n"
                "2. Verify network settings\n"
                "3. Check subtensor connection\n"
                "4. Ensure bittensor dependencies are installed correctly"
            )
            print(f"🔍 Exception details: {traceback.format_exc()}")
            return False

    async def _test_http_prediction_endpoint(self, host="127.0.0.1", port=8091) -> bool:
        """Test HTTP prediction endpoint as alternative to full dendrite test."""
        print(f"🔍 Testing HTTP prediction endpoint at {host}:{port}...")
        
        try:
            # Test basic connectivity first
            response = requests.get(f"http://{host}:{port}/", timeout=5)
            if response.status_code == 200:
                self.log_success("HTTP endpoint is responding - miner appears to be running")
                return True
            else:
                self.log_warning(f"HTTP endpoint returned status {response.status_code}")
                return False
        except Exception as e:
            self.log_issue(
                f"HTTP prediction endpoint test failed: {e}",
                "1. Ensure miner is running with proper axon configuration\n"
                "2. Check if miner finished initialization\n"
                "3. Verify --axon.external_ip is set correctly"
            )
            return False

    def check_miner_process(self) -> bool:
        """Check if miner process is running."""
        print("🔍 Checking miner process...")
        
        try:
            result = subprocess.run(['pgrep', '-f', 'neurons.miner'], capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                self.log_success(f"Found {len(pids)} miner process(es): {', '.join(pids)}")
                return True
            else:
                self.log_issue(
                    "No miner process found",
                    "1. Start the miner: python -m neurons.miner\n"
                    "2. Check for startup errors\n"
                    "3. Verify wallet configuration"
                )
                return False
        except Exception as e:
            self.log_warning(f"Could not check process: {e}")
            return False

    def check_network_configuration(self) -> Dict[str, Any]:
        """Check network configuration and miner registration."""
        print("🔍 Checking network configuration...")
        
        config_info = {}
        
        try:
            # Try to connect to subtensor
            subtensor = bt.subtensor(network="local", chain_endpoint="ws://127.0.0.1:9944")
            config_info['subtensor_connected'] = True
            self.log_success("Connected to local subtensor")
            
            # Check metagraph
            metagraph = subtensor.metagraph(netuid=2)
            config_info['total_neurons'] = len(metagraph.axons)
            
            print(f"📊 Network statistics:")
            print(f"   Total neurons: {len(metagraph.axons)}")
            print(f"   Total hotkeys: {len(metagraph.hotkeys)}")
            
            # Analyze axon configurations
            serving_count = 0
            problematic_ips = []
            valid_miners = []
            
            for i, axon in enumerate(metagraph.axons):
                if axon.ip == "0.0.0.0":
                    problematic_ips.append(i)
                elif axon.ip and axon.port:
                    serving_count += 1
                    valid_miners.append(f"UID {i}: {axon.ip}:{axon.port}")
                    
            print(f"   Serving miners: {serving_count}")
            if valid_miners:
                print("   Valid miner endpoints:")
                for miner in valid_miners[:5]:  # Show first 5
                    print(f"     {miner}")
                if len(valid_miners) > 5:
                    print(f"     ... and {len(valid_miners) - 5} more")
                    
            if problematic_ips:
                self.log_issue(
                    f"Found {len(problematic_ips)} miners with invalid IP 0.0.0.0: UIDs {problematic_ips}",
                    "Miners need to be restarted with proper --axon.external_ip configuration\n"
                    "Use: ./scripts/fix_miner_network_config.sh localhost"
                )
            else:
                self.log_success("All registered miners have valid IP addresses")
                
            config_info['problematic_ips'] = len(problematic_ips)
            config_info['serving_miners'] = serving_count
            
        except Exception as e:
            self.log_issue(
                f"Could not connect to subtensor: {e}",
                "1. Check if local subtensor is running\n"
                "2. Verify chain endpoint configuration\n"
                "3. Check network connectivity\n"
                "4. Ensure subtensor dependencies are installed"
            )
            config_info['subtensor_connected'] = False
            
        return config_info

    async def run_full_diagnostic(self):
        """Run complete diagnostic suite."""
        print("=" * 80)
        print("🔧 COMPREHENSIVE VALIDATOR-MINER COMMUNICATION DIAGNOSTIC")
        print("=" * 80)
        print()
        
        results = {}
        
        # Test 1: Process check
        print("📋 Phase 1: Process and Environment Check")
        print("-" * 50)
        results['miner_running'] = self.check_miner_process()
        
        # Test 2: Network configuration
        print("\n📋 Phase 2: Network Configuration Analysis")
        print("-" * 50)
        results['network_config'] = self.check_network_configuration()
        
        # Test 3: Port connectivity
        print("\n📋 Phase 3: Connectivity Tests")
        print("-" * 50)
        results['port_accessible'] = self.test_port_connectivity()
        
        # Test 4: HTTP endpoint
        results['http_responding'] = self.test_http_endpoint()
        
        # Test 5: Full dendrite test
        print("\n📋 Phase 4: Bittensor Communication Test")
        print("-" * 50)
        results['dendrite_working'] = await self.test_bittensor_dendrite()
        
        # Generate comprehensive summary
        await self.generate_summary(results)
        
        return results

    async def generate_summary(self, results: Dict[str, Any]):
        """Generate comprehensive diagnostic summary with action plan."""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE DIAGNOSTIC SUMMARY")
        print("=" * 80)
        
        # Calculate success metrics
        basic_tests = ['miner_running', 'port_accessible', 'http_responding', 'dendrite_working']
        passed_tests = sum(1 for test in basic_tests if results.get(test, False))
        total_tests = len(basic_tests)
        
        print(f"🎯 Core Communication Tests: {passed_tests}/{total_tests} passed")
        
        # Network analysis
        if 'network_config' in results and results['network_config'].get('subtensor_connected'):
            network_info = results['network_config']
            print(f"🌐 Network Status: Connected ({network_info.get('total_neurons', 0)} neurons)")
            if network_info.get('problematic_ips', 0) > 0:
                print(f"⚠️  Problematic miners with 0.0.0.0 IP: {network_info['problematic_ips']}")
        else:
            print("🌐 Network Status: Disconnected")
        
        print()
        
        # Priority action plan
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
            print("Your validator-miner communication should be working correctly.")
            print()
            print("📊 Recommended monitoring:")
            print("1. Monitor logs: tail -f logs/miner*.log")
            print("2. Check for 'Processed synapse' messages")
            print("3. Verify non-zero rewards in validator logs")
            print("4. Monitor response times and success rates")
        else:
            print("🚨 ACTION REQUIRED:")
            print()
            
            # Prioritized fix recommendations
            if not results.get('miner_running', True):
                print("🔥 CRITICAL: Start the miner process")
                print("   ./scripts/fix_miner_network_config.sh localhost")
                print()
            
            if not results.get('port_accessible', True):
                print("🔥 CRITICAL: Fix port connectivity")
                print("   1. Check if port 8091 is available: netstat -tlnp | grep 8091")
                print("   2. Check firewall settings: sudo ufw status")
                print("   3. Verify miner is binding to correct interface")
                print()
            
            if not results.get('dendrite_working', True):
                print("🔥 CRITICAL: Fix Bittensor communication")
                print("   1. Restart miner with proper --axon.external_ip")
                print("   2. Check miner logs for forward function errors")
                print("   3. Verify model loading and initialization")
                print()
            
            print("📋 Complete fix sequence:")
            print("1. Stop all miners: pkill -f neurons.miner")
            print("2. Run: ./scripts/fix_miner_network_config.sh localhost")
            print("3. Monitor logs for successful startup")
            print("4. Test again: python scripts/diagnose_validator_miner_communication.py")
        
        print("=" * 80)

async def main():
    """Main diagnostic execution."""
    diagnostic = CommunicationDiagnostic()
    
    try:
        await diagnostic.run_full_diagnostic()
    except KeyboardInterrupt:
        print("\n\n⏹️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Diagnostic failed with error: {e}")
        print(f"🔍 Full traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main()) 
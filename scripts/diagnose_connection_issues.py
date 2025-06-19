#!/usr/bin/env python3
"""
Comprehensive diagnostic script for validator-miner communication issues.
Run this to identify why miners are returning None predictions.
"""

import asyncio
import socket
import time
import requests
import json
import sys
import subprocess
from typing import Dict, List, Any

import bittensor as bt
from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.utils.log import logger

class ConnectionDiagnostic:
    def __init__(self):
        self.issues_found = []
        self.fixes_suggested = []
        
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
                f"1. Check if miner is running\n2. Verify port {port} is not blocked by firewall\n3. Use: netstat -tlnp | grep {port}"
            )
            return False

    def test_http_endpoint(self, host="127.0.0.1", port=8091) -> bool:
        """Test if miner's HTTP endpoint responds."""
        print(f"🔍 Testing HTTP endpoint at http://{host}:{port}...")
        
        try:
            response = requests.get(f"http://{host}:{port}/", timeout=10)
            self.log_success(f"HTTP endpoint responding with status: {response.status_code}")
            return True
        except requests.exceptions.RequestException as e:
            self.log_issue(
                f"HTTP endpoint not responding: {e}",
                "1. Ensure miner axon is properly configured\n2. Check if miner finished initialization\n3. Look for miner startup errors"
            )
            return False

    async def test_bittensor_dendrite(self) -> bool:
        """Test Bittensor dendrite connection."""
        print("🔍 Testing Bittensor dendrite connection...")
        
        try:
            # Create test wallet (read-only)
            wallet = bt.wallet()
            
            # Create dendrite
            dendrite = bt.dendrite(wallet=wallet)
            
            # Create test axon info
            axon_info = bt.AxonInfo(
                version=4,
                ip="127.0.0.1",
                port=8091,
                ip_type=4,
                hotkey="5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL",
                coldkey="5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
            )
            
            # Create test synapse
            test_features = {
                'session_id': 'diagnostic_test',
                'conversation_duration_seconds': 120.0,
                'has_target_entity': 1,
                'entities_collected_count': 3,
                'message_ratio': 1.2,
                'hour_of_day': 14,
                'day_of_week': 2,
                'is_business_hours': 1,
                'total_messages': 20,
                'user_messages_count': 10,
                'agent_messages_count': 10,
                'test_pk': 'diagnostic_test_pk'
            }
            
            synapse = ConversionSynapse(features=test_features)
            
            # Try to query the miner
            print("Attempting to query miner via dendrite...")
            response = await dendrite([axon_info], synapse, deserialize=True, timeout=30.0)
            
            if response and len(response) > 0:
                resp = response[0]
                if hasattr(resp, 'prediction') and resp.prediction is not None:
                    self.log_success(f"Dendrite connection successful! Received prediction: {resp.prediction}")
                    return True
                else:
                    self.log_issue(
                        "Dendrite connected but received None prediction",
                        "1. Check miner logs for processing errors\n2. Verify miner's forward function is working\n3. Check if miner model is loaded properly"
                    )
                    return False
            else:
                self.log_issue(
                    "Dendrite connection failed - no response received",
                    "1. Check network connectivity\n2. Verify miner is registered on network\n3. Check wallet configurations match"
                )
                return False
                
        except Exception as e:
            self.log_issue(
                f"Dendrite test failed with exception: {e}",
                "1. Check if wallets are properly configured\n2. Verify network settings\n3. Check subtensor connection"
            )
            return False

    def check_miner_process(self) -> bool:
        """Check if miner process is running."""
        print("🔍 Checking for running miner processes...")
        
        try:
            # Check for python processes running miner
            result = subprocess.run(['pgrep', '-f', 'neurons.miner'], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                self.log_success(f"Found {len(pids)} miner process(es) running: {', '.join(pids)}")
                return True
            else:
                self.log_issue(
                    "No miner processes found running",
                    "1. Start the miner using: ./scripts/fix_miner_connection.sh localhost\n2. Check for startup errors in logs"
                )
                return False
                
        except Exception as e:
            print(f"⚠️  Could not check processes: {e}")
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
            print(f"   Active miners: {sum(1 for axon in metagraph.axons if axon.is_serving)}")
            
            # Check for problematic IP addresses
            problematic_ips = []
            for i, axon in enumerate(metagraph.axons):
                if axon.ip == "0.0.0.0":
                    problematic_ips.append(i)
                    
            if problematic_ips:
                self.log_issue(
                    f"Found {len(problematic_ips)} miners with invalid IP 0.0.0.0: UIDs {problematic_ips}",
                    "Miners need to be restarted with proper --axon.external_ip configuration"
                )
            else:
                self.log_success("All miners have valid IP addresses")
                
            config_info['problematic_ips'] = len(problematic_ips)
            
        except Exception as e:
            self.log_issue(
                f"Could not connect to subtensor: {e}",
                "1. Check if local subtensor is running\n2. Verify chain endpoint configuration\n3. Check network connectivity"
            )
            config_info['subtensor_connected'] = False
            
        return config_info

    def check_environment_variables(self) -> bool:
        """Check required environment variables."""
        print("🔍 Checking environment variables...")
        
        required_vars = [
            'VOICEFORM_API_BASE_URL',
            'VOICEFORM_API_KEY',
            'VOICEFORM_VALIDATION_API_KEY'
        ]
        
        missing_vars = []
        
        import os
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            self.log_issue(
                f"Missing environment variables: {', '.join(missing_vars)}",
                "Create .env file with required API credentials"
            )
            return False
        else:
            self.log_success("All required environment variables are set")
            return True

    async def run_full_diagnostic(self):
        """Run complete diagnostic suite."""
        print("=" * 80)
        print("🔧 VALIDATOR-MINER CONNECTION DIAGNOSTIC")
        print("=" * 80)
        print()
        
        results = {}
        
        # Test 1: Process check
        results['miner_running'] = self.check_miner_process()
        
        # Test 2: Port connectivity
        results['port_accessible'] = self.test_port_connectivity()
        
        # Test 3: HTTP endpoint
        results['http_responding'] = self.test_http_endpoint()
        
        # Test 4: Environment variables
        results['env_vars_ok'] = self.check_environment_variables()
        
        # Test 5: Network configuration
        results['network_config'] = self.check_network_configuration()
        
        # Test 6: Full dendrite test
        results['dendrite_working'] = await self.test_bittensor_dendrite()
        
        # Summary
        print("=" * 80)
        print("📋 DIAGNOSTIC SUMMARY")
        print("=" * 80)
        
        passed_tests = sum(1 for k, v in results.items() if v == True or (isinstance(v, dict) and v.get('subtensor_connected', False)))
        total_tests = len([k for k in results.keys() if k != 'network_config']) + 1  # +1 for network config
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print()
        
        if self.issues_found:
            print("🚨 ISSUES FOUND:")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"{i}. {issue}")
            print()
            
            print("💡 SUGGESTED FIXES:")
            for i, fix in enumerate(self.fixes_suggested, 1):
                print(f"{i}. {fix}")
        else:
            print("🎉 No issues found! Communication should be working.")
            
        print("=" * 80)
        
        return results

async def main():
    """Main diagnostic function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python scripts/diagnose_connection_issues.py")
        print("This script diagnoses validator-miner communication issues.")
        print("\nWhat it checks:")
        print("1. Miner process status")
        print("2. Port connectivity (8091)")
        print("3. HTTP endpoint response")
        print("4. Environment variables")
        print("5. Network configuration")
        print("6. Full dendrite communication test")
        return
    
    diagnostic = ConnectionDiagnostic()
    await diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    asyncio.run(main()) 
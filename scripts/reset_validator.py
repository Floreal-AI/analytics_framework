#!/usr/bin/env python3
# Reset validator state and restart it

import os
import sys
import time
import shutil
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Reset validator state and restart it")
    parser.add_argument(
        "--state_dir", 
        type=str, 
        default="~/.bittensor/validators/",
        help="Directory where validator state is stored"
    )
    parser.add_argument(
        "--netuid", 
        type=str, 
        default="2",
        help="Subnet UID"
    )
    parser.add_argument(
        "--wallet_name", 
        type=str, 
        default="validator",
        help="Validator wallet name"
    )
    parser.add_argument(
        "--wallet_hotkey", 
        type=str, 
        default="default",
        help="Validator wallet hotkey"
    )
    parser.add_argument(
        "--restart", 
        action="store_true",
        help="Restart the validator after resetting state"
    )
    return parser.parse_args()

def reset_validator_state(state_dir, netuid, wallet_name, wallet_hotkey):
    """Reset validator state by removing state files"""
    
    # Expand user directory if needed
    state_dir = os.path.expanduser(state_dir)
    
    # Construct the path to the validator state directory
    validator_path = os.path.join(
        state_dir, 
        f"netuid{netuid}", 
        wallet_name, 
        wallet_hotkey
    )
    
    print(f"Checking for validator state at: {validator_path}")
    
    if os.path.exists(validator_path):
        # Backup the directory first
        backup_path = f"{validator_path}_backup_{int(time.time())}"
        print(f"Creating backup at: {backup_path}")
        shutil.copytree(validator_path, backup_path)
        
        # Remove state files
        for filename in os.listdir(validator_path):
            if filename.endswith('.pt') or filename.endswith('.db'):
                file_path = os.path.join(validator_path, filename)
                print(f"Removing state file: {file_path}")
                os.remove(file_path)
        
        print("Validator state reset successfully")
        return True
    else:
        print(f"Validator state directory not found at: {validator_path}")
        return False

def restart_validator():
    """Restart the validator"""
    
    # Check if validator is running
    try:
        # Find the validator process
        result = subprocess.run(
            "ps aux | grep 'neurons/validator.py' | grep -v grep",
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.stdout:
            # Kill the validator process
            print("Stopping validator...")
            subprocess.run(
                "pkill -f 'neurons/validator.py'",
                shell=True,
                check=False
            )
            time.sleep(2)  # Give it time to shut down
        
        # Start the validator
        print("Starting validator...")
        subprocess.Popen(
            "python neurons/validator.py --netuid 2 --subtensor.network local",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("Validator restarted successfully")
        return True
    except Exception as e:
        print(f"Error restarting validator: {e}")
        return False

if __name__ == "__main__":
    args = parse_args()
    
    # Reset validator state
    reset_success = reset_validator_state(
        args.state_dir,
        args.netuid,
        args.wallet_name,
        args.wallet_hotkey
    )
    
    # Restart validator if requested and reset was successful
    if args.restart and reset_success:
        restart_validator() 
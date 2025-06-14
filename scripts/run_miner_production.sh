#!/bin/bash
# Run miner configured for production (cloud deployment)

echo "Starting Miner for Production..."
echo "================================="

# Get the external IP address
EXTERNAL_IP=$(curl -s ifconfig.me || curl -s ipinfo.io/ip || curl -s icanhazip.com)

if [ -z "$EXTERNAL_IP" ]; then
    echo "ERROR: Could not determine external IP address"
    echo "Please manually set the external IP using: --axon.external_ip YOUR_PUBLIC_IP"
    exit 1
fi

echo "Detected external IP: $EXTERNAL_IP"

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the miner with production configuration
python -m neurons.miner \
  --netuid 2 \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --subtensor.network local \
  --wallet.name amitn_miner \
  --wallet.hotkey default \
  --axon.port 8091 \
  --axon.ip 0.0.0.0 \
  --logging.debug \
  --neuron.device cpu

echo "Miner stopped." 
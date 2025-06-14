#!/bin/bash
# Run miner configured for localhost testing

echo "Starting Miner for Localhost Testing..."
echo "======================================="

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the miner with localhost configuration
python -m neurons.miner \
  --netuid 2 \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --subtensor.network local \
  --wallet.name amitn_miner \
  --wallet.hotkey default \
  --axon.external_ip 127.0.0.1 \
  --axon.port 8091 \
  --axon.ip 0.0.0.0 \
  --logging.debug \
  --neuron.device cpu

echo "Miner stopped." 
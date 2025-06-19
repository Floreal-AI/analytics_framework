#!/bin/bash
# Fixed miner configuration script to resolve connection issues

echo "==================================================================="
echo "FIXING MINER CONNECTION CONFIGURATION"
echo "==================================================================="

# Check if we're running locally or on a server
echo "Detecting network environment..."

# For localhost testing
if [[ "$1" == "localhost" ]]; then
    echo "Configuring for LOCALHOST testing..."
    EXTERNAL_IP="127.0.0.1"
    BIND_IP="127.0.0.1"
    NETWORK="local"
    ENDPOINT="ws://127.0.0.1:9944"
    
# For production/server deployment
else
    echo "Configuring for PRODUCTION deployment..."
    
    # Try to get external IP automatically
    EXTERNAL_IP=$(curl -s --connect-timeout 5 ifconfig.me 2>/dev/null || curl -s --connect-timeout 5 ipinfo.io/ip 2>/dev/null || curl -s --connect-timeout 5 icanhazip.com 2>/dev/null)
    
    if [ -z "$EXTERNAL_IP" ]; then
        echo "❌ ERROR: Could not automatically detect external IP"
        echo "Please run with your public IP:"
        echo "./scripts/fix_miner_connection.sh production YOUR_PUBLIC_IP"
        exit 1
    fi
    
    # Allow manual IP override
    if [ ! -z "$2" ]; then
        EXTERNAL_IP="$2"
        echo "Using manually specified IP: $EXTERNAL_IP"
    fi
    
    BIND_IP="0.0.0.0"
    NETWORK="finney"  # or "test" for testnet
    ENDPOINT="wss://entrypoint-finney.opentensor.ai:443"
    
fi

echo "Configuration:"
echo "  External IP: $EXTERNAL_IP"
echo "  Bind IP: $BIND_IP"
echo "  Network: $NETWORK"
echo "  Endpoint: $ENDPOINT"
echo ""

# Validate IP format
if [[ ! $EXTERNAL_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ ERROR: Invalid IP format: $EXTERNAL_IP"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "==================================================================="
echo "STARTING MINER WITH FIXED CONFIGURATION"
echo "==================================================================="

# Start miner with proper configuration
python -m neurons.miner \
  --netuid 2 \
  --subtensor.chain_endpoint "$ENDPOINT" \
  --subtensor.network "$NETWORK" \
  --wallet.name amitn_miner \
  --wallet.hotkey default \
  --axon.external_ip "$EXTERNAL_IP" \
  --axon.port 8091 \
  --axon.ip "$BIND_IP" \
  --logging.debug \
  --logging.trace \
  --neuron.device cpu

echo "Miner stopped." 
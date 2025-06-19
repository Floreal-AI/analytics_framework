#!/bin/bash

# Comprehensive Miner Network Configuration Fix
# This script addresses the core communication issues between validators and miners

set -e

echo "🔧 BITTENSOR MINER NETWORK CONFIGURATION FIX"
echo "============================================="

# Function to detect environment
detect_environment() {
    if [ "$1" = "localhost" ] || [ "$1" = "local" ]; then
        echo "localhost"
    elif [ "$1" = "production" ] || [ "$1" = "prod" ]; then
        echo "production"
    else
        # Auto-detect based on network interfaces
        if ip route | grep -q "default via 192.168\|default via 10\|default via 172"; then
            echo "localhost"
        else
            echo "production"
        fi
    fi
}

# Function to get external IP
get_external_ip() {
    local env=$1
    
    if [ "$env" = "localhost" ]; then
        echo "127.0.0.1"
    else
        # Production environment - get real external IP
        external_ip=$(curl -s ifconfig.me 2>/dev/null || curl -s ipecho.net/plain 2>/dev/null || curl -s icanhazip.com 2>/dev/null)
        if [ -z "$external_ip" ]; then
            echo "❌ ERROR: Could not determine external IP automatically"
            echo "Please specify your external IP manually:"
            read -p "Enter external IP: " external_ip
        fi
        echo "$external_ip"
    fi
}

# Function to validate IP format
validate_ip() {
    local ip=$1
    if [[ $ip =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to check port availability
check_port() {
    local port=$1
    if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
        echo "⚠️  Port $port is already in use"
        netstat -tlnp | grep ":$port "
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

# Function to test connectivity
test_connectivity() {
    local ip=$1
    local port=$2
    
    echo "🔍 Testing connectivity to $ip:$port..."
    
    # Wait a moment for the miner to start
    sleep 3
    
    # Test port connectivity
    if timeout 5 nc -z "$ip" "$port" 2>/dev/null; then
        echo "✅ Port connectivity successful"
        return 0
    else
        echo "❌ Port connectivity failed"
        return 1
    fi
}

# Main execution
main() {
    # Parse environment argument
    ENV=$(detect_environment "$1")
    echo "🌍 Detected environment: $ENV"
    
    # Get external IP
    EXTERNAL_IP=$(get_external_ip "$ENV")
    echo "📍 External IP: $EXTERNAL_IP"
    
    # Validate IP
    if ! validate_ip "$EXTERNAL_IP"; then
        echo "❌ ERROR: Invalid IP format: $EXTERNAL_IP"
        exit 1
    fi
    
    # Set bind IP (always 0.0.0.0 for the axon to listen on all interfaces)
    BIND_IP="0.0.0.0"
    
    # Set network configuration
    if [ "$ENV" = "localhost" ]; then
        NETWORK="local"
        ENDPOINT="ws://127.0.0.1:9944"
    else
        NETWORK="finney"
        ENDPOINT="wss://entrypoint-finney.opentensor.ai:443"
    fi
    
    echo "⚙️  Configuration:"
    echo "   Network: $NETWORK"
    echo "   Endpoint: $ENDPOINT"
    echo "   Bind IP: $BIND_IP"
    echo "   External IP: $EXTERNAL_IP"
    echo "   Port: 8091"
    
    # Check port availability
    if ! check_port 8091; then
        echo "❌ Port 8091 is not available. Please stop other services or choose a different port."
        exit 1
    fi
    
    # Create logs directory
    mkdir -p logs
    
    echo ""
    echo "🚀 Starting miner with corrected network configuration..."
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
        --neuron.device cpu \
        2>&1 | tee logs/miner_$(date +%Y%m%d_%H%M%S).log &
    
    MINER_PID=$!
    echo "📝 Miner started with PID: $MINER_PID"
    echo "📋 Logs: logs/miner_$(date +%Y%m%d_%H%M%S).log"
    
    # Test connectivity after startup
    sleep 5
    if test_connectivity "$EXTERNAL_IP" 8091; then
        echo "🎉 Miner configuration successful!"
        echo ""
        echo "📊 Next steps:"
        echo "1. Monitor logs: tail -f logs/miner_*.log"
        echo "2. Start validator: python -m neurons.validator"
        echo "3. Check for 'Processed synapse' messages in miner logs"
        echo "4. Verify non-zero rewards in validator logs"
    else
        echo "❌ Connectivity test failed. Check firewall and network settings."
        kill $MINER_PID 2>/dev/null || true
        exit 1
    fi
    
    # Keep script running
    echo ""
    echo "⏸️  Press Ctrl+C to stop the miner..."
    wait $MINER_PID
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [localhost|production]"
    echo ""
    echo "Examples:"
    echo "  $0 localhost    # For local development"
    echo "  $0 production   # For production deployment"
    echo "  $0              # Auto-detect environment"
    echo ""
fi

# Run main function
main "$@" 
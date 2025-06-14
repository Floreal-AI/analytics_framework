# Localhost Testing Guide

This guide explains how to run the validator and miner on the same machine for development and testing.

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ and all dependencies installed
2. **Wallets**: Create separate wallets for miner and validator
3. **API Configuration**: Set up your environment variables for external APIs

## Quick Setup

### Step 1: Create Wallets

```bash
# Create miner wallet
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# Create validator wallet  
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

### Step 2: Register on Testnet (Optional)

For full testing, register both on testnet:

```bash
# Register miner
btcli subnet register --wallet.name miner --wallet.hotkey default --netuid 1 --subtensor.network test

# Register validator
btcli subnet register --wallet.name validator --wallet.hotkey default --netuid 1 --subtensor.network test
```

### Step 3: Configure Environment Variables

Create a `.env` file with your API credentials:

```bash
# API Configuration
VOICEFORM_API_BASE_URL=https://your-api-url.com
VOICEFORM_API_KEY=your_api_key_here
VALIDATION_API_TIMEOUT=30.0
DATA_API_TIMEOUT=30.0
```

### Step 4: Run Components

#### Terminal 1 - Start Miner
```bash
./scripts/run_miner_localhost.sh
```

#### Terminal 2 - Start Validator  
```bash
./scripts/run_validator_localhost.sh
```

## Key Configuration Details

### Miner Configuration

The critical setting for localhost is `--axon.external_ip 127.0.0.1`:

```bash
python -m neurons.miner \
  --axon.external_ip 127.0.0.1 \  # This tells other nodes to connect to localhost
  --axon.port 8091 \               # Port to listen on
  --axon.ip 0.0.0.0 \             # Bind to all interfaces locally
  --netuid 1 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

### Validator Configuration

The validator doesn't need special IP configuration - it reads miner addresses from the metagraph:

```bash
python -m neurons.validator \
  --netuid 1 \
  --subtensor.network test \
  --wallet.name validator \
  --wallet.hotkey default \
  --neuron.sample_size 1 \        # Query only 1 miner for testing
  --logging.debug
```

## Troubleshooting

### Problem: "Cannot connect to host 0.0.0.0:8091"

**Solution**: Ensure miner is started with `--axon.external_ip 127.0.0.1`

### Problem: "Blacklisting unrecognized hotkey"

**Solution**: Both miner and validator need to be registered on the same network/subnet

### Problem: "No prediction available"

**Solution**: Check the miner logs for feature processing errors. The fixed error handling will show detailed tracebacks.

### Problem: API configuration errors

**Solution**: Ensure your `.env` file has correct API credentials

## Testing Workflow

1. **Start Miner**: Run the miner script and verify it starts without errors
2. **Check Miner Logs**: Look for "Miner running..." messages
3. **Start Validator**: Run the validator script in a separate terminal
4. **Monitor Communication**: Watch for successful dendrite calls in validator logs
5. **Verify Predictions**: Check that miners receive challenges and return predictions

## Expected Log Output

### Successful Miner Logs:
```
INFO | Miner starting at block: XXX
INFO | Serving miner axon on network: test with netuid: 1
INFO | Miner running...
DEBUG | Processed synapse for hotkey XXX: {'conversion_happened': 0, 'time_to_conversion_seconds': -1.0}
```

### Successful Validator Logs:
```
INFO | Querying 1 miners with retry resilience for test_pk: XXX
DEBUG | Miner 0: attempting connection to 127.0.0.1:8091
INFO | Dendrite call succeeded after X.XXs total
INFO | Received 1 responses for REAL test_pk: XXX
```

## Network Architecture for Localhost

```
┌─────────────────┐    ┌─────────────────┐
│   Validator     │    │     Miner       │
│                 │    │                 │
│ Queries miners  │───▶│ Listens on      │
│ via metagraph   │    │ 127.0.0.1:8091  │
│                 │◀───│ Returns         │
│ Processes       │    │ predictions     │
│ responses       │    │                 │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
              Same Machine
              127.0.0.1
```

## Advanced Testing

### Test with Multiple Miners

To test with multiple miners on different ports:

```bash
# Miner 1
python -m neurons.miner --axon.external_ip 127.0.0.1 --axon.port 8091 --wallet.name miner1

# Miner 2  
python -m neurons.miner --axon.external_ip 127.0.0.1 --axon.port 8092 --wallet.name miner2

# Validator querying both
python -m neurons.validator --neuron.sample_size 2
```

### Mock API Testing

For testing without external APIs, you can modify the validator to use mock responses (see test files for examples).

## Production vs Localhost Differences

| Aspect | Localhost | Production |
|--------|-----------|------------|
| **Miner IP** | `127.0.0.1` | Public IP or domain |
| **Network** | `test` | `main` or specific subnet |
| **Sample Size** | `1-2` miners | `10+` miners |
| **API Config** | Can use mocks | Requires real APIs |
| **Registration** | Optional on testnet | Required on mainnet | 
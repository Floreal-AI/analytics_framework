#!/bin/bash
# Run validator configured for localhost testing

echo "Starting Validator for Localhost Testing..."
echo "==========================================="

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ensure environment file exists for API configuration
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create .env file with your API credentials:"
    echo ""
    echo "VOICEFORM_API_BASE_URL=your_api_url"
    echo "VOICEFORM_API_KEY=your_api_key"
    echo ""
    echo "You can copy the example file:"
    echo "cp env.example .env"
    echo ""
    echo "Then edit .env with your real API credentials."
    echo ""
    echo "Continuing without .env - validator will fail when trying to fetch test data..."
    sleep 3
fi

# Run the validator with localhost configuration
python -m neurons.validator \
  --netuid 2 \
  --subtensor.network local \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --wallet.name amitn_validator \
  --wallet.hotkey default \
  --neuron.sample_size 2 \
  --logging.debug \
  --neuron.device cpu

echo "Validator stopped." 
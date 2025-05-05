# Bittensor Analytics Framework - Miner Guide

## Overview

This repository contains a framework for participating as a miner in the Bittensor Analytics subnet. Miners in this subnet predict whether AI agents will successfully complete tasks in conversations and how long it will take.

## Quick Links

- [Miner Quickstart Guide](MINER_QUICKSTART.md) - Comprehensive guide to getting started
- [Advanced Custom Miner Guide](CUSTOM_MINER_GUIDE.md) - Detailed guide for implementing custom miners
- [Demo Script](test_miner.py) - Example script demonstrating the miner implementation

## Basic Workflow

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a model**:
   ```bash
   python test_miner.py --train --data_path path/to/train.csv
   ```

3. **Test the model**:
   ```bash
   python test_miner.py
   ```

4. **Run the miner**:
   ```bash
   python -m conversion_subnet.miner.run --subtensor.network <network> --wallet.name <wallet> --wallet.hotkey <hotkey>
   ```

## Miner Architecture

The default miner implementation is a binary classification model that:

1. Processes 40 conversation features
2. Predicts binary classification outcome (conversion_happened: 0 or 1)
3. Predicts the time to conversion (time_to_conversion_seconds)

## Scoring Mechanism

Validators evaluate miners based on:

- **Classification Reward (55%)**: Accuracy of conversion predictions
- **Regression Reward (35%)**: Accuracy of time predictions
- **Diversity Reward (10%)**: Encouraging unique and confident predictions
- **Time Reward**: Response time performance (within 60 seconds)

## Testing

Run the test suite to verify the framework:

```bash
pytest -xvs tests/
```

## Customization

For performance improvements, consider:

1. Building custom model architectures
2. Implementing advanced feature engineering
3. Using ensemble models
4. Optimizing response time and prediction confidence

See the [Advanced Custom Miner Guide](CUSTOM_MINER_GUIDE.md) for implementation details.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
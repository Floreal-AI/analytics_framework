Quote Prediction Subnet
The Quote Prediction Subnet is a Bittensor subnet that incentivizes miners to predict quote delivery time (target: ≤90 seconds) and quote acceptance rate (target: ≥85%) based on customer interaction features. Validators generate synthetic interaction data, and miners respond with predictions, which are scored for accuracy and speed.
Overview

Validators: Generate synthetic customer interaction data (e.g., interaction_duration, sentiment_score) and ground truth targets (quote_delivery_time, quote_acceptance). They query miners and reward accurate predictions.
Miners: Process feature vectors and predict the time to deliver a quote and the likelihood of quote acceptance.
Use Case: Optimizes customer service interactions (e.g., call center quotes) by predicting and improving efficiency and conversion rates.

Installation

Clone the repository:
git clone https://github.com/your-repo/quote-prediction-subnet.git
cd quote-prediction-subnet


Install dependencies:
pip install -r requirements.txt


Ensure Python 3.8–3.11 and Bittensor are installed.


Running the Subnet
Validator
Run a validator to generate data and score miner responses:
python -m ocr_subnet.validator

Miner
Run a miner to predict quote outcomes:
python -m neurons.miner

Configuration

Validator Config: Adjust neuron.timeout and neuron.alpha_prediction in the validator script or config file.
Miner Enhancements: Improve the baseline miner by implementing machine learning models (e.g., using scikit-learn).

Dataset
The subnet uses synthetic data based on the following features:

interaction_duration: Mean 117.13s, std 12.36
question_count: Mean 5.53, std 1.08
sentiment_score: Mean 0.71, std 0.09
... (see validator/generate.py for full list)

Targets:

quote_delivery_time: Continuous (mean ~108s, target ≤90s)
quote_acceptance: Binary (mean 80%, target ≥85%)

Contributing
See contrib/CONTRIBUTING.md for guidelines. Submit issues or PRs to improve miners, validators, or documentation.
License
MIT License. See LICENSE for details.

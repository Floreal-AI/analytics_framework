import time
import bittensor as bt
import numpy as np
from conversion_subnet.protocol import ConversionSynapse
from conversion_subnet.validator.reward import Validator
from conversion_subnet.utils.uids import get_random_uids
from conversion_subnet.validator.generate import generate_conversation
from conversion_subnet.validator.utils import validate_features, log_metrics

async def forward(self):
    """
    The forward function is called by the validator every time step.
    It queries the network with real-time conversation features and scores miner predictions.

    Args:
        self: The validator neuron object containing state (e.g., metagraph, dendrite, config).
    """
    # Select a subset of miners to query (e.g., 10 miners)
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    # TODO: Obtain real-time conversation features from our API
    # For development/testing, use synthetic features from generate_conversation
    features = validate_features(generate_conversation())  # Synthetic data, validated
    # Example real-time retrieval (uncomment and implement):
    # features = await get_realtime_features()  # Custom function to fetch from API/database
    # features = validate_features(features)

    # Create ConversionSynapse with features
    synapse = ConversionSynapse(features=features)

    # Query miners and measure response time
    start_time = time.time()
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=True,
        timeout=60.0  # 60-second timeout for real-time responses
    )
    end_time = time.time()

    # Update response times in synapses
    for response, uid in zip(responses, miner_uids):
        response.response_time = end_time - start_time
        response.miner_uid = uid

    # Log responses for monitoring
    bt.logging.info(f"Received responses: {[r.prediction for r in responses if r.prediction is not None]}")

    # TODO: Obtain ground truth post-conversation (e.g., from external system, human evaluation, or automated rules)
    ground_truth = {
        'session_id': features['session_id'],
        'conversion_happened': 1,  # Placeholder
        'time_to_conversion_seconds': 62.0  # Placeholder
    }

    # Score responses using the Incentive Mechanism
    score_validator = Validator()
    rewards = []
    for response in responses:
        if response.prediction is None or not response.prediction:
            reward = 0.0
        else:
            reward = score_validator.reward(ground_truth, response)
            log_metrics(response, reward, ground_truth)  # Log detailed metrics
        rewards.append(reward)

    # Convert rewards to numpy array for weight updates
    rewards = np.array(rewards, dtype=np.float32)

    # Log scored responses
    bt.logging.info(f"Scored responses: {rewards}")

    # Update miner scores based on rewards
    self.update_scores(rewards, miner_uids)

import bittensor as bt
import torch
from quote_prediction_subnet.validator.generate import generate_interaction
from quote_prediction_subnet.validator.reward import get_rewards
from quote_prediction_subnet.protocol import QuotePredictionSynapse

async def forward(self):
    """
    Validator forward pass. Generates interaction data, queries miners, and computes rewards.
    """
    # Generate synthetic interaction data
    data = generate_interaction()
    features = data['features']
    targets = data['targets']
    
    # Create synapse with features
    synapse = QuotePredictionSynapse(features=features)
    
    # Query miners
    responses = await self.dendrite.query(
        axons=[self.metagraph.axons[uid] for uid in self.miner_uids],
        synapse=synapse,
        deserialize=True,
        timeout=self.config.neuron.timeout
    )
    
    # Compute rewards
    rewards = get_rewards(targets, responses).to(self.device)
    
    # Update weights (example, adjust as per subnet logic)
    self.update_weights(rewards)
    
    # Log results
    bt.logging.info(f"Rewards: {rewards}")
    return rewards
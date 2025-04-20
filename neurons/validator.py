import os
import time
import hashlib
import bittensor as bt

import quote_prediction_subnet

# import base validator class which takes care of most of the boilerplate
from quote_prediction_subnet.base.validator import BaseValidatorNeuron
import bittensor as bt
from quote_prediction_subnet.protocol import QuotePredictionSynapse
from quote_prediction_subnet.validator.generate import generate_interaction
from quote_prediction_subnet.validator.reward import get_rewards

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        bt.logging.info("load_state()")
        self.load_state()

    async def forward(self):
        miner_uids = quote_prediction_subnet.utils.uids.get_random_uids(
            self, k=min(self.config.neuron.sample_size, self.metagraph.n.item())
        )
        data = generate_interaction()
        synapse = QuotePredictionSynapse(features=data['features'])
        
        responses = await self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=synapse,
            deserialize=False,
        )
        
        rewards = get_rewards(data['targets'], responses, self.config)
        bt.logging.info(f"Received responses: {responses}")
        bt.logging.info(f"Scored responses: {rewards}")
        self.update_scores(rewards, miner_uids)

if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
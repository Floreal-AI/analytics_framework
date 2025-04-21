# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 [Your Name/Organization] (modified for AI Agent Task Performance Subnet)

# [Existing license text remains unchanged]

import bittensor as bt
from conversion_subnet.base.validator import BaseValidatorNeuron
from conversion_subnet.validator.forward import forward

class Validator(BaseValidatorNeuron):
    """
    Validator neuron for the AI Agent Task Performance Subnet. Queries miners with conversation features,
    scores their predictions (conversion_happened, time_to_conversion_seconds), and updates network weights.

    Inherits from BaseValidatorNeuron, which handles Bittensor network operations (wallet, subtensor, dendrite, syncing).
    This class delegates the forward method to the forward.py implementation, which generates or obtains features,
    queries miners, and scores responses using the Incentive Mechanism.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

    async def forward(self):
        """
        The forward function is called by the validator every time step. It delegates to the forward.py implementation,
        which:
        - Obtains conversation features (synthetic for development, real-time in production)
        - Queries miners with ConversionSynapse
        - Scores responses using the Incentive Mechanism
        - Updates miner scores
        """
        await forward(self)

if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)

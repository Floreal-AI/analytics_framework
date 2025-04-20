import unittest
import bittensor as bt
from neurons.validator import Validator
from quote_prediction_subnet.protocol import QuotePredictionSynapse
from quote_prediction_subnet.validator.generate import generate_interaction

class TemplateValidatorNeuronTestCase(unittest.TestCase):
    def setUp(self):
        config = Validator.config()
        config.wallet._mock = True
        config.metagraph._mock = True
        config.subtensor._mock = True
        self.neuron = Validator(config)
        self.miner_uids = [0, 1, 2]  # Mock UIDs

    async def test_forward(self):
        data = generate_interaction()
        synapse = QuotePredictionSynapse(features=data['features'])
        responses = await self.neuron.dendrite.query(
            axons=[self.neuron.metagraph.axons[uid] for uid in self.miner_uids],
            synapse=synapse,
            deserialize=False,
        )
        self.assertEqual(len(responses), len(self.miner_uids))
        for resp in responses:
            self.assertIsNotNone(resp.quote_delivery_time_pred)
            self.assertIsNotNone(resp.quote_acceptance_pred)

if __name__ == '__main__':
    unittest.main()
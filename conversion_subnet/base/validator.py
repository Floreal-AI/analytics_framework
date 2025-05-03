import copy
import torch
import asyncio
import threading
import bittensor as bt
import numpy as np
from typing import List, Union
from traceback import print_exception

from conversion_subnet.base.neuron import BaseNeuron


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        # Convert numpy array to torch tensor before using zeros_like
        try:
            S_tensor = torch.tensor(self.metagraph.S, dtype=torch.float32)
            self.scores = torch.zeros_like(S_tensor)
        except Exception as e:
            bt.logging.warning(f"Error creating scores tensor: {e}. Using direct initialization.")
            # Fallback to direct initialization if conversion fails
            self.scores = torch.zeros(len(self.metagraph.S), dtype=torch.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(
                print_exception(type(err), err, err.__traceback__)
            )

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nn.functional.normalize(self.scores, p=1, dim=0)
        bt.logging.trace("raw_weights", raw_weights)
        bt.logging.trace("top10 values", raw_weights.sort()[0])
        bt.logging.trace("top10 uids", raw_weights.sort()[1])

        # Process the raw weights to final_weights via subtensor limitations.
        try:
            # Convert uids to torch tensor if it's not already
            if isinstance(self.metagraph.uids, np.ndarray):
                uids_tensor = torch.tensor(self.metagraph.uids, dtype=torch.long)
            else:
                uids_tensor = self.metagraph.uids.to("cpu")
                
            # Convert weights to CPU
            weights_cpu = raw_weights.detach().cpu()
            
            # Handle potential conversion issues with bt.utils.weight_utils
            try:
                (
                    processed_weight_uids,
                    processed_weights,
                ) = bt.utils.weight_utils.process_weights_for_netuid(
                    uids=uids_tensor,
                    weights=weights_cpu,
                    netuid=self.config.netuid,
                    subtensor=self.subtensor,
                    metagraph=self.metagraph,
                )
            except AttributeError as e:
                if "astype" in str(e):
                    # Handle the specific "astype" error by converting to numpy first
                    bt.logging.warning(f"Handling 'astype' error: {e}")
                    weights_numpy = weights_cpu.numpy()
                    uids_numpy = uids_tensor.numpy() if hasattr(uids_tensor, "numpy") else np.array(uids_tensor)
                    (
                        processed_weight_uids,
                        processed_weights,
                    ) = bt.utils.weight_utils.process_weights_for_netuid(
                        uids=uids_numpy,
                        weights=weights_numpy,
                        netuid=self.config.netuid,
                        subtensor=self.subtensor,
                        metagraph=self.metagraph,
                    )
                else:
                    # Rethrow other attribute errors
                    raise
        except Exception as e:
            bt.logging.error(f"Error processing weights: {e}. Using direct weights.")
            # Fallback to direct weights
            try:
                # Directly create numpy arrays for the weights
                uids_list = list(range(min(len(self.scores), len(self.metagraph.uids))))
                weights_norm = raw_weights[uids_list].detach().cpu().numpy()
                
                # Ensure weights sum to 1
                if np.sum(weights_norm) > 0:
                    weights_norm = weights_norm / np.sum(weights_norm)
                
                processed_weight_uids = uids_list
                processed_weights = [float(w) for w in weights_norm]
            except Exception as inner_e:
                bt.logging.error(f"Error in direct weights fallback: {inner_e}. Using zeros.")
                # If all else fails, use zeros
                processed_weight_uids = list(range(min(len(self.scores), len(self.metagraph.uids))))
                processed_weights = [0.0] * len(processed_weight_uids)
            
        bt.logging.trace("processed_weights", processed_weights)
        bt.logging.trace("processed_weight_uids", processed_weight_uids)

        # Set the weights on chain via our subtensor connection.
        self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=processed_weight_uids,
            weights=processed_weights,
            wait_for_finalization=False,
            version_key=self.spec_version,
        )

        bt.logging.info(f"Set weights: {processed_weights}")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        try:
            # Copies state of metagraph before syncing.
            previous_metagraph = copy.deepcopy(self.metagraph)

            # Sync the metagraph.
            self.metagraph.sync(subtensor=self.subtensor)

            # Check if the metagraph axon info has changed.
            if previous_metagraph.axons == self.metagraph.axons:
                return

            bt.logging.info(
                "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
            )
            
            # Zero out all hotkeys that have been replaced.
            try:
                for uid, hotkey in enumerate(self.hotkeys):
                    if uid < len(self.metagraph.hotkeys) and hotkey != self.metagraph.hotkeys[uid]:
                        self.scores[uid] = 0  # hotkey has been replaced
            except Exception as e:
                bt.logging.warning(f"Error updating hotkey scores: {e}")
                
            # Check to see if the metagraph has changed size.
            # If so, we need to add new hotkeys and moving averages.
            if len(self.hotkeys) < len(self.metagraph.hotkeys):
                bt.logging.info(f"Metagraph size increased from {len(self.hotkeys)} to {len(self.metagraph.hotkeys)}")
                # Update the size of the moving average scores.
                try:
                    # When creating a new scores tensor, make sure it's on the right device
                    device = getattr(self, 'device', 'cpu')
                    
                    # Create new tensor of appropriate size
                    new_scores = torch.zeros(len(self.metagraph.hotkeys), dtype=torch.float32)
                    
                    # Copy over existing scores
                    min_len = min(len(self.hotkeys), len(self.scores), len(new_scores))
                    new_scores[:min_len] = self.scores[:min_len]
                    
                    # Replace the old scores tensor
                    self.scores = new_scores
                except Exception as e:
                    bt.logging.warning(f"Error resizing scores tensor: {e}. Reinitializing scores.")
                    # If there's an error, reinitialize the scores tensor
                    self.scores = torch.zeros(len(self.metagraph.hotkeys), dtype=torch.float32)
            elif len(self.hotkeys) > len(self.metagraph.hotkeys):
                bt.logging.info(f"Metagraph size decreased from {len(self.hotkeys)} to {len(self.metagraph.hotkeys)}")
                # Metagraph got smaller, resize the scores
                try:
                    new_scores = torch.zeros(len(self.metagraph.hotkeys), dtype=torch.float32)
                    min_len = min(len(new_scores), len(self.scores))
                    new_scores[:min_len] = self.scores[:min_len]
                    self.scores = new_scores
                except Exception as e:
                    bt.logging.warning(f"Error downsizing scores tensor: {e}. Reinitializing scores.")
                    self.scores = torch.zeros(len(self.metagraph.hotkeys), dtype=torch.float32)
            
            # Update the hotkeys.
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
            
        except Exception as e:
            bt.logging.error(f"Error in resync_metagraph: {e}")
            bt.logging.debug(traceback.format_exc())

    def update_scores(self, rewards: Union[torch.FloatTensor, np.ndarray, List[float]], uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Convert rewards to torch tensor if needed
        if isinstance(rewards, np.ndarray):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        elif isinstance(rewards, list):
            rewards = torch.tensor(rewards, dtype=torch.float32)
            
        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        try:
            # Ensure uids is a tensor on the correct device
            uids_tensor = torch.tensor(uids, dtype=torch.long)
            
            # Make sure rewards is on the CPU for compatibility
            rewards_cpu = rewards.to("cpu") if hasattr(rewards, "to") else rewards
            
            # Compute forward pass rewards, assumes uids are mutually exclusive.
            # shape: [ metagraph.n ]
            scattered_rewards = torch.zeros_like(self.scores)
            for i, uid in enumerate(uids):
                if uid < len(scattered_rewards):
                    scattered_rewards[uid] = rewards_cpu[i]
                    
            bt.logging.debug(f"Scattered rewards: {scattered_rewards}")

            # Update scores with rewards produced by this step.
            # shape: [ metagraph.n ]
            alpha: float = self.config.neuron.moving_average_alpha
            self.scores = alpha * scattered_rewards + (1 - alpha) * self.scores
            bt.logging.debug(f"Updated moving avg scores: {self.scores}")
        except Exception as e:
            bt.logging.warning(f"Error updating scores: {e}. Using simpler update method.")
            # Fallback to direct indexing
            try:
                for i, uid in enumerate(uids):
                    if uid < len(self.scores):
                        # Apply EMA directly
                        alpha: float = self.config.neuron.moving_average_alpha
                        reward_value = float(rewards[i]) if i < len(rewards) else 0.0
                        self.scores[uid] = alpha * reward_value + (1 - alpha) * float(self.scores[uid])
            except Exception as e2:
                bt.logging.error(f"Error in fallback score update: {e2}. Scores not updated.")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        state = torch.load(self.config.neuron.full_path + "/state.pt")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]

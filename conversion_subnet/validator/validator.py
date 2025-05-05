# conversion_subnet/validator/validator.py
"""
Validator neuron for the Conversion‑Analytics subnet.

Responsibilities
----------------
1.  Receives `ConversionSynapse` responses from miners.
2.  Computes per‑miner rewards via conversion_subnet.reward helpers.
3.  Maintains an exponentially‑weighted moving average (EMA) of scores
    so short‑lived spikes do not dominate hot‑keys.
4.  Exposes blacklist & priority logic as required by Bittensor SDK.
"""
from __future__ import annotations

import time
from typing import Dict, List

import bittensor as bt

from conversion_subnet.reward import (
    classification,
    regression,
    diversity,
    latency,
    prediction_score,
    total_score,
    ema,
)
from conversion_subnet.types import FeatureMap, Prediction             # noqa: F401  (linted but unused directly)
from conversion_subnet.constants import TIMEOUT_SEC

# ────────────────────────────────────────────────────────────────────
#   Data classes
# ────────────────────────────────────────────────────────────────────
class MinerStats:
    """Mutable stats we cache per UID."""
    __slots__ = ("ema_score", "last_seen")

    def __init__(self) -> None:
        self.ema_score: float = 0.0
        self.last_seen: float = time.time()


# ────────────────────────────────────────────────────────────────────
#   Validator implementation
# ────────────────────────────────────────────────────────────────────
class ConversionValidatorNeuron(bt.SynapseValidator):
    """
    Bittensor validator that evaluates miner predictions for
    conversation‑conversion events.
    """

    def __init__(self, config: bt.Config) -> None:
        super().__init__(config)
        self.stats: Dict[int, MinerStats] = {}
        bt.logging.info("🧮  ConversionValidatorNeuron initialised")

    # -------------------------------------------------------------- #
    #   ─── Core bittensor API hooks ──────────────────────────────── #
    # -------------------------------------------------------------- #
    def epoch(self) -> None:  # called once per epoch by the framework
        """Purge very old miner stats to bound memory usage."""
        threshold = time.time() - (2 * 60 * 60)  # 2 h
        to_remove: List[int] = [
            uid for uid, s in self.stats.items() if s.last_seen < threshold
        ]
        for uid in to_remove:
            self.stats.pop(uid, None)
        bt.logging.info(f"Validator purged {len(to_remove)} inactive miners")

    def forward(self, synapse: bt.Synapse) -> bt.Synapse:  # type: ignore[override]
        """
        Evaluate miner answer contained in `synapse.resp`.
        Populate `synapse.score` so the SDK can update stake.
        """
        uid = synapse.uid
        resp = synapse.resp  # expected .prediction, .confidence, .response_time
        target = synapse.labels

        # --- granular metrics ------------------------------------------------
        cls_r = classification(resp.prediction, target)
        reg_r = regression(resp.prediction, target)
        div_r = diversity(resp.confidence)
        pred_r = prediction_score(cls_r, reg_r, div_r)
        time_r = latency(resp.response_time)

        score = total_score(pred_r, time_r)

        # --- stats -----------------------------------------------------------
        stat = self.stats.setdefault(uid, MinerStats())
        stat.ema_score = ema(score, stat.ema_score)
        stat.last_seen = time.time()

        synapse.score = score
        synapse.debug = {
            "cls": cls_r,
            "reg": reg_r,
            "div": div_r,
            "lat": time_r,
            "ema": stat.ema_score,
        }
        return synapse

    # -------------------------------------------------------------- #
    #   ─── Optional SDK hooks ────────────────────────────────────── #
    # -------------------------------------------------------------- #
    def blacklist(self, synapse: bt.Synapse) -> bool:  # type: ignore[override]
        """
        Blacklist miners that consistently exceed TIMEOUT_SEC
        or whose EMA drops below 0.05.
        """
        uid = synapse.uid
        stat = self.stats.get(uid)
        if synapse.resp.response_time >= TIMEOUT_SEC:
            return True
        if stat and stat.ema_score < 0.05:
            return True
        return False

    def priority(self, synapse: bt.Synapse) -> float:  # type: ignore[override]
        """
        Prioritise miners with higher EMA so they receive
        more queries in future epochs.
        """
        stat = self.stats.get(synapse.uid)
        return stat.ema_score if stat else 0.1

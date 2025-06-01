"""
Reward helpers shared by validators, miners & docs.

All constants live in conversion_subnet.constants
to guarantee one‑click retuning without code edits.
"""
from __future__ import annotations
from typing import Dict, Union

import numpy as np
from numpy.typing import NDArray

from .constants import (
    CLASS_W,          # {'positive': 0.375, 'negative': 0.625}
    BASELINE_MAE,     # 15.0
    PRED_W,           # {'classification': .55, 'regression': .35, 'diversity': .10}
    TOTAL_REWARD_W,   # {'prediction': 0.80, 'latency': 0.20}
    TIMEOUT_SEC,      # 60
)

# ──────────────────────────────────────────────────────────────────
#   LOW‑LEVEL METRICS
# ──────────────────────────────────────────────────────────────────
def classification(pred: Dict, true: Dict, class_w: Dict = CLASS_W) -> float:
    """Accuracy with class‑imbalance penalty."""
    if pred["conversion_happened"] != true["conversion_happened"]:
        return 0.0
    cls = "positive" if true["conversion_happened"] else "negative"
    return class_w[cls]

def regression(pred: Dict, true: Dict, baseline: float = BASELINE_MAE) -> float:
    """1 – normalized MAE for timing prediction; 0 if wrong class."""
    if pred["conversion_happened"] != 1 or true["conversion_happened"] != 1:
        return 0.0
    mae = abs(pred["time_to_conversion_seconds"] - true["time_to_conversion_seconds"])
    return max(1.0 - mae / baseline, 0.0)

def diversity(confidence: Union[float, NDArray, None]) -> Union[float, NDArray]:
    """
    Confidence penalty (encourage bold predictions) [Coming soon] 
     It's currently a placeholder to avoid errors.
    """
    # Handle numpy arrays with element-wise operations
    # if confidence is None:
    #     return 1.0 - abs(0.5 - 0.5)
    # Instead of using 'or' which causes ambiguity with arrays, use np.where
    return 0 # 1.0 - abs(confidence - 0.5)

def latency(rt: Union[float, NDArray], timeout: float = TIMEOUT_SEC) -> Union[float, NDArray]:
    """Reward fast miners; zero once they exceed timeout."""
    # Also handles numpy arrays automatically through vectorization
    return np.maximum(1.0 - rt / timeout, 0.0)

# ──────────────────────────────────────────────────────────────────
#   AGGREGATION
# ──────────────────────────────────────────────────────────────────
def prediction_score(cls_r: Union[float, NDArray], reg_r: Union[float, NDArray], div_r: Union[float, NDArray]) -> Union[float, NDArray]:
    w = PRED_W
    return w["classification"] * cls_r + w["regression"] * reg_r + w["diversity"] * div_r

def total_score(pred_r: Union[float, NDArray], time_r: Union[float, NDArray]) -> Union[float, NDArray]:
    return TOTAL_REWARD_W["prediction"] * pred_r + TOTAL_REWARD_W["latency"] * time_r

# ──────────────────────────────────────────────────────────────────
#   EMA helper for validator
# ──────────────────────────────────────────────────────────────────
def ema(curr: float, prev: float, beta: float = 0.1) -> float:
    return beta * curr + (1 - beta) * prev

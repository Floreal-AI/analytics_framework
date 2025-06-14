# conversion_subnet/constants.py
"""
Authoritative constants for reward computation and runtime tuning.

‣ Every default lives in ALL_CAPS at module import‑time.
‣ A YAML / env‑var override mechanism lets ops adjust values in prod
  while your source code – and unit‑tests – stay frozen.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

# --------------------------------------------------------------------------- #
# 1.  Default constants (💡 feel free to tweak – these are version‑controlled)
# --------------------------------------------------------------------------- #
# Classification balance weights (positive samples are fewer than negative)
CLASS_W: Dict[str, float] = {"positive": 0.375, "negative": 0.625}

# Regression metrics
BASELINE_MAE: float = 15.0                 # seconds
MIN_CONVERSION_TIME: float = 30.0          # minimum time to conversion in seconds
TIME_SCALE_FACTOR: float = 0.7             # scale factor for time to conversion calculation

# Reward component weights
PRED_W: Dict[str, float] = {               # prediction sub‑weights
    "classification": 0.55,                # weight for classification accuracy
    "regression": 0.45,                    # weight for regression accuracy
    "diversity": 0.0,                     # weight for prediction diversity
}
TOTAL_REWARD_W: Dict[str, float] = {       # total reward weights
    "prediction": 0.80,                    # weight for prediction quality
    "latency": 0.20,                       # weight for response time
}

# Timeouts and limits
TIMEOUT_SEC: int = 120                     # Increased timeout for slow networks (zero reward once response_time ≥ 120 s)
EMA_BETA: float = 0.1                      # EMA smoothing factor (higher = more weight to new data)

# Model parameters
MLP_LAYER_SIZES: list[int] = [64, 32, 1]   # Hidden layer sizes for MLP model

# Feature validation constants
REQUIRED_FEATURES: list[str] = [
    'conversation_duration_seconds', 
    'has_target_entity',
    'entities_collected_count', 
    'message_ratio',
    'session_id'
]

# Ground truth generation parameters
ENTITY_THRESHOLD: int = 4                  # Minimum entities needed for conversion
MESSAGE_RATIO_THRESHOLD: float = 1.2       # Minimum agent/user message ratio for conversion
MIN_CONVERSATION_DURATION: float = 90.0    # Minimum conversation duration for conversion

# Validator parameters
SAMPLE_SIZE: int = 10                      # Number of miners to sample in each forward pass
HISTORY_UPDATE_INTERVAL: int = 100         # Update class weights every N conversations

# --------------------------------------------------------------------------- #
# 2.  Runtime override helper (optional)                                      #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf‑8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as err:
        raise RuntimeError(f"Invalid YAML in {path}: {err}") from err


def _apply_overrides() -> None:
    """
    Order of precedence (highest → lowest):

    1. Environment variables, e.g.  CLASS_W__positive=0.4
    2. YAML file pointed to by env var  ANALYTICS_CONSTANTS_PATH
       (default: ~/.analytics_constants.yml)
    3. Defaults defined above
    """
    # 2️⃣ YAML
    yaml_path = Path(
        os.getenv("ANALYTICS_CONSTANTS_PATH", Path.home() / ".analytics_constants.yml")
    )
    data = _load_yaml(yaml_path)

    # 3️⃣ merge YAML → globals
    for key, val in data.items():
        key_upper = key.upper()
        if key_upper in globals():
            globals()[key_upper] = val

    # 1️⃣ individual env‑vars (supports nested dict keys using __)
    for env, val in os.environ.items():
        if "__" in env:  # e.g. CLASS_W__positive
            top, sub = env.split("__", 1)
            if top in globals() and isinstance(globals()[top], dict):
                try:
                    globals()[top][sub] = type(globals()[top][sub])(val)
                except (KeyError, ValueError):
                    continue
        elif env in globals():
            # cast to the existing type
            try:
                globals()[env] = type(globals()[env])(val)
            except ValueError:
                continue


_apply_overrides()

# --------------------------------------------------------------------------- #
# 3.  Debug helper                                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # quick smoke‑test
    import pprint

    pprint.pp(
        {
            "CLASS_W": CLASS_W,
            "BASELINE_MAE": BASELINE_MAE,
            "PRED_W": PRED_W,
            "TOTAL_REWARD_W": TOTAL_REWARD_W,
            "TIMEOUT_SEC": TIMEOUT_SEC,
            "EMA_BETA": EMA_BETA,
        }
    )

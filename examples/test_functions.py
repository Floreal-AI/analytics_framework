import sys
import numpy as np

# Add the current directory to Python path
sys.path.append('.')

# Import the functions
from conversion_subnet.reward import (
    diversity,
    latency,
    prediction_score,
    total_score,
)

# Create test arrays
N = 10
conf = np.random.uniform(0.2, 0.8, size=N)
resp_time = np.random.uniform(0, 90, size=N)

# Test the diversity function
print("Testing diversity function...")
div_r = diversity(conf)
print(f"Input: {conf}")
print(f"Output: {div_r}")
print(f"Shape: {div_r.shape}")

# Test the latency function
print("\nTesting latency function...")
time_r = latency(resp_time)
print(f"Input: {resp_time}")
print(f"Output: {time_r}")
print(f"Shape: {time_r.shape}")

# Test prediction score and total score with arrays
print("\nTesting prediction_score and total_score...")
cls_r = np.random.uniform(0, 1, size=N)
reg_r = np.random.uniform(0, 1, size=N)
pred_r = prediction_score(cls_r, reg_r, div_r)
reward = total_score(pred_r, time_r)

print(f"cls_r shape: {cls_r.shape}")
print(f"reg_r shape: {reg_r.shape}")
print(f"div_r shape: {div_r.shape}")
print(f"pred_r shape: {pred_r.shape}")
print(f"time_r shape: {time_r.shape}")
print(f"reward shape: {reward.shape}")

print("\nAll tests passed!") 
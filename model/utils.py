import numpy as np
import torch

def smoothed_scaled_score(score, pivot=-0.2, alpha=0.1):
    centered = score - pivot
    return torch.tanh(alpha * centered)

def sigmoid_centered(score, pivot=-2, scale=1.5):
    return 2 / (1 + np.exp(-scale * (score - pivot))) - 1
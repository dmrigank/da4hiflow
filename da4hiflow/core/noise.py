"""Noise utilities."""

import numpy as np


def add_gaussian_noise(
    y: np.ndarray, sigma: float, rng: np.random.RandomState
) -> np.ndarray:
    """Return y plus Gaussian noise with standard deviation sigma."""
    return y + rng.randn(*y.shape) * sigma

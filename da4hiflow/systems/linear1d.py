"""Toy linear 1D system with stable dynamics."""

import numpy as np

# use two-level relative import so the module works in-place or when installed
from ..core.system_base import SystemBase


class Linear1DSystem(SystemBase):
    """Simple linear dynamics x_{k+1} = A x_k with stable random A."""

    def __init__(self, n: int, seed: int = 0):
        """Create a reproducible stable linear system.

        Args:
            n: dimension of the state vector.
            seed: RNG seed used to generate matrix A and initial state.
        """
        self.n = n
        self.rng = np.random.RandomState(seed)
        # build random matrix and scale to have spectral radius < 1
        M = self.rng.randn(n, n)
        norm = np.linalg.norm(M, 2)
        if norm == 0:
            # degenerate case
            self.A = np.eye(n) * 0.9
        else:
            self.A = M / norm * 0.9

    def get_initial_state(self) -> np.ndarray:
        return self.rng.randn(self.n)

    def step(self, state: np.ndarray) -> np.ndarray:
        return self.A.dot(state)

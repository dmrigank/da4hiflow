"""Base class for data assimilation methods."""

from abc import ABC, abstractmethod
import numpy as np


class AssimilatorBase(ABC):
    """Abstract base class for data assimilation methods."""

    @abstractmethod
    def assimilate(self, state: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Assimilate an observation into the state.

        Args:
            state: Prior state vector.
            observation: Observation vector.

        Returns:
            Posterior state vector.
        """
        pass

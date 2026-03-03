"""Base class for dynamical systems."""

from abc import ABC, abstractmethod
import numpy as np


class SystemBase(ABC):
    """Abstract base class for dynamical systems."""

    @abstractmethod
    def step(self, state: np.ndarray) -> np.ndarray:
        """Advance the system by one time step.

        Args:
            state: Current state vector.

        Returns:
            Next state vector.
        """
        pass

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Get the initial state of the system.

        Returns:
            Initial state vector.
        """
        pass

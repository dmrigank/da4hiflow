"""Main runner for data assimilation experiments."""

import numpy as np
from typing import List
from .system_base import SystemBase
from .assimilator_base import AssimilatorBase


class DummySystem(SystemBase):
    """Dummy system for testing: linear advection x_new = 1.1 * x."""

    def __init__(self, state_size: int = 5, seed: int = 42):
        """Initialize dummy system.

        Args:
            state_size: Dimension of state vector.
            seed: Random seed for reproducibility.
        """
        self.state_size = state_size
        self.rng = np.random.RandomState(seed)

    def get_initial_state(self) -> np.ndarray:
        """Get initial state."""
        return self.rng.randn(self.state_size) * 0.1

    def step(self, state: np.ndarray) -> np.ndarray:
        """Step the system: simple linear scaling."""
        return 1.1 * state


class DummyAssimilator(AssimilatorBase):
    """Dummy assimilator for testing: simple weight average."""

    def __init__(self, weight: float = 0.5):
        """Initialize dummy assimilator.

        Args:
            weight: Weight for blending state and observation.
        """
        self.weight = weight

    def assimilate(self, state: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Blend state and observation."""
        return self.weight * state + (1.0 - self.weight) * observation


class Runner:
    """Simple runner for DA experiments."""

    def __init__(
        self, system: SystemBase, assimilator: AssimilatorBase, num_steps: int = 3
    ):
        """Initialize runner.

        Args:
            system: Dynamical system.
            assimilator: Data assimilation method.
            num_steps: Number of steps to run.
        """
        self.system = system
        self.assimilator = assimilator
        self.num_steps = num_steps
        self.trajectory: List[np.ndarray] = []

    def run(self) -> List[np.ndarray]:
        """Run the DA loop for num_steps iterations.

        Returns:
            List of state vectors at each step.
        """
        state = self.system.get_initial_state()
        self.trajectory = [state.copy()]

        for _ in range(self.num_steps):
            # Advance system
            state = self.system.step(state)

            # Dummy observation: state + small noise
            observation = state + np.random.randn(*state.shape) * 0.01

            # Assimilate
            state = self.assimilator.assimilate(state, observation)

            self.trajectory.append(state.copy())

        return self.trajectory

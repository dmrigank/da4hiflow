"""Main runner for data assimilation experiments."""

import numpy as np
from typing import List, Union
from pathlib import Path
from .system_base import SystemBase
from .assimilator_base import AssimilatorBase

import json
import csv
import datetime

# new imports used in run_experiment
from da4hiflow.systems.linear1d import Linear1DSystem
from da4hiflow.core.obs import ObservationSpec, observe
from da4hiflow.core.noise import add_gaussian_noise
from da4hiflow.assimilators.direct_insertion import DirectInsertionAssimilator


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


def run_experiment(
    config: dict,
    output_root: Union[Path, str] = "runs",
    run_name: Union[str, None] = None,
) -> Path:
    """Execute a simple DA experiment and write results to disk.

    The configuration dictionary should contain the keys:
    ``seed``, ``n``, ``steps``, ``dt``, ``sensor_idx``, ``noise_sigma``.

    Returns the path to the directory where outputs were written.
    """
    output_root = Path(output_root)
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # build objects
    spec = ObservationSpec(
        sensor_idx=config["sensor_idx"],
        noise_sigma=config["noise_sigma"],
    )
    system = Linear1DSystem(n=config["n"], seed=config["seed"])
    assimilator = DirectInsertionAssimilator()
    steps = config["steps"]
    dt = config.get("dt", 1.0)
    n = config["n"]
    sensor_idx = config["sensor_idx"]

    truth = np.zeros((steps + 1, n))
    forecast = np.zeros((steps + 1, n))
    analysis = np.zeros((steps + 1, n))
    observations = np.zeros((steps, len(sensor_idx)))

    truth[0] = system.get_initial_state()
    analysis[0] = truth[0].copy()
    forecast[0] = analysis[0].copy()

    rng = np.random.RandomState(config.get("seed", 0))
    metrics = []

    for k in range(steps):
        truth[k + 1] = system.step(truth[k])
        forecast[k + 1] = system.step(analysis[k])

        y_true = observe(truth[k + 1], spec)
        y = add_gaussian_noise(y_true, spec.noise_sigma, rng)
        observations[k] = y

        innov = y - forecast[k + 1][np.array(sensor_idx)]
        analysis[k + 1] = assimilator.analysis_step(forecast[k + 1], y, spec)

        rmse_analysis = np.sqrt(np.mean((analysis[k + 1] - truth[k + 1]) ** 2))
        rmse_innov = np.sqrt(np.mean(innov**2))
        time = dt * (k + 1)
        metrics.append((k + 1, time, rmse_analysis, rmse_innov))

    # write metrics
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "rmse_analysis", "rmse_innov"])
        writer.writerows(metrics)

    # save snapshots
    np.savez(
        output_dir / "snapshots.npz",
        truth=truth,
        forecast=forecast,
        analysis=analysis,
        observations=observations,
    )

    return output_dir

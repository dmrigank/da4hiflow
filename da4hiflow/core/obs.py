"""Observation specification and operator utilities.

Provides backward-compatible simple `ObservationSpec` and richer
`ObservationOperator` classes used by ensemble assimilators.
"""

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np


@dataclass
class ObservationSpec:
    """Backward-compatible simple spec for selecting indices.

    sensor_idx: list of indices into a flat state vector
    noise_sigma: standard deviation of additive noise
    """
    sensor_idx: List[int]
    noise_sigma: float


class ObservationOperator:
    """Base class for observation operators.

    Subclasses must implement `apply(state_flat) -> y_pred` and
    `measurement_noise_cov()` which returns either a scalar sigma or
    a diagonal covariance array.
    """

    def apply(self, state_flat: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def measurement_noise_cov(self):
        raise NotImplementedError()


class PointProbeObservation(ObservationOperator):
    """Observe a single primitive variable at a set of cell indices.

    variable: one of 'rho', 'u', 'p'
    sensor_idx: sequence of cell indices
    noise_sigma: standard deviation
    """

    def __init__(self, variable: str, sensor_idx: Sequence[int], noise_sigma: float):
        self.variable = variable
        self.sensor_idx = np.array(list(sensor_idx), dtype=int)
        self.noise_sigma = float(noise_sigma)

    def apply(self, state_flat: np.ndarray) -> np.ndarray:
        # expect state_flat represents (nx,3) conservative flattened
        U = state_flat.reshape(-1, 3)
        rho = U[:, 0]
        u = U[:, 1] / rho
        E = U[:, 2]
        p = (1.4 - 1.0) * (E - 0.5 * rho * u ** 2)  # gamma=1.4 assumed for observation
        if self.variable == "rho":
            arr = rho
        elif self.variable == "u":
            arr = u
        elif self.variable == "p":
            arr = p
        else:
            raise ValueError("Unknown variable for PointProbeObservation")
        return arr[self.sensor_idx]

    def measurement_noise_cov(self):
        return np.eye(len(self.sensor_idx)) * (self.noise_sigma ** 2)


class SchlierenObservation(ObservationOperator):
    """Compute a Schlieren-like measurement: |grad(rho)| or |grad(log rho)|.

    sensor_idx: indices where gradient magnitude is sampled (cell centers)
    use_logrho: if True use grad(log rho)
    """

    def __init__(self, sensor_idx: Sequence[int], noise_sigma: float, use_logrho: bool = False):
        self.sensor_idx = np.array(list(sensor_idx), dtype=int)
        self.noise_sigma = float(noise_sigma)
        self.use_logrho = bool(use_logrho)

    def apply(self, state_flat: np.ndarray) -> np.ndarray:
        U = state_flat.reshape(-1, 3)
        rho = U[:, 0]
        if self.use_logrho:
            val = np.log(np.maximum(rho, 1e-12))
        else:
            val = rho
        # central differences
        grad = np.zeros_like(val)
        grad[1:-1] = 0.5 * (val[2:] - val[:-2])
        grad[0] = val[1] - val[0]
        grad[-1] = val[-1] - val[-2]
        out = np.abs(grad)
        return out[self.sensor_idx]

    def measurement_noise_cov(self):
        return np.eye(len(self.sensor_idx)) * (self.noise_sigma ** 2)


def observe(x: np.ndarray, spec: ObservationSpec) -> np.ndarray:
    """Return observations of selected state components (back-compat)."""
    return x[np.array(spec.sensor_idx)]


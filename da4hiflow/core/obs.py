"""Observation specification and helper utilities."""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ObservationSpec:
    sensor_idx: List[int]
    noise_sigma: float


def observe(x: np.ndarray, spec: ObservationSpec) -> np.ndarray:
    """Return observations of selected state components."""
    return x[np.array(spec.sensor_idx)]

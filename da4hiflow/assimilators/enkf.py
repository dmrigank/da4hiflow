"""Stochastic Ensemble Kalman Filter (EnKF) implementation."""

import numpy as np
from typing import Optional


class EnKF:
    """Stochastic EnKF with perturbed observations and optional inflation.

    Methods:
        analysis(ensemble, y, sensor_idx, R_sigma, rng, inflation=1.0)
    """

    def __init__(self, inflation: float = 1.0):
        self.inflation = float(inflation)

    def analysis(
        self,
        ensemble: np.ndarray,
        y: np.ndarray,
        sensor_idx: np.ndarray,
        R_sigma: float,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Perform stochastic EnKF analysis update.

        Args:
            ensemble: array shape (Ne, n)
            y: observation vector shape (m,)
            sensor_idx: array-like indices of observed variables (length m)
            R_sigma: observation noise std
            rng: RandomState for perturbed observations

        Returns:
            analysis_ensemble: array shape (Ne, n)
        """
        if rng is None:
            rng = np.random.RandomState(0)

        Ne, n = ensemble.shape
        m = len(sensor_idx)

        # Forecast mean and anomalies
        xf_mean = ensemble.mean(axis=0)
        Af = ensemble - xf_mean
        # Apply multiplicative inflation
        Af = Af * self.inflation

        # Observation operator H: picks indices
        H = np.zeros((m, n), dtype=float)
        for i, idx in enumerate(sensor_idx):
            H[i, idx] = 1.0

        # Perturbed observations
        Y = np.tile(y.reshape(m, 1), (1, Ne)) + rng.randn(m, Ne) * R_sigma

        # Project ensemble to observation space
        HX = H @ ensemble.T  # (m, Ne)

        # Compute sample covariance between state and observations
        PfHT = (Af.T @ HX.T) / (Ne - 1)  # (n, m)
        HPfHT = (HX @ HX.T) / (Ne - 1)  # (m, m)

        R = np.eye(m) * (R_sigma**2)
        S = HPfHT + R

        # Kalman gain
        K = PfHT @ np.linalg.inv(S)

        # Update each ensemble member using its perturbed obs
        Xa = ensemble.copy()
        for j in range(Ne):
            innov = Y[:, j] - HX[:, j]
            Xa[j] = ensemble[j] + K @ innov

        return Xa

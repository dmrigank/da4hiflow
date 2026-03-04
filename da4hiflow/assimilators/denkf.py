"""Deterministic EnKF (DEnKF) implementation following Sakov & Oke-style update."""

import numpy as np
from typing import Optional


class DEnKF:
    """Deterministic EnKF (simplified)."""

    def __init__(self, inflation: float = 1.0):
        self.inflation = float(inflation)

    def analysis(
        self,
        ensemble: np.ndarray,
        y: np.ndarray,
        sensor_idx: object,
        R_sigma: float,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Perform deterministic EnKF (DEnKF) analysis update.

        This implementation uses the square-root style deterministic update where
        the update to anomalies uses a transform matrix derived from Kalman gain.

        Args:
            ensemble: array shape (Ne, n)
            y: observation vector shape (m,)
            sensor_idx: array-like indices of observed variables (length m)
            R_sigma: observation noise std
            rng: unused but kept for interface compatibility

        Returns:
            analysis_ensemble: array shape (Ne, n)
        """
        Ne, n = ensemble.shape
        m = len(sensor_idx)

        xf_mean = ensemble.mean(axis=0)
        Af = ensemble - xf_mean
        Af = Af * self.inflation

        # build HX either via operator or index selection
        if hasattr(sensor_idx, "apply"):
            HX_cols = [sensor_idx.apply(ensemble[i]) for i in range(Ne)]
            HX = np.column_stack(HX_cols)
            m = HX.shape[0]
        else:
            H = np.zeros((m, n), dtype=float)
            for i, idx in enumerate(sensor_idx):
                H[i, idx] = 1.0
            HX = H @ ensemble.T  # (m, Ne)

        PfHT = (Af.T @ HX.T) / (Ne - 1)  # (n, m)
        HPfHT = (HX @ HX.T) / (Ne - 1)  # (m, m)
        R = np.eye(HPfHT.shape[0]) * (R_sigma ** 2)
        S = HPfHT + R

        K = PfHT @ np.linalg.inv(S)

        # Analysis mean update
        # innovation uses operator or index selection
        if hasattr(sensor_idx, "apply"):
            y_pred = sensor_idx.apply(xf_mean)
            innov = y - y_pred
        else:
            innov = y - (H @ xf_mean)
        xa_mean = xf_mean + K @ innov

        # Update anomalies deterministically using (I - KH/2)
        I_n = np.eye(n)
        KH = K @ H
        T = I_n - 0.5 * KH
        Aa = (T @ Af.T).T  # (Ne, n)

        Xa = Aa + xa_mean
        return Xa

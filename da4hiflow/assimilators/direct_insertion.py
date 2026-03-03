"""Very simple assimilation method for testing pipelines."""

import numpy as np

from da4hiflow.core.obs import ObservationSpec


class DirectInsertionAssimilator:
    """Faulty but deterministic assimilation that copies observations.

    The class does not subclass ``AssimilatorBase``; it provides a
    standalone ``analysis_step`` helper matching the milestone spec.
    """

    def analysis_step(
        self,
        x_forecast: np.ndarray,
        y: np.ndarray,
        spec: ObservationSpec,
    ) -> np.ndarray:
        """Return a state where sensor indices in ``spec`` are replaced.

        Args:
            x_forecast: forecast state vector
            y: observation vector (same length as ``spec.sensor_idx``)
            spec: observation specification
        """
        x = x_forecast.copy()
        x[np.array(spec.sensor_idx)] = y
        return x

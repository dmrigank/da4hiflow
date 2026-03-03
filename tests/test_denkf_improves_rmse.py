"""Test that DEnKF analysis typically improves RMSE over forecast."""

import json
from pathlib import Path

import numpy as np

from da4hiflow.core.runner import run_experiment


def test_denkf_improves_rmse(tmp_path):
    cfg_path = Path("configs/linear1d_denkf.json")
    config = json.loads(cfg_path.read_text())
    out = run_experiment(config, output_root=tmp_path, run_name="denkf_test")

    m = np.genfromtxt(out / "metrics.csv", delimiter=",", names=True)
    forecast_rmse = m["rmse_forecast"]
    analysis_rmse = m["rmse_analysis"]

    frac_better = np.mean(analysis_rmse < forecast_rmse)
    assert frac_better >= 0.5, f"DEnKF did not improve RMSE sufficiently: {frac_better}"

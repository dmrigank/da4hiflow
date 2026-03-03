"""Test that EnKF analysis typically improves RMSE over forecast."""

import json
from pathlib import Path

import numpy as np

from da4hiflow.core.runner import run_experiment


def test_enkf_improves_rmse(tmp_path):
    cfg_path = Path("configs/linear1d_enkf.json")
    config = json.loads(cfg_path.read_text())
    out = run_experiment(config, output_root=tmp_path, run_name="enkf_test")

    m = np.genfromtxt(out / "metrics.csv", delimiter=",", names=True)
    # metrics columns: rmse_forecast, rmse_analysis
    forecast_rmse = m["rmse_forecast"]
    analysis_rmse = m["rmse_analysis"]

    # fraction of steps where analysis is better
    frac_better = np.mean(analysis_rmse < forecast_rmse)
    assert frac_better >= 0.5, f"EnKF did not improve RMSE sufficiently: {frac_better}"

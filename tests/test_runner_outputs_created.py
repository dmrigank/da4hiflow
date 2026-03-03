"""Tests verifying that runner creates expected output files."""

import json
from pathlib import Path

import numpy as np

from da4hiflow.core.runner import run_experiment


def test_runner_outputs_created(tmp_path):
    cfg_path = Path("configs/linear1d_direct_insert.json")
    config = json.loads(cfg_path.read_text())
    out = run_experiment(config, output_root=tmp_path, run_name="test")

    assert out.exists(), "output directory was not created"
    assert (out / "config.json").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "snapshots.npz").exists()

    metrics = np.genfromtxt(out / "metrics.csv", delimiter=",", names=True)
    assert metrics.shape[0] == config["steps"]
    # check columns exist
    assert "rmse_forecast" in metrics.dtype.names
    assert "rmse_analysis" in metrics.dtype.names
    assert "rmse_innov" in metrics.dtype.names

    snaps = np.load(out / "snapshots.npz")
    assert snaps["truth"].shape == (config["steps"] + 1, config["n"])
    # runner now saves mean forecasts/analysis as keys
    assert snaps["forecast_mean"].shape == (config["steps"] + 1, config["n"])
    assert snaps["analysis_mean"].shape == (config["steps"] + 1, config["n"])
    assert snaps["observations"].shape == (config["steps"], len(config["sensor_idx"]))

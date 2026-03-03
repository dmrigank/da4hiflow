"""Reproducibility test for DEnKF runs."""

import json
from pathlib import Path

import numpy as np

from da4hiflow.core.runner import run_experiment


def test_reproducibility_denkf(tmp_path):
    cfg_path = Path("configs/linear1d_denkf.json")
    config = json.loads(cfg_path.read_text())

    out1 = run_experiment(config, output_root=tmp_path, run_name="rep_denkf")
    out2 = run_experiment(config, output_root=tmp_path, run_name="rep_denkf")

    m1 = np.genfromtxt(out1 / "metrics.csv", delimiter=",", names=True)
    m2 = np.genfromtxt(out2 / "metrics.csv", delimiter=",", names=True)
    assert np.allclose(m1["rmse_forecast"], m2["rmse_forecast"])
    assert np.allclose(m1["rmse_analysis"], m2["rmse_analysis"])

    s1 = np.load(out1 / "snapshots.npz")
    s2 = np.load(out2 / "snapshots.npz")
    for key in s1.files:
        assert np.array_equal(s1[key], s2[key])

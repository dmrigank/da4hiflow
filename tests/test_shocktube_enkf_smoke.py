import numpy as np
from da4hiflow.core.runner import run_experiment


def test_shocktube_enkf_run_smoke(tmp_path):
    config = {
        "system": "shocktube",
        "assimilator": "enkf",
        "nx": 48,
        "steps": 3,
        "dt": 0.001,
        "Ne": 6,
        "init_ens_sigma": 1e-3,
        "observation": {
            "type": "point",
            "variable": "rho",
            "sensor_idx": [10, 24, 36],
            "noise_sigma": 1e-3,
        },
        "seed": 2026,
    }

    out = run_experiment(config, output_root=tmp_path, run_name="test_shocktube_enkf")
    # verify outputs exist
    snapshots = np.load(out / "snapshots.npz")
    assert "truth" in snapshots
    truth = snapshots["truth"]
    assert truth.shape[0] == config["steps"] + 1

    # metrics file
    metrics_file = out / "metrics.csv"
    assert metrics_file.exists()

import csv
from pathlib import Path

import numpy as np

from DigitalTwin.analysis.train_uncertainty import extract_run_examples
from DigitalTwin.uncertainty import LEARNED_FEATURE_COLUMNS


def test_training_target_is_derived_from_raw_benign_motion(tmp_path: Path):
    path = tmp_path / (
        "speed-low_surface-smooth_kitchen_floor_latency-wifi_baseline_"
        "route-square0p5x3_attack-none_trial-1_20260101_000000.csv"
    )
    fields = [
        "cycle_ok",
        "gps_valid",
        "sample_ms",
        "edge_arrival_time_s",
        "enc_left",
        "enc_right",
        "lat",
        "lon",
        "az",
        "gz",
        "y",
        "hdop",
        "sat",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(24):
            moving_index = max(0, index - 9)
            writer.writerow(
                {
                    "cycle_ok": "True",
                    "gps_valid": "True",
                    "sample_ms": index * 1000,
                    "edge_arrival_time_s": index + 0.1,
                    "enc_left": -20 * moving_index,
                    "enc_right": -22 * moving_index,
                    "lat": 37.0,
                    "lon": -80.0 + moving_index * 0.000001,
                    "az": 1000 + (index % 3),
                    "gz": 0.1 * (index % 2),
                    "y": 0.2 * moving_index,
                    "hdop": 1.1,
                    "sat": 10,
                }
            )

    X, y = extract_run_examples(path)
    assert X.shape[1] == len(LEARNED_FEATURE_COLUMNS)
    assert y.shape[1] == 3
    assert len(X) == len(y)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert (y > 0).all()

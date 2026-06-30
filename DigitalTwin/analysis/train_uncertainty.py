"""Train a Random Forest uncertainty model from benign CSVs.

This is a pipeline stub: it preserves the proposal's feature contract and model
artifact flow.  With real rover data, replace the default target with empirical
EKF consistency labels derived from benign residuals.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .common import parse_run_name, read_rows


FEATURE_COLUMNS = [
    "dead_reckoning_residual_m",
    "imu_vertical_std",
    "imu_yaw_std",
    "velocity_variance",
    "packet_dt_s",
]
DEFAULT_TARGET_COLUMNS = ["q_xx", "q_yy", "q_tt"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="benign CSV files or glob patterns")
    parser.add_argument("--out", default="DigitalTwin/configs/uncertainty_model.pkl")
    parser.add_argument("--metadata-out", default="DigitalTwin/configs/uncertainty_model.json")
    args = parser.parse_args()

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    paths: list[Path] = []
    for item in args.inputs:
        matches = sorted(Path().glob(item)) if any(ch in item for ch in "*?[") else [Path(item)]
        paths.extend(path for path in matches if path.exists() and parse_run_name(path).get("attack") in {"none", ""})

    X: list[list[float]] = []
    y: list[list[float]] = []
    for path in paths:
        for row in read_rows(path):
            X.append([float(row[column]) for column in FEATURE_COLUMNS])
            y.append([float(row[column]) for column in DEFAULT_TARGET_COLUMNS])

    if len(X) < 10:
        raise RuntimeError("need at least 10 benign rows to train uncertainty model")

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.25, random_state=7)
    model = RandomForestRegressor(n_estimators=120, random_state=7, min_samples_leaf=3)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred, multioutput="raw_values")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model": "RandomForestRegressor",
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": DEFAULT_TARGET_COLUMNS,
        "target_note": "stub target uses current q diagonal; replace with empirical consistency labels from real benign data",
        "rows": len(X),
        "source_files": [str(path) for path in paths],
        "mae": {column: float(value) for column, value in zip(DEFAULT_TARGET_COLUMNS, mae)},
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

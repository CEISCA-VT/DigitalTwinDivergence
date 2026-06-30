"""CSV logging for experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "time_s",
    "seq",
    "gps_x_m",
    "gps_y_m",
    "truth_x_m",
    "truth_y_m",
    "ekf_x_m",
    "ekf_y_m",
    "ekf_theta_rad",
    "innovation_x_m",
    "innovation_y_m",
    "mahalanobis",
    "threshold",
    "lambda_star",
    "lambda_max_s",
    "detected",
    "epsilon_min_m",
    "epsilon_stealth_max_m",
    "confidence",
    "envelope_region",
    "q_xx",
    "q_yy",
    "q_tt",
    "s_xx",
    "s_yy",
    "dead_reckoning_residual_m",
    "imu_vertical_std",
    "imu_yaw_std",
    "velocity_variance",
    "packet_dt_s",
    "arrival_dt_s",
    "transport_latency_s",
    "attack_label",
]


class CSVExperimentLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDNAMES)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow({name: row.get(name, "") for name in FIELDNAMES})

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "CSVExperimentLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

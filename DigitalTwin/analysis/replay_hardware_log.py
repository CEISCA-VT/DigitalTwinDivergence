"""Replay UGV01 bench telemetry CSVs through the digital-twin pipeline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from DigitalTwin.detector import InnovationDetector
from DigitalTwin.ekf import RoverEKF
from DigitalTwin.kinematics import ugv01_calibrated_geometry
from DigitalTwin.logger import CSVExperimentLogger
from DigitalTwin.telemetry import gps_to_local_xy
from DigitalTwin.uncertainty import (
    TelemetryDrivenUncertaintyEstimator,
    TelemetryStatisticsWindow,
    add_turn_slip_uncertainty,
)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", "None", "null"}:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float(row, key, float(default))))


def _bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {"1", "true", "yes"}


def replay_hardware_log(input_csv: str | Path, output_csv: str | Path) -> Path:
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    with input_path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("cycle_ok") == "True" and _bool(row, "gps_valid")]

    if not rows:
        raise RuntimeError(f"{input_path} has no successful GPS-valid rows")

    origin_lat = _float(rows[0], "lat")
    origin_lon = _float(rows[0], "lon")

    geometry = ugv01_calibrated_geometry()
    uncertainty = TelemetryDrivenUncertaintyEstimator()
    stats = TelemetryStatisticsWindow()
    detector = InnovationDetector()
    ekf = RoverEKF()

    prev_time_s: float | None = None
    prev_arrival_s: float | None = None
    prev_left = _int(rows[0], "enc_left")
    prev_right = _int(rows[0], "enc_right")
    prev_seq: int | None = None
    last_dead_reckoning_residual_m = 0.0

    with CSVExperimentLogger(output_path) as logger:
        for row in rows:
            sample_time_s = _float(row, "sample_ms", _float(row, "millis")) / 1000.0
            edge_arrival_s = _float(row, "edge_arrival_time_s", _float(row, "t_edge_rx_ns") / 1_000_000_000.0)
            dt_s = 0.1 if prev_time_s is None else max(sample_time_s - prev_time_s, 1e-6)
            arrival_dt_s = dt_s if prev_arrival_s is None else max(edge_arrival_s - prev_arrival_s, 1e-6)
            prev_time_s = sample_time_s
            prev_arrival_s = edge_arrival_s

            enc_left = _int(row, "enc_left")
            enc_right = _int(row, "enc_right")
            delta_left = enc_left - prev_left
            delta_right = enc_right - prev_right
            prev_left = enc_left
            prev_right = enc_right
            v_est, omega_est = geometry.ticks_to_control(delta_left, delta_right, dt_s)

            gps_xy = np.array(gps_to_local_xy(_float(row, "lat"), _float(row, "lon"), origin_lat, origin_lon))
            stats.observe(
                dead_reckoning_residual_m=last_dead_reckoning_residual_m,
                # T:147 reports acceleration in mg and gyro rate in deg/s.
                accel_z=_float(row, "az") * 9.80665 / 1000.0,
                gyro_z=math.radians(_float(row, "gz")),
                velocity_mps=v_est,
                packet_dt_s=arrival_dt_s,
            )
            features = stats.features(
                gps_hdop=_float(row, "hdop", 99.99),
                gps_satellites=_int(row, "sat"),
                fallback_dt_s=arrival_dt_s,
            )
            Q = uncertainty.process_covariance(features, dt_s)
            Q = add_turn_slip_uncertainty(Q, omega_est, dt_s)
            R = uncertainty.measurement_covariance(features)
            ekf.predict(v_est, omega_est, dt_s, Q)
            dead_reckoning_residual_m = float(np.linalg.norm(gps_xy - ekf.state.x[:2]))
            last_dead_reckoning_residual_m = dead_reckoning_residual_m
            ekf.update_gps(gps_xy, R)
            detection = detector.evaluate(ekf.last_innovation, ekf.last_S)

            seq = _int(row, "seq", _int(row, "sample_idx"))
            packet_loss_count = 0 if prev_seq is None else max(0, seq - prev_seq - 1)
            stale_packet = prev_seq is not None and seq <= prev_seq
            prev_seq = seq

            logger.write(
                {
                    "time_s": sample_time_s,
                    "seq": seq,
                    "source_sample_time_s": sample_time_s,
                    "edge_send_time_s": _float(row, "t_edge_send_ns") / 1_000_000_000.0,
                    "edge_arrival_time_s": edge_arrival_s,
                    "queue_release_time_s": edge_arrival_s,
                    "estimate_time_s": edge_arrival_s,
                    "alarm_time_s": edge_arrival_s if detection.detected else "",
                    "clock_offset_s": _float(row, "clock_offset_s"),
                    "clock_calibrated": int(_bool(row, "clock_calibrated")),
                    "packet_loss_count": packet_loss_count,
                    "stale_packet": int(stale_packet),
                    "queue_depth": _int(row, "queue_depth"),
                    "gps_x_m": gps_xy[0],
                    "gps_y_m": gps_xy[1],
                    "truth_x_m": "",
                    "truth_y_m": "",
                    "ekf_x_m": ekf.state.x[0],
                    "ekf_y_m": ekf.state.x[1],
                    "ekf_theta_rad": ekf.state.x[2],
                    "innovation_x_m": ekf.last_innovation[0],
                    "innovation_y_m": ekf.last_innovation[1],
                    "mahalanobis": detection.mahalanobis,
                    "threshold": detection.threshold,
                    "lambda_star": detection.lambda_star,
                    "lambda_max_s": detection.lambda_max_s,
                    "detected": int(detection.detected),
                    "epsilon_min_m": detection.epsilon_min_m,
                    "epsilon_stealth_max_m": detection.epsilon_stealth_max_m,
                    "confidence": detection.confidence,
                    "envelope_region": detection.envelope_region,
                    "q_xx": Q[0, 0],
                    "q_yy": Q[1, 1],
                    "q_tt": Q[2, 2],
                    "s_xx": ekf.last_S[0, 0],
                    "s_yy": ekf.last_S[1, 1],
                    "dead_reckoning_residual_m": features.dead_reckoning_residual_m,
                    "imu_vertical_std": features.imu_vertical_std,
                    "imu_yaw_std": features.imu_yaw_std,
                    "velocity_variance": features.velocity_variance,
                    "packet_dt_s": features.packet_dt_s,
                    "arrival_dt_s": arrival_dt_s,
                    "transport_latency_s": _float(row, "http_latency_ms") / 1000.0,
                    "attack_label": "hardware-bench",
                }
            )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.out) if args.out else input_path.with_name(input_path.stem + "_digital_twin.csv")
    print(replay_hardware_log(input_path, output_path))


if __name__ == "__main__":
    main()

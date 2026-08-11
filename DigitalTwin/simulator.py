"""Synthetic rover telemetry generator and end-to-end pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path

import numpy as np

from .attack import AttackConfig, AttackInjector
from .detector import InnovationDetector
from .ekf import RoverEKF
from .kinematics import DifferentialDriveGeometry, integrate_unicycle, trajectory_control
from .latency import LatencyQueue
from .logger import CSVExperimentLogger
from .telemetry import TelemetryPacket, gps_to_local_xy, local_xy_to_gps
from .timing import SessionClockCalibrator
from .uncertainty import (
    TelemetryDrivenUncertaintyEstimator,
    TelemetryStatisticsWindow,
    add_turn_slip_uncertainty,
)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    trajectory: str = "square"
    duration_s: float = 60.0
    dt_s: float = 0.1
    speed_mps: float = 0.25
    latency_ms: float = 0.0
    latency_jitter_ms: float | None = None
    terrain_index: float = 0.0
    gps_noise_m: float = 0.45
    origin_lat_deg: float = 40.0
    origin_lon_deg: float = -74.0
    seed: int = 7


def run_simulation(config: SimulationConfig, attack: AttackConfig, output_csv: str | Path) -> Path:
    geometry = DifferentialDriveGeometry()
    rng = random.Random(config.seed)
    uncertainty = TelemetryDrivenUncertaintyEstimator()
    stats = TelemetryStatisticsWindow()
    detector = InnovationDetector(
        measurement_dim=2,
        false_alarm_probability=0.05,
        target_detection_probability=0.95,
        blind_epsilon_m=5.0,
    )
    injector = AttackInjector(attack, dt_s=config.dt_s, seed=config.seed + 101)
    latency = LatencyQueue(config.latency_ms, jitter_ms=config.latency_jitter_ms, seed=config.seed + 303)
    ekf = RoverEKF()
    clock = SessionClockCalibrator(min_samples=1)

    truth = np.array([0.0, 0.0, 0.0], dtype=float)
    prev_left = 0
    prev_right = 0
    total_left = 0
    total_right = 0
    prev_packet_time_s: float | None = None
    prev_arrival_time_s: float | None = None
    prev_seq: int | None = None
    last_dead_reckoning_residual_m = 0.0

    output_path = Path(output_csv)
    with CSVExperimentLogger(output_path) as logger:
        steps = int(config.duration_s / config.dt_s)
        for seq in range(steps):
            t_s = seq * config.dt_s
            v_cmd, omega_cmd = trajectory_control(config.trajectory, t_s, config.speed_mps)
            truth = integrate_unicycle(truth, v_cmd, omega_cmd, config.dt_s)
            dl, dr = geometry.control_to_ticks(v_cmd, omega_cmd, config.dt_s)
            total_left += dl
            total_right += dr

            noisy_xy = truth[:2] + np.array(
                [rng.gauss(0.0, config.gps_noise_m), rng.gauss(0.0, config.gps_noise_m)]
            )
            attacked_xy, attack_label = injector.apply(t_s, noisy_xy)
            lat, lon = local_xy_to_gps(attacked_xy[0], attacked_xy[1], config.origin_lat_deg, config.origin_lon_deg)
            terrain_accel = rng.gauss(0.0, 0.25 * config.terrain_index)
            terrain_gyro = rng.gauss(0.0, 0.04 * config.terrain_index)
            packet = TelemetryPacket(
                seq=seq,
                timestamp_us=int(t_s * 1_000_000),
                enc_left_ticks=total_left,
                enc_right_ticks=total_right,
                accel_z=9.81 + terrain_accel,
                gyro_z=omega_cmd + terrain_gyro,
                gps_lat_deg=lat,
                gps_lon_deg=lon,
                gps_speed_mps=v_cmd,
                gps_course_rad=float(truth[2]),
                gps_fix_type=3,
                gps_satellites=10,
                gps_hdop_cm=120,
            )
            latency.push(t_s, (packet.pack(), truth.copy(), attack_label))

            for delivered in latency.pop_ready(t_s):
                raw_frame, delayed_truth, delayed_label = delivered.item
                parsed = TelemetryPacket.unpack(raw_frame)
                gps_xy = np.array(gps_to_local_xy(parsed.gps_lat_deg, parsed.gps_lon_deg, config.origin_lat_deg, config.origin_lon_deg))
                packet_time_s = parsed.timestamp_us / 1_000_000.0
                clock_estimate = clock.observe(packet_time_s, delivered.delivery_s)
                sensor_dt_s = config.dt_s if prev_packet_time_s is None else max(packet_time_s - prev_packet_time_s, 1e-6)
                stale_packet = prev_packet_time_s is not None and packet_time_s <= prev_packet_time_s
                prev_packet_time_s = packet_time_s
                arrival_dt_s = sensor_dt_s if prev_arrival_time_s is None else max(delivered.delivery_s - prev_arrival_time_s, 1e-6)
                prev_arrival_time_s = delivered.delivery_s
                transport_latency_s = max(0.0, delivered.delivery_s - delivered.generated_s)
                effective_packet_dt_s = arrival_dt_s + transport_latency_s
                packet_loss_count = 0 if prev_seq is None else max(0, parsed.seq - prev_seq - 1)
                prev_seq = parsed.seq
                delta_left = parsed.enc_left_ticks - prev_left
                delta_right = parsed.enc_right_ticks - prev_right
                prev_left = parsed.enc_left_ticks
                prev_right = parsed.enc_right_ticks
                v_est, omega_est = geometry.ticks_to_control(delta_left, delta_right, sensor_dt_s)

                stats.observe(
                    dead_reckoning_residual_m=last_dead_reckoning_residual_m,
                    accel_z=parsed.accel_z,
                    gyro_z=parsed.gyro_z,
                    velocity_mps=v_est,
                    packet_dt_s=effective_packet_dt_s,
                )

                features = stats.features(
                    gps_hdop=parsed.gps_hdop_cm / 100.0,
                    gps_satellites=parsed.gps_satellites,
                    fallback_dt_s=arrival_dt_s,
                )
                Q = uncertainty.process_covariance(features, sensor_dt_s)
                Q = add_turn_slip_uncertainty(Q, omega_est, sensor_dt_s)
                R = uncertainty.measurement_covariance(features)
                ekf.predict(v_est, omega_est, sensor_dt_s, Q)
                dead_reckoning_residual_m = float(np.linalg.norm(gps_xy - ekf.state.x[:2]))
                last_dead_reckoning_residual_m = dead_reckoning_residual_m
                ekf.update_gps(gps_xy, R)
                detection = detector.evaluate(ekf.last_innovation, ekf.last_S)
                alarm_time_s = delivered.delivery_s if detection.detected else ""

                logger.write(
                    {
                        "time_s": parsed.timestamp_us / 1_000_000.0,
                        "seq": parsed.seq,
                        "source_sample_time_s": packet_time_s,
                        "edge_send_time_s": delivered.generated_s,
                        "edge_arrival_time_s": delivered.delivery_s,
                        "queue_release_time_s": delivered.delivery_s,
                        "estimate_time_s": delivered.delivery_s,
                        "alarm_time_s": alarm_time_s,
                        "clock_offset_s": clock_estimate.offset_s,
                        "clock_calibrated": int(clock_estimate.calibrated),
                        "packet_loss_count": packet_loss_count,
                        "stale_packet": int(stale_packet),
                        "queue_depth": delivered.queue_depth_after_pop,
                        "gps_x_m": gps_xy[0],
                        "gps_y_m": gps_xy[1],
                        "truth_x_m": delayed_truth[0],
                        "truth_y_m": delayed_truth[1],
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
                        "transport_latency_s": transport_latency_s,
                        "attack_label": delayed_label,
                    }
                )
    return output_path

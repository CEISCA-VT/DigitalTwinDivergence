"""Compare UGV01 encoder/IMU prediction with fixed-camera AprilTag ground truth."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from DigitalTwin.kinematics import (
    DifferentialDriveGeometry,
    UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M,
    integrate_unicycle,
    wrap_angle,
)
from DigitalTwin.motion import DEFAULT_MOTION_FUSION_POLICY, fuse_encoder_imu_motion


DEFAULT_TRACKING = Path(
    "DigitalTwin/datasets/analysis/apriltag_trial1_every_frame/apriltag_still_summary.json"
)
DEFAULT_TELEMETRY = Path(
    "raw_logs/telemetry/ugv_t147_interactive_20260805_174551.csv"
)
DEFAULT_OUTPUT = Path("DigitalTwin/datasets/analysis/apriltag_trial1_fidelity")
DEFAULT_INTERVALS = ((0.0, 16.93), (140.88, 152.21), (213.68, 226.58))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _in_interval(value: float, intervals: tuple[tuple[float, float], ...]) -> bool:
    return any(start <= value <= end for start, end in intervals)


def _rolling_median(values: np.ndarray, radius: int = 2) -> np.ndarray:
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        result[index] = np.median(values[start:end], axis=0)
    return result


def _load_ground_truth(
    path: Path,
    intervals: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fps = float(payload["video"]["fps"])
    selected = [
        row
        for row in payload["frame_summaries"]
        if row.get("rover_xy_m") is not None
        and _in_interval(float(row["time_s"]), intervals)
    ]
    if not selected:
        raise RuntimeError("selected intervals contain no mapped rover poses")

    # Keep approximately 10 Hz so video frames do not masquerade as independent samples.
    minimum_step_s = 0.09
    sampled = []
    last_time = -math.inf
    for row in selected:
        time_s = float(row["time_s"])
        if time_s - last_time >= minimum_step_s:
            sampled.append(row)
            last_time = time_s

    times = np.asarray([float(row["time_s"]) for row in sampled], dtype=float)
    positions = _rolling_median(
        np.asarray([row["rover_xy_m"] for row in sampled], dtype=float)
    )
    headings = np.asarray(
        [float(row["rover_heading_rad"]) for row in sampled], dtype=float
    )
    interval_ids = np.asarray(
        [
            next(
                index
                for index, (start, end) in enumerate(intervals, start=1)
                if start <= float(row["time_s"]) <= end
            )
            for row in sampled
        ],
        dtype=int,
    )
    tracking_status = np.asarray(
        [str(row.get("rover_tracking_status", "decoded")) for row in sampled],
        dtype=object,
    )
    return times, positions, headings, interval_ids, tracking_status


def _load_prediction(
    path: Path,
    *,
    effective_track_width_m: float = UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M,
    distance_scale: float = 1.0,
    clockwise_track_width_m: float | None = None,
    counterclockwise_track_width_m: float | None = None,
    gyro_weight: float = DEFAULT_MOTION_FUSION_POLICY.gyro_weight,
    gyro_scale: float = 1.0,
) -> dict[str, np.ndarray | float]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("cycle_ok") == "True"]
    if len(rows) < 2:
        raise RuntimeError("telemetry log has fewer than two successful rows")

    raw_times = np.asarray(
        [_f(row, "source_sample_time_s", _f(row, "sample_ms") / 1000.0) for row in rows]
    )
    elapsed = raw_times - raw_times[0]
    geometry = DifferentialDriveGeometry(wheel_base_m=effective_track_width_m)
    encoder_controls = np.zeros((len(rows), 2), dtype=float)
    for index in range(1, len(rows)):
        dt_s = max(float(elapsed[index] - elapsed[index - 1]), 1e-3)
        delta_left = int(round(_f(rows[index], "enc_left"))) - int(
            round(_f(rows[index - 1], "enc_left"))
        )
        delta_right = int(round(_f(rows[index], "enc_right"))) - int(
            round(_f(rows[index - 1], "enc_right"))
        )
        encoder_controls[index] = geometry.ticks_to_control(
            delta_left, delta_right, dt_s
        )
    base_yaw = encoder_controls[:, 1].copy()
    encoder_controls[:, 0] *= distance_scale
    clockwise_width = clockwise_track_width_m or effective_track_width_m
    counterclockwise_width = (
        counterclockwise_track_width_m or effective_track_width_m
    )
    direction_widths = np.where(
        base_yaw >= 0.0,
        counterclockwise_width,
        clockwise_width,
    )
    encoder_controls[:, 1] *= (
        distance_scale * effective_track_width_m / direction_widths
    )

    mission_candidates = np.flatnonzero(
        (np.abs(encoder_controls[:, 0]) > 0.02)
        | (np.abs(encoder_controls[:, 1]) > 0.15)
    )
    mission_start = int(mission_candidates[0]) if len(mission_candidates) else 1
    gyro_radps = (
        np.radians(np.asarray([_f(row, "gz") for row in rows], dtype=float))
        * gyro_scale
    )
    motion_policy = replace(DEFAULT_MOTION_FUSION_POLICY, gyro_weight=gyro_weight)
    fusion = fuse_encoder_imu_motion(
        encoder_controls, gyro_radps, mission_start, motion_policy
    )
    return {
        "elapsed_s": elapsed,
        "controls": fusion.controls,
        "encoder_controls": encoder_controls,
        "corrected_gyro_radps": fusion.corrected_gyro_radps,
        "gyro_bias_radps": fusion.gyro_bias_radps,
        "mission_start": float(mission_start),
    }


def _integrate(
    elapsed: np.ndarray,
    controls: np.ndarray,
    initial_xy: np.ndarray,
    initial_heading: float,
) -> np.ndarray:
    states = np.zeros((len(elapsed), 3), dtype=float)
    states[0] = [initial_xy[0], initial_xy[1], initial_heading]
    for index in range(1, len(elapsed)):
        dt_s = max(float(elapsed[index] - elapsed[index - 1]), 1e-3)
        states[index] = integrate_unicycle(
            states[index - 1],
            float(controls[index, 0]),
            float(controls[index, 1]),
            dt_s,
        )
    return states


def _interpolate_states(
    elapsed: np.ndarray, states: np.ndarray, query: np.ndarray
) -> np.ndarray:
    heading = np.unwrap(states[:, 2])
    return np.column_stack(
        [
            np.interp(query, elapsed, states[:, 0]),
            np.interp(query, elapsed, states[:, 1]),
            np.interp(query, elapsed, heading),
        ]
    )


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else math.nan


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _rpe(
    times: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    interval_ids: np.ndarray,
    horizon_s: float = 1.0,
) -> np.ndarray:
    errors = []
    for index, time_s in enumerate(times):
        candidates = np.flatnonzero(
            (interval_ids == interval_ids[index])
            & (np.abs(times - (time_s + horizon_s)) <= 0.06)
        )
        if not len(candidates):
            continue
        later = int(candidates[0])
        truth_delta = truth[later] - truth[index]
        estimate_delta = estimate[later] - estimate[index]
        errors.append(float(np.linalg.norm(estimate_delta - truth_delta)))
    return np.asarray(errors, dtype=float)


def _motion_activity(
    times: np.ndarray, positions: np.ndarray, headings: np.ndarray
) -> np.ndarray:
    translation = np.linalg.norm(np.gradient(positions, times, axis=0), axis=1)
    rotation = np.abs(np.gradient(np.unwrap(headings), times))
    activity = translation + 0.15 * rotation
    window = min(11, len(activity))
    if window >= 3:
        activity = np.convolve(activity, np.ones(window) / window, mode="same")
    return activity


def _activity_sync_offset(
    gt_time: np.ndarray,
    gt_xy: np.ndarray,
    gt_heading: np.ndarray,
    elapsed: np.ndarray,
    controls: np.ndarray,
) -> tuple[float, float, float]:
    ground_truth_activity = _motion_activity(gt_time, gt_xy, gt_heading)
    telemetry_activity = np.abs(controls[:, 0]) + 0.15 * np.abs(controls[:, 1])
    best_score = -math.inf
    best_offset = 0.0
    scored_offsets: list[tuple[float, float]] = []
    for offset in np.arange(-20.0, 20.001, 0.05):
        query = gt_time - offset
        valid = (query >= elapsed[0]) & (query <= elapsed[-1])
        if int(valid.sum()) < 30:
            continue
        interpolated = np.interp(query[valid], elapsed, telemetry_activity)
        observed = ground_truth_activity[valid]
        if float(np.std(interpolated)) <= 1e-9 or float(np.std(observed)) <= 1e-9:
            continue
        score = float(np.corrcoef(observed, interpolated)[0, 1])
        scored_offsets.append((float(offset), score))
        if score > best_score:
            best_score = score
            best_offset = float(offset)
    if not math.isfinite(best_score):
        raise RuntimeError("could not align camera and telemetry motion activity")
    near_peak = [
        offset for offset, score in scored_offsets if score >= best_score - 0.01
    ]
    uncertainty = (
        0.5 * (max(near_peak) - min(near_peak)) if len(near_peak) > 1 else 0.05
    )
    return best_offset, best_score, float(uncertainty)


def _stationary_jitter(
    times: np.ndarray, positions: np.ndarray, headings: np.ndarray
) -> np.ndarray:
    activity = _motion_activity(times, positions, headings)
    stationary = activity <= 0.02
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(np.append(stationary, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if not runs:
        return np.asarray([], dtype=float)
    start, end = max(runs, key=lambda run: run[1] - run[0])
    selected = positions[start:end]
    if len(selected) < 5:
        return np.asarray([], dtype=float)
    center = np.median(selected, axis=0)
    return np.linalg.norm(selected - center, axis=1)


def analyze(
    tracking_path: Path,
    telemetry_path: Path,
    output_dir: Path,
    intervals: tuple[tuple[float, float], ...] = DEFAULT_INTERVALS,
    sync_mode: str = "onset",
    effective_track_width_m: float = UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M,
    distance_scale: float = 1.0,
    clockwise_track_width_m: float | None = None,
    counterclockwise_track_width_m: float | None = None,
    gyro_weight: float = DEFAULT_MOTION_FUSION_POLICY.gyro_weight,
    gyro_scale: float = 1.0,
) -> dict[str, object]:
    gt_time, gt_xy, gt_heading, interval_ids, tracking_status = _load_ground_truth(
        tracking_path, intervals
    )
    prediction = _load_prediction(
        telemetry_path,
        effective_track_width_m=effective_track_width_m,
        distance_scale=distance_scale,
        clockwise_track_width_m=clockwise_track_width_m,
        counterclockwise_track_width_m=counterclockwise_track_width_m,
        gyro_weight=gyro_weight,
        gyro_scale=gyro_scale,
    )
    elapsed = np.asarray(prediction["elapsed_s"], dtype=float)
    controls = np.asarray(prediction["controls"], dtype=float)
    encoder_controls = np.asarray(prediction["encoder_controls"], dtype=float)

    first_motion = np.flatnonzero(np.abs(controls[:, 0]) > 0.02)
    reverse_model_axis = bool(
        len(first_motion) and controls[int(first_motion[0]), 0] < 0.0
    )
    synchronization_score = None
    gt_onset_s = None
    telemetry_onset_s = None
    if sync_mode == "activity":
        best_offset, synchronization_score, synchronization_uncertainty_s = _activity_sync_offset(
            gt_time, gt_xy, gt_heading, elapsed, controls
        )
        anchor_telemetry_s = float(gt_time[0] - best_offset)
        anchor_index = int(np.argmin(np.abs(elapsed - anchor_telemetry_s)))
        heading_anchor = gt_time <= gt_time[0] + 1.0
        initial_xy = gt_xy[0].copy()
        physical_heading = math.atan2(
            float(np.mean(np.sin(gt_heading[heading_anchor]))),
            float(np.mean(np.cos(gt_heading[heading_anchor]))),
        )
        integration_elapsed = elapsed[anchor_index:]
        integration_controls = controls[anchor_index:].copy()
        integration_encoder_controls = encoder_controls[anchor_index:].copy()
        integration_controls[0] = 0.0
        integration_encoder_controls[0] = 0.0
        synchronization_method = "full_motion_activity_correlation"
    elif sync_mode == "onset":
        stationary = gt_time <= min(10.0, intervals[0][1])
        initial_xy = np.median(gt_xy[stationary], axis=0)
        physical_heading = math.atan2(
            float(np.mean(np.sin(gt_heading[stationary]))),
            float(np.mean(np.cos(gt_heading[stationary]))),
        )
        gt_origin = np.median(gt_xy[stationary], axis=0)
        gt_displacement = np.linalg.norm(gt_xy - gt_origin, axis=1)
        gt_onset_candidates = np.flatnonzero(
            (interval_ids == 1) & (gt_displacement >= 0.05)
        )
        encoder_distance = np.zeros(len(elapsed), dtype=float)
        for index in range(1, len(elapsed)):
            dt_s = max(float(elapsed[index] - elapsed[index - 1]), 1e-3)
            encoder_distance[index] = (
                encoder_distance[index - 1]
                + abs(float(encoder_controls[index, 0])) * dt_s
            )
        encoder_onset_candidates = np.flatnonzero(encoder_distance >= 0.05)
        if not len(gt_onset_candidates) or not len(encoder_onset_candidates):
            raise RuntimeError("could not identify the 5 cm synchronization event")
        gt_onset_s = float(gt_time[int(gt_onset_candidates[0])])
        telemetry_onset_index = int(encoder_onset_candidates[0])
        telemetry_onset_s = float(elapsed[telemetry_onset_index])
        best_offset = gt_onset_s - telemetry_onset_s
        previous_time = float(elapsed[max(0, telemetry_onset_index - 1)])
        synchronization_uncertainty_s = telemetry_onset_s - previous_time
        synchronization_method = "first_5cm_motion_onset"
        integration_elapsed = elapsed
        integration_controls = controls
        integration_encoder_controls = encoder_controls
    else:
        raise ValueError(f"unsupported synchronization mode {sync_mode!r}")

    model_heading = wrap_angle(
        physical_heading + (math.pi if reverse_model_axis else 0.0)
    )
    fused_states = _integrate(
        integration_elapsed, integration_controls, initial_xy, model_heading
    )
    encoder_states = _integrate(
        integration_elapsed,
        integration_encoder_controls,
        initial_xy,
        model_heading,
    )

    query = gt_time - best_offset
    valid = (query >= integration_elapsed[0]) & (query <= integration_elapsed[-1])
    gt_time = gt_time[valid]
    gt_xy = gt_xy[valid]
    gt_heading = gt_heading[valid]
    interval_ids = interval_ids[valid]
    tracking_status = tracking_status[valid]
    query = query[valid]
    fused = _interpolate_states(integration_elapsed, fused_states, query)
    encoder = _interpolate_states(integration_elapsed, encoder_states, query)
    fused_error = np.linalg.norm(fused[:, :2] - gt_xy, axis=1)
    encoder_error = np.linalg.norm(encoder[:, :2] - gt_xy, axis=1)
    physical_estimated_heading = fused[:, 2] - (
        math.pi if reverse_model_axis else 0.0
    )
    heading_error = np.asarray(
        [
            abs(wrap_angle(float(estimate - truth)))
            for estimate, truth in zip(physical_estimated_heading, gt_heading)
        ]
    )
    rpe_1s = _rpe(gt_time, gt_xy, fused[:, :2], interval_ids)

    stationary_error = _stationary_jitter(gt_time, gt_xy, gt_heading)
    per_interval = []
    for interval_id, (start, end) in enumerate(intervals, start=1):
        mask = interval_ids == interval_id
        if not mask.any():
            continue
        truth_length = _path_length(gt_xy[mask])
        estimated_length = _path_length(fused[mask, :2])
        errors = fused_error[mask]
        per_interval.append(
            {
                "interval": interval_id,
                "video_start_s": start,
                "video_end_s": end,
                "samples": int(mask.sum()),
                "position_rmse_m": _rmse(errors),
                "position_median_m": float(np.median(errors)),
                "position_p95_m": float(np.quantile(errors, 0.95)),
                "truth_path_length_m": truth_length,
                "estimated_path_length_m": estimated_length,
                "path_length_ratio": (
                    estimated_length / truth_length if truth_length > 1e-6 else None
                ),
            }
        )

    summary: dict[str, object] = {
        "schema": "ugv01_apriltag_fidelity_v1",
        "tracking_source": str(tracking_path),
        "telemetry_source": str(telemetry_path),
        "selected_intervals_s": [list(interval) for interval in intervals],
        "effective_track_width_m": effective_track_width_m,
        "gyro_weight": gyro_weight,
        "gyro_scale": gyro_scale,
        "evaluation_samples_approximately_10hz": int(len(gt_time)),
        "directly_decoded_evaluation_samples": int(
            np.sum(tracking_status == "decoded")
        ),
        "recovered_evaluation_samples": int(
            np.sum(tracking_status != "decoded")
        ),
        "directly_decoded_evaluation_fraction": float(
            np.mean(tracking_status == "decoded")
        ),
        "estimated_video_minus_telemetry_offset_s": best_offset,
        "synchronization_method": synchronization_method,
        "synchronization_correlation": synchronization_score,
        "synchronization_event_video_s": gt_onset_s,
        "synchronization_event_telemetry_s": telemetry_onset_s,
        "synchronization_uncertainty_s": synchronization_uncertainty_s,
        "position_ate_rmse_m": _rmse(fused_error),
        "position_error_median_m": float(np.median(fused_error)),
        "position_error_p95_m": float(np.quantile(fused_error, 0.95)),
        "position_error_max_m": float(np.max(fused_error)),
        "within_0p05_fraction": float(np.mean(fused_error <= 0.05)),
        "within_0p10_fraction": float(np.mean(fused_error <= 0.10)),
        "within_0p25_fraction": float(np.mean(fused_error <= 0.25)),
        "rpe_1s_rmse_m": _rmse(rpe_1s),
        "rpe_1s_median_m": float(np.median(rpe_1s)) if len(rpe_1s) else None,
        "heading_mae_deg": float(np.degrees(np.mean(heading_error))),
        "heading_p95_deg": float(np.degrees(np.quantile(heading_error, 0.95))),
        "encoder_only_ate_rmse_m": _rmse(encoder_error),
        "gyro_bias_deg_s": math.degrees(float(prediction["gyro_bias_radps"])),
        "camera_stationary_jitter_rmse_m": _rmse(stationary_error),
        "camera_stationary_jitter_p95_m": (
            float(np.quantile(stationary_error, 0.95))
            if len(stationary_error)
            else None
        ),
        "per_interval": per_interval,
        "interpretation": (
            "Preliminary prediction-only fidelity. The run has no GPS updates and no "
            "hardware camera/telemetry synchronization pulse; the scalar time offset "
            "was estimated from the recorded motion sequence."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fidelity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    fieldnames = [
        "video_time_s",
        "telemetry_elapsed_s",
        "interval",
        "truth_x_m",
        "truth_y_m",
        "truth_heading_deg",
        "twin_x_m",
        "twin_y_m",
        "twin_heading_deg",
        "position_error_m",
        "heading_error_deg",
        "tracking_status",
    ]
    with (output_dir / "aligned_fidelity_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(gt_time)):
            writer.writerow(
                {
                    "video_time_s": gt_time[index],
                    "telemetry_elapsed_s": query[index],
                    "interval": interval_ids[index],
                    "truth_x_m": gt_xy[index, 0],
                    "truth_y_m": gt_xy[index, 1],
                    "truth_heading_deg": math.degrees(gt_heading[index]),
                    "twin_x_m": fused[index, 0],
                    "twin_y_m": fused[index, 1],
                    "twin_heading_deg": math.degrees(physical_estimated_heading[index]),
                    "position_error_m": fused_error[index],
                    "heading_error_deg": math.degrees(heading_error[index]),
                    "tracking_status": tracking_status[index],
                }
            )

    report = [
        "# AprilTag Digital-Twin Fidelity Pilot",
        "",
        f"- Position ATE RMSE: **{summary['position_ate_rmse_m']:.3f} m**",
        f"- Median / p95 position error: **{summary['position_error_median_m']:.3f} / {summary['position_error_p95_m']:.3f} m**",
        f"- 1-second RPE RMSE: **{summary['rpe_1s_rmse_m']:.3f} m**",
        f"- Heading MAE / p95: **{summary['heading_mae_deg']:.1f} / {summary['heading_p95_deg']:.1f} deg**",
        f"- Samples within 5 / 10 / 25 cm: **{100*summary['within_0p05_fraction']:.1f}% / {100*summary['within_0p10_fraction']:.1f}% / {100*summary['within_0p25_fraction']:.1f}%**",
        f"- Video-minus-telemetry offset ({synchronization_method}): **{best_offset:.2f} s**",
        f"- Synchronization uncertainty from telemetry sampling: **up to {synchronization_uncertainty_s:.2f} s**",
        f"- Camera stationary jitter RMSE / p95: **{summary['camera_stationary_jitter_rmse_m']:.3f} / {summary['camera_stationary_jitter_p95_m']:.3f} m**",
        f"- Directly decoded / recovered evaluation samples: **{summary['directly_decoded_evaluation_samples']} / {summary['recovered_evaluation_samples']}**",
        "",
        "## Selected Windows",
        "",
        "| Window | Video interval | Samples | ATE RMSE | Median | p95 | Path ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_interval:
        ratio = row["path_length_ratio"]
        report.append(
            f"| {row['interval']} | {row['video_start_s']:.2f}-{row['video_end_s']:.2f} s | "
            f"{row['samples']} | {row['position_rmse_m']:.3f} m | "
            f"{row['position_median_m']:.3f} m | {row['position_p95_m']:.3f} m | "
            f"{ratio:.3f} |"
        )
    report.extend(
        [
            "",
            "## Scope",
            "",
            "This is a preliminary encoder/IMU prediction-versus-AprilTag result. GPS was disconnected, so it does not evaluate GPS-aided EKF correction or attack detection. The camera and telemetry lacked a common hardware synchronization event, so synchronization was estimated from the recorded motion sequence. The elevated rover tag was mapped through the floor-plane homography; unmeasured parallax and tag-to-rover extrinsics therefore remain limitations.",
            "",
        ]
    )
    (output_dir / "fidelity_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for interval_id in sorted(set(interval_ids.tolist())):
        mask = interval_ids == interval_id
        axis.plot(
            gt_xy[mask, 0], gt_xy[mask, 1], linewidth=2.0, label=f"AprilTag {interval_id}"
        )
        axis.plot(
            fused[mask, 0], fused[mask, 1], linestyle="--", linewidth=1.7,
            label=f"Twin {interval_id}"
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World y (m)")
    axis.set_title("UGV01 prediction-only fidelity pilot")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory_fidelity.png", dpi=180)
    plt.close(figure)

    diagnostic, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    path_axis, error_axis, heading_axis, distance_axis = axes.ravel()

    path_axis.plot(
        gt_xy[:, 0], gt_xy[:, 1], color="#16697a", linewidth=2.2, label="AprilTag"
    )
    path_axis.plot(
        fused[:, 0], fused[:, 1], color="#d95d39", linewidth=1.8,
        linestyle="--", label="Digital twin"
    )
    recovered_mask = tracking_status != "decoded"
    path_axis.scatter(
        gt_xy[recovered_mask, 0], gt_xy[recovered_mask, 1], s=5,
        color="#7b2cbf", alpha=0.35, label="Recovered tag pose"
    )
    path_axis.scatter(*gt_xy[0], color="#2a9d8f", s=45, marker="o", label="Start")
    path_axis.scatter(*gt_xy[-1], color="#111111", s=50, marker="x", label="End")
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("World x (m)")
    path_axis.set_ylabel("World y (m)")
    path_axis.set_title("Trajectory shape")
    path_axis.legend(fontsize=8)
    path_axis.grid(alpha=0.25)

    error_axis.plot(gt_time, fused_error, color="#d95d39", linewidth=1.5)
    for threshold, color in ((0.10, "#2a9d8f"), (0.25, "#e9c46a"), (0.50, "#9c6644")):
        error_axis.axhline(
            threshold, color=color, linestyle=":", linewidth=1.2,
            label=f"{threshold:.2f} m"
        )
    error_axis.set_xlabel("Video time (s)")
    error_axis.set_ylabel("Position error (m)")
    error_axis.set_title("Absolute position error")
    error_axis.legend(fontsize=8, ncol=3)
    error_axis.grid(alpha=0.25)

    truth_heading_deg = np.degrees(
        np.asarray([wrap_angle(float(value)) for value in gt_heading])
    )
    twin_heading_deg = np.degrees(
        np.asarray([wrap_angle(float(value)) for value in physical_estimated_heading])
    )
    heading_axis.plot(
        gt_time, truth_heading_deg, color="#16697a", linewidth=1.5,
        label="AprilTag heading"
    )
    heading_axis.plot(
        gt_time, twin_heading_deg, color="#d95d39", linewidth=1.3,
        linestyle="--", label="Twin heading"
    )
    heading_axis.set_xlabel("Video time (s)")
    heading_axis.set_ylabel("Heading (deg, wrapped)")
    heading_axis.set_ylim(-185, 185)
    heading_axis.set_title("Heading agreement")
    heading_axis.legend(fontsize=8)
    heading_axis.grid(alpha=0.25)

    truth_cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1))]
    )
    twin_cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(fused[:, :2], axis=0), axis=1))]
    )
    distance_axis.plot(
        gt_time, truth_cumulative, color="#16697a", linewidth=1.8,
        label="AprilTag distance"
    )
    distance_axis.plot(
        gt_time, twin_cumulative, color="#d95d39", linewidth=1.5,
        linestyle="--", label="Twin distance"
    )
    distance_axis.set_xlabel("Video time (s)")
    distance_axis.set_ylabel("Cumulative distance (m)")
    distance_axis.set_title("Distance accumulation")
    distance_axis.legend(fontsize=8)
    distance_axis.grid(alpha=0.25)

    diagnostic.suptitle(
        "UGV01 AprilTag digital-twin fidelity\n"
        f"ATE RMSE {summary['position_ate_rmse_m']:.3f} m | "
        f"1 s RPE {summary['rpe_1s_rmse_m']:.3f} m | "
        f"heading MAE {summary['heading_mae_deg']:.1f} deg",
        fontsize=14,
    )
    diagnostic.tight_layout()
    diagnostic.savefig(output_dir / "fidelity_diagnostics.png", dpi=180)
    diagnostic.savefig(output_dir / "fidelity_diagnostics.pdf")
    plt.close(diagnostic)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--interval",
        action="append",
        default=[],
        metavar="START:END",
        help="Video interval in seconds; repeat for multiple windows.",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("onset", "activity"),
        default="onset",
        help="Use activity for videos that begin after telemetry or during motion.",
    )
    parser.add_argument(
        "--effective-track-width-m",
        type=float,
        default=UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M,
    )
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--clockwise-track-width-m", type=float)
    parser.add_argument("--counterclockwise-track-width-m", type=float)
    parser.add_argument(
        "--gyro-weight",
        type=float,
        default=DEFAULT_MOTION_FUSION_POLICY.gyro_weight,
    )
    parser.add_argument("--gyro-scale", type=float, default=1.0)
    args = parser.parse_args()
    intervals = DEFAULT_INTERVALS
    if args.interval:
        parsed = []
        for value in args.interval:
            start_text, end_text = value.split(":", maxsplit=1)
            start, end = float(start_text), float(end_text)
            if start < 0 or end <= start:
                raise ValueError(f"invalid interval {value!r}")
            parsed.append((start, end))
        intervals = tuple(parsed)
    summary = analyze(
        args.tracking,
        args.telemetry,
        args.output_dir,
        intervals=intervals,
        sync_mode=args.sync_mode,
        effective_track_width_m=args.effective_track_width_m,
        distance_scale=args.distance_scale,
        clockwise_track_width_m=args.clockwise_track_width_m,
        counterclockwise_track_width_m=args.counterclockwise_track_width_m,
        gyro_weight=args.gyro_weight,
        gyro_scale=args.gyro_scale,
    )
    print(args.output_dir / "fidelity_report.md")
    print(f"ATE_RMSE={summary['position_ate_rmse_m']:.3f} m")
    print(f"RPE_1S_RMSE={summary['rpe_1s_rmse_m']:.3f} m")
    print(f"HEADING_MAE={summary['heading_mae_deg']:.1f} deg")


if __name__ == "__main__":
    main()

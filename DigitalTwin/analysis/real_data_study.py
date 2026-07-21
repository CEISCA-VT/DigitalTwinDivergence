"""Run the reproducible UGV01 benign-replay and semantic-GPS attack study.

The raw rover logs remain immutable. Attacks are injected only into the GPS
coordinates passed to the digital twin, matching the primary A1 threat model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from DigitalTwin.alarm import (
    AlarmConfig,
    PersistentAlarm,
    motion_start_index,
    operational_run_statistic,
    robust_initial_state,
)
from DigitalTwin.detector import InnovationDetector
from DigitalTwin.ekf import RoverEKF
from DigitalTwin.kinematics import DifferentialDriveGeometry, wrap_angle
from DigitalTwin.latency import LatencyQueue
from DigitalTwin.telemetry import gps_to_local_xy
from DigitalTwin.uncertainty import (
    DEFAULT_EVIDENCE_GATE_POLICY,
    FixedUncertaintyEstimator,
    GPSIndependentUncertaintyEstimator,
    NaiveAdaptiveUncertaintyEstimator,
    TelemetryStatisticsWindow,
)

from .common import parse_bool, parse_float, parse_int, parse_run_name, quantile, read_rows, write_rows


VARIANTS = ("fixed", "naive_adaptive", "frozen_clean", "gps_independent", "evidence_gated")
STEP_MAGNITUDES_M = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)
DRIFT_RATES_MPS = (0.01, 0.03, 0.05)
ATTACK_START_FRACTIONS = (0.25, 0.50, 0.70)
EPSILON_TARGETS = (0.50, 0.90, 0.95)
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260721
MISSION_TOLERANCE_M = 5.0
TARGET_RUN_FALSE_ALARM = 0.05
STANDARD_GRAVITY_MPS2 = 9.80665
ALARM_CONFIG = AlarmConfig()


@dataclass(frozen=True, slots=True)
class AttackSpec:
    kind: str = "none"
    direction: str = "none"
    magnitude_m: float = 0.0
    rate_mps: float = 0.0
    replay_delay_s: float = 5.0
    start_fraction: float = 0.30

    @property
    def label(self) -> str:
        if self.kind == "step":
            return f"step_{self.direction}_{self.magnitude_m:g}m"
        if self.kind in {"drift", "strategic_drift"}:
            return f"{self.kind}_{self.direction}_{self.rate_mps:g}mps"
        if self.kind == "replay":
            return f"replay_{self.replay_delay_s:g}s"
        return self.kind


@dataclass(slots=True)
class ReplayResult:
    elapsed_s: np.ndarray
    scores: np.ndarray
    detected: np.ndarray
    states_xy: np.ndarray
    clean_gps_xy: np.ndarray
    attacked_gps_xy: np.ndarray
    q_trace: np.ndarray
    s_trace: np.ndarray
    q_matrices: list[np.ndarray]
    r_matrices: list[np.ndarray]
    active: np.ndarray
    alarm_enabled: np.ndarray
    rows: list[dict[str, object]]


@dataclass(slots=True)
class PreparedRun:
    rows: list[dict[str, str]]
    elapsed_s: np.ndarray
    headings: np.ndarray
    controls: np.ndarray
    clean_gps_xy: np.ndarray
    uncertainty_proxy: np.ndarray
    baseline_arrivals_s: np.ndarray
    mission_start_index: int


MANIFEST_FIELDS = [
    "run_id",
    "speed",
    "surface",
    "network",
    "route",
    "trial",
    "split",
    "rows",
    "gps_valid_rows",
    "sequence_gaps",
    "stale_packets",
    "request_failures",
    "source_csv",
    "sha256",
]


def _successful_gps_rows(path: Path) -> list[dict[str, str]]:
    rows = read_rows(path)
    return [
        row
        for row in rows
        if parse_bool(row.get("cycle_ok", "True"))
        and parse_bool(row.get("gps_valid", "False"))
        and parse_float(row.get("lat", ""), None) is not None
        and parse_float(row.get("lon", ""), None) is not None
    ]


def _counter_max(rows: Iterable[dict[str, str]], key: str) -> int:
    return max((parse_int(row.get(key, ""), 0) or 0 for row in rows), default=0)


def _quality(path: Path) -> dict[str, object]:
    rows = read_rows(path)
    valid = [
        row
        for row in rows
        if parse_bool(row.get("cycle_ok", "True")) and parse_bool(row.get("gps_valid", "False"))
    ]
    gaps = _counter_max(rows, "sequence_gap_count")
    stale = sum(parse_bool(row.get("stale_packet", "False")) for row in rows)
    failures = _counter_max(rows, "request_failure_count")
    accepted = len(rows) >= 120 and len(valid) == len(rows) and gaps == 0 and stale == 0 and failures == 0
    return {
        "rows": len(rows),
        "gps_valid_rows": len(valid),
        "sequence_gaps": gaps,
        "stale_packets": stale,
        "request_failures": failures,
        "quality_pass": accepted,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_benign_manifest(input_dir: Path, out_dir: Path) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    pattern = "speed-*_surface-*_latency-wifi_baseline_route-square0p5x3_attack-none_trial-*.csv"
    for path in sorted(input_dir.glob(pattern)):
        meta = parse_run_name(path)
        trial = parse_int(meta.get("trial", ""), None)
        if meta.get("speed") not in {"low", "medium"} or trial not in {1, 2, 3, 4, 5}:
            continue
        quality = _quality(path)
        candidate = {"path": path, "meta": meta, **quality}
        candidates.append(candidate)
        grouped[(meta["speed"], meta["surface"], int(trial))].append(candidate)

    audit_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for key, choices in sorted(grouped.items()):
        passing = [choice for choice in choices if choice["quality_pass"]]
        selected = max(passing, key=lambda choice: (int(choice["rows"]), str(choice["path"]))) if passing else None
        for choice in choices:
            audit_rows.append(
                {
                    "speed": key[0],
                    "surface": key[1],
                    "trial": key[2],
                    "selected": choice is selected,
                    "quality_pass": choice["quality_pass"],
                    "rows": choice["rows"],
                    "gps_valid_rows": choice["gps_valid_rows"],
                    "sequence_gaps": choice["sequence_gaps"],
                    "stale_packets": choice["stale_packets"],
                    "request_failures": choice["request_failures"],
                    "source_csv": str(choice["path"]),
                }
            )
        if selected is None:
            continue
        speed, surface, trial_id = key
        split = "development" if trial_id <= 3 else "validation" if trial_id == 4 else "test"
        path = Path(selected["path"])
        manifest.append(
            {
                "run_id": f"{speed}_{surface}_trial-{trial_id}",
                "speed": speed,
                "surface": surface,
                "network": "wifi_baseline",
                "route": "square0p5x3",
                "trial": trial_id,
                "split": split,
                "rows": selected["rows"],
                "gps_valid_rows": selected["gps_valid_rows"],
                "sequence_gaps": selected["sequence_gaps"],
                "stale_packets": selected["stale_packets"],
                "request_failures": selected["request_failures"],
                "source_csv": str(path),
                "sha256": _sha256(path),
            }
        )

    expected = {(speed, surface, trial) for speed in ("low", "medium") for surface in (
        "rough_permeable_concrete", "smooth_kitchen_floor"
    ) for trial in range(1, 6)}
    actual = {(str(row["speed"]), str(row["surface"]), int(row["trial"])) for row in manifest}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"benign matrix is incomplete; missing accepted runs: {missing}")

    write_rows(out_dir / "benign_candidate_audit.csv", audit_rows, audit_rows[0].keys())
    write_rows(out_dir / "benign_manifest.csv", manifest, MANIFEST_FIELDS)
    split_payload = {
        "policy": "stratified by speed and surface; trials 1-3 development, 4 validation, 5 held-out test",
        "development": [row["run_id"] for row in manifest if row["split"] == "development"],
        "validation": [row["run_id"] for row in manifest if row["split"] == "validation"],
        "test": [row["run_id"] for row in manifest if row["split"] == "test"],
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    return manifest


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    return float(parse_float(row.get(key, ""), default) or default)


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(parse_int(row.get(key, ""), default) or default)


def _sample_time_s(row: dict[str, str]) -> float:
    if parse_float(row.get("sample_ms", ""), None) is not None:
        return _f(row, "sample_ms") / 1000.0
    if parse_float(row.get("rover_millis_s", ""), None) is not None:
        return _f(row, "rover_millis_s")
    return _f(row, "source_sample_time_s")


def _attack_specs() -> list[AttackSpec]:
    specs = [AttackSpec()]
    for direction in ("along", "cross"):
        specs.extend(AttackSpec("step", direction, magnitude_m=value) for value in STEP_MAGNITUDES_M)
    specs.extend([AttackSpec("freeze"), AttackSpec("replay", replay_delay_s=5.0)])
    for direction in ("along", "cross"):
        specs.extend(AttackSpec("drift", direction, rate_mps=value) for value in DRIFT_RATES_MPS)
    for direction in ("along", "cross"):
        specs.append(AttackSpec("strategic_drift", direction, rate_mps=0.03))
    return specs


def _estimator(mode: str):
    if mode == "fixed":
        return FixedUncertaintyEstimator()
    if mode == "gps_independent":
        return GPSIndependentUncertaintyEstimator()
    return NaiveAdaptiveUncertaintyEstimator()


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(
        probability * (1.0 - probability) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _precompute_motion(
    rows: list[dict[str, str]], geometry: DifferentialDriveGeometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([_sample_time_s(row) for row in rows], dtype=float)
    elapsed = times - times[0]
    headings = np.zeros(len(rows), dtype=float)
    controls = np.zeros((len(rows), 2), dtype=float)
    prev_left = _i(rows[0], "enc_left")
    prev_right = _i(rows[0], "enc_right")
    for index, row in enumerate(rows):
        dt_s = 0.1 if index == 0 else max(times[index] - times[index - 1], 1e-3)
        left = _i(row, "enc_left")
        right = _i(row, "enc_right")
        controls[index] = geometry.ticks_to_control(left - prev_left, right - prev_right, dt_s)
        prev_left, prev_right = left, right
        if index:
            headings[index] = wrap_angle(headings[index - 1] + controls[index, 1] * dt_s)
    return elapsed, headings, controls


def _buffered_delivery_times(elapsed: np.ndarray, seed: int) -> np.ndarray:
    queue = LatencyQueue(200.0, jitter_ms=40.0, buffered=True, seed=seed)
    for index, source_time in enumerate(elapsed):
        queue.push(float(source_time), index)
    deliveries = queue.pop_ready(float("inf"))
    result = np.zeros(len(elapsed), dtype=float)
    for delivery in deliveries:
        result[int(delivery.item)] = delivery.delivery_s
    return result


def _attack_measurements(
    clean_xy: np.ndarray,
    elapsed: np.ndarray,
    headings: np.ndarray,
    spec: AttackSpec,
    uncertainty_proxy: np.ndarray,
    earliest_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    attacked = clean_xy.copy()
    active = np.zeros(len(clean_xy), dtype=bool)
    if spec.kind == "none" or len(clean_xy) == 0:
        return attacked, active

    earliest_index = min(max(int(earliest_index), 0), len(elapsed) - 1)
    start_s = float(elapsed[earliest_index]) + spec.start_fraction * float(
        elapsed[-1] - elapsed[earliest_index]
    )
    start_index = int(np.searchsorted(elapsed, start_s, side="left"))
    active[start_index:] = True
    theta = float(headings[min(start_index, len(headings) - 1)])
    along = np.array([math.cos(theta), math.sin(theta)])
    direction = along if spec.direction == "along" else np.array([-along[1], along[0]])

    if spec.kind == "step":
        attacked[start_index:] += spec.magnitude_m * direction
    elif spec.kind == "freeze":
        attacked[start_index:] = clean_xy[start_index]
    elif spec.kind == "replay":
        for index in range(start_index, len(clean_xy)):
            source_time = elapsed[index] - spec.replay_delay_s
            source_index = max(0, int(np.searchsorted(elapsed, source_time, side="right") - 1))
            attacked[index] = clean_xy[source_index]
    elif spec.kind == "drift":
        ages = np.maximum(0.0, elapsed - elapsed[start_index])
        attacked += ages[:, None] * spec.rate_mps * direction
    elif spec.kind == "strategic_drift":
        threshold = float(np.quantile(uncertainty_proxy[start_index:], 0.75))
        offset = 0.0
        for index in range(start_index, len(clean_xy)):
            dt_s = 0.0 if index == 0 else max(0.0, elapsed[index] - elapsed[index - 1])
            if uncertainty_proxy[index] >= threshold:
                offset += spec.rate_mps * dt_s
            attacked[index] += offset * direction
    else:
        raise ValueError(f"unsupported attack kind {spec.kind!r}")
    return attacked, active


def _prepare_run(path: Path) -> PreparedRun:
    rows = _successful_gps_rows(path)
    if not rows:
        raise RuntimeError(f"{path} has no successful GPS-valid rows")
    geometry = DifferentialDriveGeometry()
    elapsed, headings, controls = _precompute_motion(rows, geometry)
    origin_lat = _f(rows[0], "lat")
    origin_lon = _f(rows[0], "lon")
    clean_xy = np.asarray(
        [gps_to_local_xy(_f(row, "lat"), _f(row, "lon"), origin_lat, origin_lon) for row in rows],
        dtype=float,
    )
    uncertainty_proxy = np.asarray(
        [
            abs(math.radians(_f(row, "gz")))
            + 0.2 * abs(
                _f(row, "az") * STANDARD_GRAVITY_MPS2 / 1000.0
                - STANDARD_GRAVITY_MPS2
            )
            + 2.0 * _f(row, "http_latency_ms") / 1000.0
            for row in rows
        ]
    )
    baseline_arrivals = np.asarray(
        [_f(row, "edge_arrival_time_s", _f(row, "t_edge_rx_ns") / 1e9) for row in rows],
        dtype=float,
    )
    return PreparedRun(
        rows=rows,
        elapsed_s=elapsed,
        headings=headings,
        controls=controls,
        clean_gps_xy=clean_xy,
        uncertainty_proxy=uncertainty_proxy,
        baseline_arrivals_s=baseline_arrivals,
        mission_start_index=motion_start_index(controls, ALARM_CONFIG),
    )


def replay(
    path: Path,
    mode: str,
    attack: AttackSpec,
    *,
    threshold: float | None = None,
    frozen_schedule: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
    transport: str = "baseline",
    prepared: PreparedRun | None = None,
) -> ReplayResult:
    prepared = prepared or _prepare_run(path)
    rows = prepared.rows
    elapsed = prepared.elapsed_s
    headings = prepared.headings
    controls = prepared.controls
    if transport not in {"baseline", "buffered_200ms_jitter40ms"}:
        raise ValueError(f"unsupported transport profile {transport!r}")
    seed = int(hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8], 16)
    buffered_arrivals = _buffered_delivery_times(elapsed, seed) if transport != "baseline" else None
    clean_xy = prepared.clean_gps_xy
    uncertainty_proxy = prepared.uncertainty_proxy
    mission_start = prepared.mission_start_index
    attacked_xy, active = _attack_measurements(
        clean_xy,
        elapsed,
        headings,
        attack,
        uncertainty_proxy,
        earliest_index=mission_start,
    )
    alarm_enabled = np.arange(len(rows)) >= mission_start
    initial_state, initial_covariance = robust_initial_state(attacked_xy, mission_start, ALARM_CONFIG)

    estimator = _estimator(mode)
    stats = TelemetryStatisticsWindow()
    detector = InnovationDetector(threshold=threshold)
    alarm = PersistentAlarm(detector.threshold, ALARM_CONFIG)
    ekf = RoverEKF()
    scores: list[float] = []
    detections: list[bool] = []
    states: list[np.ndarray] = []
    q_trace: list[float] = []
    s_trace: list[float] = []
    q_matrices: list[np.ndarray] = []
    r_matrices: list[np.ndarray] = []
    output_rows: list[dict[str, object]] = []
    last_residual = 0.0
    previous_score = 0.0
    previous_arrival: float | None = None

    for index, row in enumerate(rows):
        dt_s = 0.1 if index == 0 else max(elapsed[index] - elapsed[index - 1], 1e-3)
        arrival = (
            float(buffered_arrivals[index])
            if buffered_arrivals is not None
            else float(prepared.baseline_arrivals_s[index])
        )
        arrival_dt = dt_s if previous_arrival is None else max(arrival - previous_arrival, 1e-3)
        previous_arrival = arrival
        accel_z_mps2 = _f(row, "az") * STANDARD_GRAVITY_MPS2 / 1000.0
        yaw_rate_radps = math.radians(_f(row, "gz"))
        independent_evidence = (
            abs(accel_z_mps2 - STANDARD_GRAVITY_MPS2)
            >= DEFAULT_EVIDENCE_GATE_POLICY.acceleration_deviation_mps2
            or abs(yaw_rate_radps) >= DEFAULT_EVIDENCE_GATE_POLICY.yaw_rate_radps
            or abs(arrival_dt - dt_s) >= DEFAULT_EVIDENCE_GATE_POLICY.timing_mismatch_s
        )
        evidence_allowed = (
            previous_score <= detector.threshold
            and independent_evidence
            and (
                not DEFAULT_EVIDENCE_GATE_POLICY.reject_stale_packets
                or not parse_bool(row.get("stale_packet", "False"))
            )
        )
        observed_residual = last_residual
        if mode == "gps_independent" or (mode == "evidence_gated" and not evidence_allowed):
            observed_residual = 0.0
        residual_feedback_active = mode == "naive_adaptive" or (
            mode == "evidence_gated" and evidence_allowed
        )
        stats.observe(
            dead_reckoning_residual_m=observed_residual,
            # T:147 reports acceleration in mg and gyro rate in deg/s.
            accel_z=accel_z_mps2,
            gyro_z=yaw_rate_radps,
            velocity_mps=float(controls[index, 0]),
            packet_dt_s=arrival_dt,
        )
        features = stats.features(
            gps_hdop=_f(row, "hdop", 99.99),
            gps_satellites=_i(row, "sat"),
            fallback_dt_s=arrival_dt,
        )
        if mode == "frozen_clean" and frozen_schedule is not None:
            Q = frozen_schedule[0][index]
            R = frozen_schedule[1][index]
        else:
            Q = estimator.process_covariance(features, dt_s)
            R = estimator.measurement_covariance(features)
        if index == mission_start:
            ekf = RoverEKF(initial_state=initial_state, initial_covariance=initial_covariance)
        else:
            ekf.predict(float(controls[index, 0]), float(controls[index, 1]), dt_s, Q)
        last_residual = float(np.linalg.norm(attacked_xy[index] - ekf.state.x[:2]))
        ekf.update_gps(attacked_xy[index], R)
        detection = detector.evaluate(ekf.last_innovation, ekf.last_S)
        alarm_detected = alarm.observe(detection.mahalanobis, enabled=bool(alarm_enabled[index]))
        previous_score = detection.mahalanobis
        scores.append(detection.mahalanobis)
        detections.append(alarm_detected)
        states.append(ekf.state.x[:2].copy())
        q_trace.append(float(np.trace(Q)))
        s_trace.append(float(np.trace(ekf.last_S)))
        q_matrices.append(Q.copy())
        r_matrices.append(R.copy())
        output_rows.append(
            {
                "elapsed_s": elapsed[index],
                "seq": _i(row, "seq", index),
                "clean_gps_x_m": clean_xy[index, 0],
                "clean_gps_y_m": clean_xy[index, 1],
                "attacked_gps_x_m": attacked_xy[index, 0],
                "attacked_gps_y_m": attacked_xy[index, 1],
                "ekf_x_m": ekf.state.x[0],
                "ekf_y_m": ekf.state.x[1],
                "mahalanobis": detection.mahalanobis,
                "threshold": detection.threshold,
                "instantaneous_exceedance": int(detection.detected),
                "alarm_enabled": int(alarm_enabled[index]),
                "detected": int(alarm_detected),
                "q_trace": q_trace[-1],
                "s_trace": s_trace[-1],
                "attack_active": int(active[index]),
                "attack_label": attack.label,
                "attack_start_fraction": attack.start_fraction if attack.kind != "none" else "",
                "detector_variant": mode,
                "transport": transport,
                "independent_evidence": int(independent_evidence),
                "residual_feedback_active": int(residual_feedback_active),
            }
        )

    return ReplayResult(
        elapsed_s=elapsed,
        scores=np.asarray(scores),
        detected=np.asarray(detections, dtype=bool),
        states_xy=np.asarray(states),
        clean_gps_xy=clean_xy,
        attacked_gps_xy=attacked_xy,
        q_trace=np.asarray(q_trace),
        s_trace=np.asarray(s_trace),
        q_matrices=q_matrices,
        r_matrices=r_matrices,
        active=active,
        alarm_enabled=alarm_enabled,
        rows=output_rows,
    )


def lock_thresholds(manifest: list[dict[str, object]], out_dir: Path) -> dict[str, dict[str, object]]:
    calibration = list(manifest)
    thresholds: dict[str, dict[str, object]] = {}
    for mode in VARIANTS:
        clean_mode = "naive_adaptive" if mode == "frozen_clean" else mode
        results = [replay(Path(row["source_csv"]), clean_mode, AttackSpec()) for row in calibration]
        run_statistics = [
            operational_run_statistic(result.scores, result.alarm_enabled, ALARM_CONFIG)
            for result in results
        ]
        all_scores = [
            float(value)
            for result in results
            for value in result.scores[result.alarm_enabled]
        ]
        locked = max(run_statistics)
        leave_one_out_alarms = 0
        for index, statistic in enumerate(run_statistics):
            fold_threshold = max(run_statistics[:index] + run_statistics[index + 1 :])
            leave_one_out_alarms += int(statistic > fold_threshold)
        pfa_low, pfa_high = _wilson_interval(leave_one_out_alarms, len(results))
        thresholds[mode] = {
            "threshold": locked,
            "target_run_false_alarm_rate": TARGET_RUN_FALSE_ALARM,
            "calibration_runs": len(results),
            "calibration_updates": len(all_scores),
            "calibration_run_false_alarms": sum(value > locked for value in run_statistics),
            "calibration_run_false_alarm_rate": 0.0,
            "empirical_run_rate_resolution": 1.0 / len(results),
            "zero_alarm_one_sided_95pct_upper_bound": 1.0 - 0.05 ** (1.0 / len(results)),
            "leave_one_run_out_false_alarms": leave_one_out_alarms,
            "leave_one_run_out_false_alarm_rate": leave_one_out_alarms / len(results),
            "leave_one_run_out_pfa_wilson95_low": pfa_low,
            "leave_one_run_out_pfa_wilson95_high": pfa_high,
            "per_update_p95": quantile(all_scores, 0.95),
            "per_update_p99": quantile(all_scores, 0.99),
            "alarm_window_size": ALARM_CONFIG.window_size,
            "alarm_required_exceedances": ALARM_CONFIG.required_exceedances,
            "initialization_gps_samples": ALARM_CONFIG.initialization_gps_samples,
            "policy": (
                "all-benign final freeze plus leave-one-run-out false-alarm "
                "estimate; 3-of-5 alarm after motion-gated robust initialization"
            ),
        }
    (out_dir / "locked_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    locked_policy = {
        "schema": "ugv01_alarm_policy_v1",
        "status": "frozen_offline",
        "threat_horizon": "after sustained tracked-drive motion begins",
        "gps_initialization_samples": ALARM_CONFIG.initialization_gps_samples,
        "motion_speed_threshold_mps": ALARM_CONFIG.motion_speed_threshold_mps,
        "motion_yaw_rate_threshold_radps": ALARM_CONFIG.motion_yaw_rate_threshold_radps,
        "motion_consecutive_updates": ALARM_CONFIG.motion_consecutive_updates,
        "alarm_window_size": ALARM_CONFIG.window_size,
        "required_exceedances": ALARM_CONFIG.required_exceedances,
        "target_run_false_alarm_rate": TARGET_RUN_FALSE_ALARM,
        "threshold_source": "all 20 checksum-identified benign runs",
        "validation_method": "leave-one-run-out at complete-run level",
        "thresholds": {mode: values["threshold"] for mode, values in thresholds.items()},
    }
    policy_path = Path("DigitalTwin/configs/locked_alarm_policy.json")
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(locked_policy, indent=2), encoding="utf-8")
    return thresholds


def _metrics(
    manifest_row: dict[str, object],
    mode: str,
    attack: AttackSpec,
    attacked: ReplayResult,
    clean: ReplayResult,
    threshold: float,
    transport: str,
) -> dict[str, object]:
    active_indices = np.flatnonzero(attacked.active & attacked.alarm_enabled)
    evaluation_indices = active_indices if attack.kind != "none" else np.flatnonzero(attacked.alarm_enabled)
    deviation = np.linalg.norm(attacked.states_xy - clean.states_xy, axis=1)
    attack_offset = np.linalg.norm(attacked.attacked_gps_xy - attacked.clean_gps_xy, axis=1)
    detected_indices = evaluation_indices[attacked.detected[evaluation_indices]]
    first_detection = int(detected_indices[0]) if len(detected_indices) else None
    if attack.kind != "none" and len(active_indices):
        pre_alarm = active_indices if first_detection is None else active_indices[active_indices < first_detection]
        attack_start_s = float(attacked.elapsed_s[active_indices[0]])
    else:
        pre_alarm = np.array([], dtype=int)
        attack_start_s = float("nan")
    evaluation_horizon_s = (
        float(attacked.elapsed_s[active_indices[-1]] - attack_start_s)
        if len(active_indices) and attack.kind != "none"
        else ""
    )
    max_undetected = float(deviation[pre_alarm].max()) if len(pre_alarm) else 0.0
    time_above = 0.0
    for index in pre_alarm:
        if deviation[index] > MISSION_TOLERANCE_M and index > 0:
            time_above += max(0.0, float(attacked.elapsed_s[index] - attacked.elapsed_s[index - 1]))
    attack_window = active_indices if attack.kind != "none" else evaluation_indices
    mean_attack_q = float(attacked.q_trace[attack_window].mean()) if len(attack_window) else 0.0
    mean_clean_q = float(clean.q_trace[attack_window].mean()) if len(attack_window) else 0.0
    mean_attack_s = float(attacked.s_trace[attack_window].mean()) if len(attack_window) else 0.0
    mean_clean_s = float(clean.s_trace[attack_window].mean()) if len(attack_window) else 0.0
    mean_attack_nis = float(attacked.scores[attack_window].mean()) if len(attack_window) else 0.0
    mean_clean_nis = float(clean.scores[attack_window].mean()) if len(attack_window) else 0.0
    feedback = (
        [int(attacked.rows[index]["residual_feedback_active"]) for index in attack_window]
        if attacked.rows
        else [0] * len(attack_window)
    )
    clean_feedback = (
        [int(clean.rows[index]["residual_feedback_active"]) for index in attack_window]
        if clean.rows
        else [0] * len(attack_window)
    )
    independent_evidence = (
        [int(attacked.rows[index]["independent_evidence"]) for index in attack_window]
        if attacked.rows
        else [0] * len(attack_window)
    )
    feedback_fraction = sum(feedback) / len(feedback) if feedback else 0.0
    clean_feedback_fraction = sum(clean_feedback) / len(clean_feedback) if clean_feedback else 0.0
    return {
        **{field: manifest_row[field] for field in ("run_id", "speed", "surface", "trial", "split", "source_csv")},
        "detector_variant": mode,
        "transport": transport,
        "threshold": threshold,
        "attack": attack.kind,
        "attack_label": attack.label,
        "direction": attack.direction,
        "magnitude_m": attack.magnitude_m or "",
        "rate_mps": attack.rate_mps or "",
        "replay_delay_s": attack.replay_delay_s if attack.kind == "replay" else "",
        "attack_start_fraction": attack.start_fraction if attack.kind != "none" else "",
        "attack_start_s": attack_start_s if attack.kind != "none" else "",
        "evaluation_horizon_s": evaluation_horizon_s,
        "updates": len(attacked.scores),
        "attack_updates": len(active_indices),
        "run_detected": int(first_detection is not None),
        "detection_delay_s": (
            ""
            if first_detection is None or attack.kind == "none"
            else float(attacked.elapsed_s[first_detection] - attack_start_s)
        ),
        "max_nis": float(attacked.scores.max()),
        "mean_attack_window_q_trace": mean_attack_q,
        "mean_clean_window_q_trace": mean_clean_q,
        "attack_window_q_trace_ratio": mean_attack_q / max(mean_clean_q, 1e-15),
        "mean_attack_window_s_trace": mean_attack_s,
        "mean_clean_window_s_trace": mean_clean_s,
        "attack_window_s_trace_ratio": mean_attack_s / max(mean_clean_s, 1e-15),
        "mean_attack_window_nis": mean_attack_nis,
        "mean_clean_window_nis": mean_clean_nis,
        "attack_window_nis_ratio": mean_attack_nis / max(mean_clean_nis, 1e-15),
        "residual_feedback_activation_fraction": feedback_fraction,
        "clean_feedback_activation_fraction": clean_feedback_fraction,
        "attack_induced_feedback_activation_delta": feedback_fraction
        - clean_feedback_fraction,
        "independent_evidence_fraction": (
            sum(independent_evidence) / len(independent_evidence)
            if independent_evidence
            else 0.0
        ),
        "max_attack_offset_m": float(attack_offset.max()),
        "max_paired_state_deviation_m": float(deviation.max()),
        "max_undetected_state_deviation_m": max_undetected,
        "harmful_but_stealthy": int(max_undetected > MISSION_TOLERANCE_M),
        "time_above_5m_before_alarm_s": time_above,
        "mean_q_trace_ratio": float(attacked.q_trace.mean() / max(clean.q_trace.mean(), 1e-15)),
        "mean_s_trace_ratio": float(attacked.s_trace.mean() / max(clean.s_trace.mean(), 1e-15)),
    }


def run_campaign(
    manifest: list[dict[str, object]],
    thresholds: dict[str, dict[str, object]],
    out_dir: Path,
    *,
    attacks: list[AttackSpec] | None = None,
    start_fractions: tuple[float, ...] = ATTACK_START_FRACTIONS,
    include_buffered: bool = True,
    summary_filename: str = "campaign_summary.csv",
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    exemplar_written: set[tuple[str, str]] = set()
    for manifest_row in manifest:
        path = Path(manifest_row["source_csv"])
        prepared = _prepare_run(path)
        transports = ["baseline"]
        if include_buffered and manifest_row["split"] == "test":
            transports.append("buffered_200ms_jitter40ms")
        for transport in transports:
            clean_cache: dict[str, ReplayResult] = {}
            for mode in VARIANTS:
                clean_mode = "naive_adaptive" if mode == "frozen_clean" else mode
                threshold = float(thresholds[mode]["threshold"])
                clean_cache[mode] = replay(
                    path,
                    clean_mode,
                    AttackSpec(),
                    threshold=threshold,
                    transport=transport,
                    prepared=prepared,
                )
                summaries.append(
                    _metrics(
                        manifest_row, mode, AttackSpec(), clean_cache[mode], clean_cache[mode], threshold, transport
                    )
                )

            # Buffered transport is retained as a benign transfer check. The
            # statistical attack matrix is evaluated on all 20 baseline runs.
            if transport != "baseline":
                continue
            for base_attack in attacks or _attack_specs()[1:]:
                for start_fraction in start_fractions:
                    attack = replace(base_attack, start_fraction=start_fraction)
                    for mode in VARIANTS:
                        threshold = float(thresholds[mode]["threshold"])
                        schedule = None
                        if mode == "frozen_clean":
                            clean = clean_cache[mode]
                            schedule = (clean.q_matrices, clean.r_matrices)
                        attacked = replay(
                            path,
                            mode,
                            attack,
                            threshold=threshold,
                            frozen_schedule=schedule,
                            transport=transport,
                            prepared=prepared,
                        )
                        summaries.append(
                            _metrics(
                                manifest_row,
                                mode,
                                attack,
                                attacked,
                                clean_cache[mode],
                                threshold,
                                transport,
                            )
                        )
                        exemplar_key = (mode, attack.label)
                        if (
                            manifest_row["split"] == "test"
                            and math.isclose(start_fraction, 0.50)
                            and exemplar_key
                            in {
                                ("naive_adaptive", "drift_cross_0.03mps"),
                                ("evidence_gated", "drift_cross_0.03mps"),
                            }
                            and exemplar_key not in exemplar_written
                        ):
                            exemplar_path = out_dir / "exemplars" / f"{mode}_{attack.label}.csv"
                            write_rows(exemplar_path, attacked.rows, attacked.rows[0].keys())
                            exemplar_written.add(exemplar_key)
    write_rows(out_dir / summary_filename, summaries, summaries[0].keys())
    return summaries


def _stratified_run_clusters(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str], list[list[dict[str, object]]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(str(row["speed"]), str(row["surface"]))][str(row["run_id"])].append(row)
    return {
        stratum: [runs[run_id] for run_id in sorted(runs)]
        for stratum, runs in sorted(grouped.items())
    }


def _resample_run_clusters(
    clusters: dict[tuple[str, str], list[list[dict[str, object]]]],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    sampled: list[dict[str, object]] = []
    for stratum_clusters in clusters.values():
        indices = rng.integers(0, len(stratum_clusters), size=len(stratum_clusters))
        for index in indices:
            sampled.extend(stratum_clusters[int(index)])
    return sampled


def _bootstrap_interval(
    rows: list[dict[str, object]],
    statistic,
    *,
    iterations: int,
    seed: int,
) -> tuple[float | str, float | str]:
    clusters = _stratified_run_clusters(rows)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        value = statistic(_resample_run_clusters(clusters, rng))
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return "", ""
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _group_seed(key: tuple[object, ...]) -> int:
    digest = hashlib.sha256("|".join(map(str, key)).encode("utf-8")).hexdigest()
    return BOOTSTRAP_SEED ^ int(digest[:8], 16)


def _probability(rows: list[dict[str, object]], field: str) -> float:
    return sum(int(row[field]) for row in rows) / len(rows)


def _detected_delay_median(rows: list[dict[str, object]]) -> float | None:
    delays = [float(row["detection_delay_s"]) for row in rows if row["detection_delay_s"] != ""]
    return float(np.median(delays)) if delays else None


def aggregate_campaign(
    rows: list[dict[str, object]],
    out_dir: Path,
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["transport"], row["detector_variant"], row["attack"], row["direction"],
            row["magnitude_m"], row["rate_mps"], row["replay_delay_s"],
        )
        grouped[key].append(row)
    aggregates: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        numeric_delay = [float(row["detection_delay_s"]) for row in group if row["detection_delay_s"] != ""]
        seed = _group_seed(key)
        pd_low, pd_high = _bootstrap_interval(
            group,
            lambda sample: _probability(sample, "run_detected"),
            iterations=bootstrap_iterations,
            seed=seed,
        )
        harmful_low, harmful_high = _bootstrap_interval(
            group,
            lambda sample: _probability(sample, "harmful_but_stealthy"),
            iterations=bootstrap_iterations,
            seed=seed + 1,
        )
        delay_low, delay_high = _bootstrap_interval(
            group,
            _detected_delay_median,
            iterations=bootstrap_iterations,
            seed=seed + 2,
        )
        fractions = sorted(
            {float(row["attack_start_fraction"]) for row in group if row["attack_start_fraction"] != ""}
        )
        aggregates.append(
            {
                "scope": "all_accepted_runs" if key[0] == "baseline" else "recorded_test_runs",
                "transport": key[0],
                "detector_variant": key[1],
                "attack": key[2],
                "direction": key[3],
                "magnitude_m": key[4],
                "rate_mps": key[5],
                "replay_delay_s": key[6],
                "physical_runs": len({str(row["run_id"]) for row in group}),
                "scenarios": len(group),
                "attack_start_fractions": ";".join(f"{value:.2f}" for value in fractions),
                "detected_scenarios": sum(int(row["run_detected"]) for row in group),
                "run_detection_probability": _probability(group, "run_detected"),
                "run_detection_probability_ci95_low": pd_low,
                "run_detection_probability_ci95_high": pd_high,
                "harmful_but_stealthy_scenarios": sum(
                    int(row["harmful_but_stealthy"]) for row in group
                ),
                "harmful_but_stealthy_probability": _probability(group, "harmful_but_stealthy"),
                "harmful_but_stealthy_probability_ci95_low": harmful_low,
                "harmful_but_stealthy_probability_ci95_high": harmful_high,
                "detected_delay_samples": len(numeric_delay),
                "median_detection_delay_s": quantile(numeric_delay, 0.5) if numeric_delay else "",
                "median_detection_delay_ci95_low_s": delay_low,
                "median_detection_delay_ci95_high_s": delay_high,
                "p95_detection_delay_s": quantile(numeric_delay, 0.95) if numeric_delay else "",
                "detection_delay_censored_fraction": 1.0 - _probability(group, "run_detected"),
                "mean_evaluation_horizon_s": (
                    sum(float(row["evaluation_horizon_s"]) for row in group) / len(group)
                    if group[0]["evaluation_horizon_s"] != ""
                    else ""
                ),
                "mean_max_undetected_state_deviation_m": sum(
                    float(row["max_undetected_state_deviation_m"]) for row in group
                )
                / len(group),
                "max_undetected_state_deviation_m": max(
                    float(row["max_undetected_state_deviation_m"]) for row in group
                ),
                "mean_q_trace_ratio": sum(float(row["mean_q_trace_ratio"]) for row in group) / len(group),
                "mean_s_trace_ratio": sum(float(row["mean_s_trace_ratio"]) for row in group) / len(group),
            }
        )
    write_rows(out_dir / "campaign_aggregate.csv", aggregates, aggregates[0].keys())
    return aggregates


def _isotonic_non_decreasing(values: list[float], weights: list[int]) -> np.ndarray:
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(value), float(weight), float(index), float(index)])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[1] + right[1]
            mean = (left[0] * left[1] + right[0] * right[1]) / total_weight
            blocks.append([mean, total_weight, left[2], right[3]])
    fitted = np.zeros(len(values), dtype=float)
    for mean, _, start, end in blocks:
        fitted[int(start) : int(end) + 1] = mean
    return fitted


def _epsilon_at_probability(
    magnitudes: tuple[float, ...],
    probabilities: list[float],
    weights: list[int],
    target: float,
) -> tuple[float | None, str]:
    fitted = _isotonic_non_decreasing(probabilities, weights)
    if fitted[0] >= target:
        return float(magnitudes[0]), "at_or_below_minimum_tested"
    if fitted[-1] < target:
        return None, "above_maximum_tested"
    upper = int(np.flatnonzero(fitted >= target)[0])
    lower = upper - 1
    probability_span = fitted[upper] - fitted[lower]
    if probability_span <= 1e-12:
        return float(magnitudes[upper]), "within_tested_range"
    fraction = (target - fitted[lower]) / probability_span
    estimate = magnitudes[lower] + fraction * (magnitudes[upper] - magnitudes[lower])
    return float(estimate), "within_tested_range"


def _order_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(probability * (len(ordered) - 1)))))
    return float(ordered[index])


def estimate_epsilons(
    rows: list[dict[str, object]],
    out_dir: Path,
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> list[dict[str, object]]:
    step_rows = [
        row
        for row in rows
        if row["transport"] == "baseline" and row["attack"] == "step"
    ]
    summaries: list[dict[str, object]] = []
    for mode in VARIANTS:
        for direction in ("along", "cross"):
            group = [
                row
                for row in step_rows
                if row["detector_variant"] == mode and row["direction"] == direction
            ]
            by_magnitude = {
                magnitude: [row for row in group if float(row["magnitude_m"]) == magnitude]
                for magnitude in STEP_MAGNITUDES_M
            }
            probabilities = [
                _probability(by_magnitude[magnitude], "run_detected")
                for magnitude in STEP_MAGNITUDES_M
            ]
            weights = [len(by_magnitude[magnitude]) for magnitude in STEP_MAGNITUDES_M]
            clusters = _stratified_run_clusters(group)
            rng = np.random.default_rng(_group_seed((mode, direction, "epsilon")))
            bootstrap_values: dict[float, list[float]] = {target: [] for target in EPSILON_TARGETS}
            above_counts = {target: 0 for target in EPSILON_TARGETS}
            below_counts = {target: 0 for target in EPSILON_TARGETS}
            for _ in range(bootstrap_iterations):
                sample = _resample_run_clusters(clusters, rng)
                sample_probabilities: list[float] = []
                sample_weights: list[int] = []
                for magnitude in STEP_MAGNITUDES_M:
                    magnitude_rows = [
                        row for row in sample if float(row["magnitude_m"]) == magnitude
                    ]
                    sample_probabilities.append(_probability(magnitude_rows, "run_detected"))
                    sample_weights.append(len(magnitude_rows))
                for target in EPSILON_TARGETS:
                    value, status = _epsilon_at_probability(
                        STEP_MAGNITUDES_M,
                        sample_probabilities,
                        sample_weights,
                        target,
                    )
                    if value is None:
                        above_counts[target] += 1
                    else:
                        bootstrap_values[target].append(value)
                        if status == "at_or_below_minimum_tested":
                            below_counts[target] += 1
            for target in EPSILON_TARGETS:
                estimate, status = _epsilon_at_probability(
                    STEP_MAGNITUDES_M, probabilities, weights, target
                )
                finite = bootstrap_values[target]
                low = _order_quantile(finite, 0.025) if finite else ""
                upper_rank_is_censored = above_counts[target] / bootstrap_iterations > 0.025
                high = "" if upper_rank_is_censored or not finite else _order_quantile(finite, 0.975)
                summaries.append(
                    {
                        "detector_variant": mode,
                        "direction": direction,
                        "target_detection_probability": target,
                        "epsilon_estimate_m": estimate if estimate is not None else "",
                        "estimate_status": status,
                        "epsilon_ci95_low_m": low,
                        "epsilon_ci95_high_m": high,
                        "bootstrap_above_maximum_fraction": above_counts[target]
                        / bootstrap_iterations,
                        "bootstrap_at_or_below_minimum_fraction": below_counts[target]
                        / bootstrap_iterations,
                        "physical_runs": len({str(row["run_id"]) for row in group}),
                        "starts_per_run": len(ATTACK_START_FRACTIONS),
                        "tested_magnitudes_m": ";".join(map(str, STEP_MAGNITUDES_M)),
                    }
                )
    write_rows(out_dir / "epsilon_summary.csv", summaries, summaries[0].keys())
    return summaries


def validate_campaign(
    rows: list[dict[str, object]],
    manifest: list[dict[str, object]],
    out_dir: Path,
) -> dict[str, object]:
    baseline_attacks = [
        row for row in rows if row["transport"] == "baseline" and row["attack"] != "none"
    ]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in baseline_attacks:
        key = (
            row["detector_variant"],
            row["attack"],
            row["direction"],
            row["magnitude_m"],
            row["rate_mps"],
            row["replay_delay_s"],
        )
        grouped[key].append(row)
    expected_runs = len(manifest)
    expected_scenarios_per_condition = expected_runs * len(ATTACK_START_FRACTIONS)
    invalid_conditions = []
    for key, group in grouped.items():
        runs = {str(row["run_id"]) for row in group}
        fractions = {
            round(float(row["attack_start_fraction"]), 8) for row in group
        }
        if (
            len(group) != expected_scenarios_per_condition
            or len(runs) != expected_runs
            or fractions != set(ATTACK_START_FRACTIONS)
        ):
            invalid_conditions.append(list(key))
    expected_profiles = len(_attack_specs()) - 1
    expected_attack_scenarios = (
        expected_runs * expected_profiles * len(ATTACK_START_FRACTIONS) * len(VARIANTS)
    )
    payload = {
        "schema": "ugv01_attack_campaign_validation_v1",
        "status": "pass"
        if len(baseline_attacks) == expected_attack_scenarios and not invalid_conditions
        else "fail",
        "physical_runs": expected_runs,
        "detector_variants": len(VARIANTS),
        "attack_profiles": expected_profiles,
        "starts_per_run": len(ATTACK_START_FRACTIONS),
        "expected_attack_scenarios": expected_attack_scenarios,
        "observed_attack_scenarios": len(baseline_attacks),
        "expected_scenarios_per_condition": expected_scenarios_per_condition,
        "condition_groups": len(grouped),
        "invalid_condition_groups": invalid_conditions,
    }
    (out_dir / "campaign_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if payload["status"] != "pass":
        raise RuntimeError(f"attack campaign validation failed: {payload}")
    return payload


def _make_plots(
    rows: list[dict[str, object]],
    aggregates: list[dict[str, object]],
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_rows = [row for row in rows if row["transport"] == "baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, direction in zip(axes, ("along", "cross")):
        for mode in VARIANTS:
            values = []
            lower_errors = []
            upper_errors = []
            for magnitude in STEP_MAGNITUDES_M:
                aggregate = next(
                    row
                    for row in aggregates
                    if row["transport"] == "baseline"
                    and row["detector_variant"] == mode
                    and row["attack"] == "step"
                    and row["direction"] == direction
                    and float(row["magnitude_m"]) == magnitude
                )
                value = float(aggregate["run_detection_probability"])
                values.append(value)
                lower_errors.append(value - float(aggregate["run_detection_probability_ci95_low"]))
                upper_errors.append(float(aggregate["run_detection_probability_ci95_high"]) - value)
            axis.errorbar(
                STEP_MAGNITUDES_M,
                values,
                yerr=np.asarray([lower_errors, upper_errors]),
                marker="o",
                linewidth=1.4,
                capsize=2,
                label=mode.replace("_", " "),
            )
        axis.set_title(f"{direction.title()}-track step")
        axis.set_xlabel("Injected offset (m)")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Detection probability (run-clustered 95% CI)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Real-log step-attack detection: 20 runs, three start times")
    fig.tight_layout()
    fig.savefig(out_dir / "step_detection_probability.png", dpi=180)
    plt.close(fig)

    labels = []
    q_ratios = []
    stealth = []
    for mode in VARIANTS:
        group = [
            row
            for row in baseline_rows
            if row["detector_variant"] == mode
            and row["attack_label"] == "drift_cross_0.05mps"
        ]
        labels.append(mode.replace("_", "\n"))
        q_ratios.append(sum(float(row["mean_q_trace_ratio"]) for row in group) / len(group))
        stealth.append(sum(float(row["max_undetected_state_deviation_m"]) for row in group) / len(group))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, q_ratios, color="#3f6f8f")
    axes[0].axhline(1.0, color="black", linewidth=1)
    axes[0].set_ylabel("Attacked / clean mean trace(Q)")
    axes[0].set_title("Covariance response")
    axes[1].bar(labels, stealth, color="#a65244")
    axes[1].axhline(MISSION_TOLERANCE_M, color="black", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Mean max undetected state deviation (m)")
    axes[1].set_title("Paired state impact before alarm")
    fig.suptitle("Cross-track drift at 0.05 m/s (all accepted runs and starts)")
    fig.tight_layout()
    fig.savefig(out_dir / "covariance_poisoning.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for mode in VARIANTS:
        detection = []
        harmful = []
        for rate in DRIFT_RATES_MPS:
            aggregate = next(
                row
                for row in aggregates
                if row["transport"] == "baseline"
                and row["detector_variant"] == mode
                and row["attack"] == "drift"
                and row["direction"] == "cross"
                and float(row["rate_mps"]) == rate
            )
            detection.append(float(aggregate["run_detection_probability"]))
            harmful.append(float(aggregate["harmful_but_stealthy_probability"]))
        label = mode.replace("_", " ")
        axes[0].plot(DRIFT_RATES_MPS, detection, marker="o", label=label)
        axes[1].plot(DRIFT_RATES_MPS, harmful, marker="o", label=label)
    axes[0].set_title("Detection probability")
    axes[1].set_title("Harmful-but-stealthy probability")
    for axis in axes:
        axis.set_xlabel("Cross-track drift rate (m/s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylim(-0.005, 0.05)
    axes[1].set_ylim(-0.01, 0.20)
    axes[0].set_ylabel("Probability")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Slow-drift outcomes across all accepted runs")
    fig.tight_layout()
    fig.savefig(out_dir / "drift_attack_outcomes.png", dpi=180)
    plt.close(fig)


def _epsilon_text(row: dict[str, object]) -> str:
    estimate = row["epsilon_estimate_m"]
    if estimate == "":
        return f">{STEP_MAGNITUDES_M[-1]:g} m"
    prefix = "<=" if row["estimate_status"] == "at_or_below_minimum_tested" else ""
    low = row["epsilon_ci95_low_m"]
    high = row["epsilon_ci95_high_m"]
    if low == "" or high == "":
        interval = "censored"
    else:
        interval = f"{float(low):.2f}-{float(high):.2f}"
    return f"{prefix}{float(estimate):.2f} m [{interval}]"


def _render_report(
    manifest: list[dict[str, object]],
    thresholds: dict[str, dict[str, object]],
    rows: list[dict[str, object]],
    aggregates: list[dict[str, object]],
    epsilons: list[dict[str, object]],
) -> str:
    buffered_test = [
        row for row in rows if row["split"] == "test" and row["transport"] == "buffered_200ms_jitter40ms"
    ]
    attack_rows = [
        row for row in rows if row["transport"] == "baseline" and row["attack"] != "none"
    ]
    lines = [
        "# Complete Offline Statistical Attack Campaign",
        "",
        (
            "This report is generated from the canonical 20-run benign UGV01 "
            "dataset. GPS attacks are replay-injected; raw logs and rover "
            "behavior are unchanged."
        ),
        "",
        "## Dataset and split",
        "",
        f"- Accepted benign runs: {len(manifest)}. The earlier 12/4/4 split remains recorded for provenance.",
        (
            "- Because the previous trial-5 results were inspected during alarm "
            "diagnosis, all 20 benign runs now form the alarm-design corpus; final "
            "false-alarm performance uses leave-one-run-out evaluation."
        ),
        "- Conditions: smooth kitchen floor and rough permeable concrete, each at low and medium speed.",
        "- Route: 0.5 m square repeated three times under baseline Wi-Fi.",
        (
            "- Geometry: Waveshare motion-model values, 0.0523 m drive diameter, "
            "1092 counts/revolution, and 0.141 m model track width."
        ),
        (
            f"- Attack starts: {', '.join(f'{value:.0%}' for value in ATTACK_START_FRACTIONS)} "
            "of each post-motion run horizon."
        ),
        (
            f"- Baseline-transport attack scenarios: {len(attack_rows):,}, "
            f"clustered within {len(manifest)} physical runs."
        ),
        "",
        "## Benign-only threshold lock",
        "",
        "| Variant | Locked NIS threshold | Benign runs | LORO false alarms | P_FA (95% Wilson CI) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in VARIANTS:
        false_alarms = int(thresholds[mode]["leave_one_run_out_false_alarms"])
        rate = float(thresholds[mode]["leave_one_run_out_false_alarm_rate"])
        low = float(thresholds[mode]["leave_one_run_out_pfa_wilson95_low"])
        high = float(thresholds[mode]["leave_one_run_out_pfa_wilson95_high"])
        lines.append(
            f"| {mode} | {float(thresholds[mode]['threshold']):.4f} | "
            f"{thresholds[mode]['calibration_runs']} | "
            f"{false_alarms}/{thresholds[mode]['calibration_runs']} | "
            f"{rate:.3f} [{low:.3f}, {high:.3f}] |"
        )

    lines.extend([
        "",
        (
            "The operational policy uses robust pre-mission GPS initialization, "
            "enables monitoring at sustained tracked-drive motion, and alarms "
            "after 3 of 5 NIS updates exceed the variant threshold."
        ),
        (
            "For each leave-one-run-out fold, the threshold is computed from the "
            "other 19 complete benign runs. The final deployed threshold is then "
            "frozen from all 20 benign runs. No attack data are used."
        ),
        (
            "The retained anomalous run is the single leave-one-run-out false "
            "alarm, giving the target point estimate 1/20 = 0.05 rather than "
            "removing the run post hoc."
        ),
        "",
        "## Directional step detectability",
        "",
        (
            "Values are monotone isotonic estimates with run-clustered bootstrap "
            "95% intervals. A censored value means the requested probability was "
            "not reached reliably within the 0.5-10 m test grid."
        ),
        "",
        "| Variant | Direction | epsilon_50 | epsilon_90 | epsilon_95 |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for mode in VARIANTS:
        for direction in ("along", "cross"):
            values = {
                float(row["target_detection_probability"]): row
                for row in epsilons
                if row["detector_variant"] == mode and row["direction"] == direction
            }
            lines.append(
                f"| {mode} | {direction} | {_epsilon_text(values[0.50])} | "
                f"{_epsilon_text(values[0.90])} | {_epsilon_text(values[0.95])} |"
            )

    lines.extend([
        "",
        "## Representative slow-drift results",
        "",
        (
            "The complete CSV contains both directions and all three drift rates. "
            "This table shows the most severe preregistered 0.05 m/s cross-track "
            "case."
        ),
        "",
        "| Variant | P_D (95% CI) | Median detected delay (95% CI) | Harmful-but-stealthy P (95% CI) |",
        "| --- | ---: | ---: | ---: |",
    ])
    for mode in VARIANTS:
        aggregate = next(
            row
            for row in aggregates
            if row["transport"] == "baseline"
            and row["detector_variant"] == mode
            and row["attack"] == "drift"
            and row["direction"] == "cross"
            and float(row["rate_mps"]) == 0.05
        )
        delay = aggregate["median_detection_delay_s"]
        delay_text = "not detected" if delay in {"", None} else (
            f"{float(delay):.2f} s [{float(aggregate['median_detection_delay_ci95_low_s']):.2f}, "
            f"{float(aggregate['median_detection_delay_ci95_high_s']):.2f}]"
        )
        lines.append(
            f"| {mode} | {float(aggregate['run_detection_probability']):.3f} "
            f"[{float(aggregate['run_detection_probability_ci95_low']):.3f}, "
            f"{float(aggregate['run_detection_probability_ci95_high']):.3f}] | "
            f"{delay_text} | {float(aggregate['harmful_but_stealthy_probability']):.3f} "
            f"[{float(aggregate['harmful_but_stealthy_probability_ci95_low']):.3f}, "
            f"{float(aggregate['harmful_but_stealthy_probability_ci95_high']):.3f}] |"
        )

    lines.extend([
        "",
        "## Buffered transport check",
        "",
        (
            "A deterministic edge-side 200 ms buffered delay with 40 ms jitter "
            "was applied to the diagnostic trial-5 subset only; source packets "
            "and rover behavior were unchanged."
        ),
        "",
        "| Variant | Buffered benign run alarms |",
        "| --- | ---: |",
    ])
    for mode in VARIANTS:
        benign = [row for row in buffered_test if row["detector_variant"] == mode and row["attack"] == "none"]
        alarms = sum(int(row["run_detected"]) for row in benign)
        lines.append(f"| {mode} | {alarms}/{len(benign)} |")

    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- These are counterfactual replay results, not live attacks against the rover.",
        (
            "- State impact is measured against the paired clean replay of the "
            "same physical log; no overhead-video ground truth is available."
        ),
        (
            "- Attack intervals use stratified run-cluster bootstrap resampling; "
            "the three starts within a run are not treated as independent runs."
        ),
        (
            "- All 20 runs were available during alarm design, so these intervals "
            "describe the design corpus, not prospective generalization."
        ),
        (
            "- The leave-one-run-out false-alarm estimate uses 20 clustered runs "
            "and must be reported with its confidence interval."
        ),
        (
            "- Evidence gating is frozen in "
            "`DigitalTwin/configs/uncertainty_policies.json`; prospective runs are "
            "still required before supporting a strong causal claim."
        ),
        (
            "- The evidence gate admits GPS-residual-driven Q adaptation only "
            "when IMU or timing evidence independently indicates a disturbance."
        ),
        "",
        "## Generated artifacts",
        "",
        "- `benign_manifest.csv` and `split_manifest.json`: immutable run selection and split.",
        (
            "- `locked_thresholds.json` and "
            "`DigitalTwin/configs/locked_alarm_policy.json`: benign-only "
            "thresholds and frozen operational policy."
        ),
        "- `campaign_summary.csv` and `campaign_aggregate.csv`: per-run and grouped attack metrics.",
        "- `campaign_validation.json`: matrix-completeness certificate.",
        (
            "- `epsilon_summary.csv`: directional epsilon_50, epsilon_90, and "
            "epsilon_95 estimates and censoring information."
        ),
        "- `step_detection_probability.png`: direction-dependent step results with clustered intervals.",
        "- `drift_attack_outcomes.png`: drift detection and harmful-but-stealthy probabilities.",
        "- `covariance_poisoning.png`: covariance and paired-state response to slow drift.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="raw_logs/telemetry")
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis/real_data_study")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="regenerate plots/report from existing campaign CSV artifacts",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_benign_manifest(Path(args.input_dir), out_dir)
    if args.manifest_only:
        print(out_dir / "benign_manifest.csv")
        return
    if args.summarize_existing:
        thresholds = json.loads((out_dir / "locked_thresholds.json").read_text(encoding="utf-8"))
        campaign = read_rows(out_dir / "campaign_summary.csv")
        aggregates = read_rows(out_dir / "campaign_aggregate.csv")
        epsilons = read_rows(out_dir / "epsilon_summary.csv")
        validate_campaign(campaign, manifest, out_dir)
        _make_plots(campaign, aggregates, out_dir)
        (out_dir / "real_data_study_report.md").write_text(
            _render_report(manifest, thresholds, campaign, aggregates, epsilons),
            encoding="utf-8",
        )
        print(out_dir / "real_data_study_report.md")
        return
    thresholds = lock_thresholds(manifest, out_dir)
    campaign = run_campaign(manifest, thresholds, out_dir)
    aggregates = aggregate_campaign(
        campaign,
        out_dir,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    epsilons = estimate_epsilons(
        campaign,
        out_dir,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    validate_campaign(campaign, manifest, out_dir)
    _make_plots(campaign, aggregates, out_dir)
    (out_dir / "real_data_study_report.md").write_text(
        _render_report(manifest, thresholds, campaign, aggregates, epsilons),
        encoding="utf-8",
    )
    print(out_dir / "real_data_study_report.md")


if __name__ == "__main__":
    main()

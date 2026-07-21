import json
from pathlib import Path

import numpy as np

from DigitalTwin.analysis.common import parse_run_name
from DigitalTwin.analysis.real_data_study import (
    ATTACK_START_FRACTIONS,
    DRIFT_RATES_MPS,
    EPSILON_TARGETS,
    STEP_MAGNITUDES_M,
    AttackSpec,
    ReplayResult,
    _attack_measurements,
    _epsilon_at_probability,
    _isotonic_non_decreasing,
    _metrics,
    _wilson_interval,
)
from DigitalTwin.detector import InnovationDetector
from DigitalTwin.kinematics import DifferentialDriveGeometry


def test_hardware_run_name_parser_preserves_dataset_factors():
    path = Path(
        "speed-medium_surface-smooth_kitchen_floor_latency-wifi_baseline_"
        "route-square0p5x3_attack-none_trial-5_20260721_115709.csv"
    )
    parsed = parse_run_name(path)
    assert parsed["speed"] == "medium"
    assert parsed["surface"] == "smooth_kitchen_floor"
    assert parsed["route"] == "square0p5x3"
    assert parsed["trial"] == "5"


def test_ugv01_geometry_uses_locked_vendor_model_and_encoder_sign():
    geometry = DifferentialDriveGeometry()
    assert geometry.wheel_radius_m == 0.02615
    assert geometry.wheel_base_m == 0.141
    assert geometry.ticks_per_rev == 1092
    velocity, yaw_rate = geometry.ticks_to_control(-100, -100, 1.0)
    assert velocity > 0
    assert abs(yaw_rate) < 1e-12
    assert geometry.control_to_ticks(velocity, yaw_rate, 1.0) == (-100, -100)


def test_step_attack_starts_after_clean_prefix_and_has_exact_magnitude():
    clean = np.zeros((11, 2))
    elapsed = np.arange(11, dtype=float)
    headings = np.zeros(11)
    attacked, active = _attack_measurements(
        clean,
        elapsed,
        headings,
        AttackSpec("step", "cross", magnitude_m=2.0),
        np.zeros(11),
    )
    assert not active[:3].any()
    assert active[3:].all()
    assert np.allclose(attacked[:3], clean[:3])
    assert np.allclose(np.linalg.norm(attacked[3:] - clean[3:], axis=1), 2.0)


def test_attack_start_fraction_is_measured_after_motion_start():
    clean = np.zeros((11, 2))
    elapsed = np.arange(11, dtype=float)
    attacked, active = _attack_measurements(
        clean,
        elapsed,
        np.zeros(11),
        AttackSpec("step", "along", magnitude_m=1.0, start_fraction=0.5),
        np.zeros(11),
        earliest_index=4,
    )
    assert not active[:7].any()
    assert active[7:].all()
    assert np.allclose(attacked[7:, 0], 1.0)


def test_epsilon_estimate_uses_monotone_detection_curve():
    fitted = _isotonic_non_decreasing([0.1, 0.6, 0.5, 1.0], [10, 10, 10, 10])
    assert np.all(np.diff(fitted) >= 0)
    estimate, status = _epsilon_at_probability(
        (1.0, 2.0, 3.0, 4.0),
        [0.1, 0.6, 0.5, 1.0],
        [10, 10, 10, 10],
        0.9,
    )
    assert status == "within_tested_range"
    assert estimate is not None
    assert 3.0 < estimate < 4.0


def test_epsilon_estimate_reports_upper_censoring():
    estimate, status = _epsilon_at_probability(
        (1.0, 2.0, 3.0),
        [0.0, 0.2, 0.4],
        [10, 10, 10],
        0.9,
    )
    assert estimate is None
    assert status == "above_maximum_tested"


def test_frozen_attack_campaign_config_matches_code():
    payload = json.loads(
        Path("DigitalTwin/configs/attack_campaign.json").read_text(encoding="utf-8")
    )
    assert payload["attack_start_fractions_of_post_motion_horizon"] == list(
        ATTACK_START_FRACTIONS
    )
    assert payload["step_magnitudes_m"] == list(STEP_MAGNITUDES_M)
    assert payload["drift_rates_mps"] == list(DRIFT_RATES_MPS)
    assert payload["epsilon_detection_targets"] == list(EPSILON_TARGETS)


def test_detector_accepts_locked_empirical_threshold():
    detector = InnovationDetector(threshold=2.5)
    assert detector.threshold == 2.5


def test_benign_metrics_count_false_alarms_without_attack_window():
    result = ReplayResult(
        elapsed_s=np.array([0.0, 1.0]),
        scores=np.array([0.5, 3.0]),
        detected=np.array([False, True]),
        states_xy=np.zeros((2, 2)),
        clean_gps_xy=np.zeros((2, 2)),
        attacked_gps_xy=np.zeros((2, 2)),
        q_trace=np.ones(2),
        s_trace=np.ones(2),
        q_matrices=[np.eye(2), np.eye(2)],
        r_matrices=[np.eye(2), np.eye(2)],
        active=np.array([False, False]),
        alarm_enabled=np.array([True, True]),
        rows=[],
    )
    manifest = {
        "run_id": "r1",
        "speed": "low",
        "surface": "smooth",
        "trial": 1,
        "split": "test",
        "source_csv": "source.csv",
    }
    metrics = _metrics(manifest, "fixed", AttackSpec(), result, result, 2.5, "baseline")
    assert metrics["run_detected"] == 1
    assert metrics["detection_delay_s"] == ""


def test_wilson_interval_for_one_alarm_in_twenty_contains_point_estimate():
    low, high = _wilson_interval(1, 20)
    assert low < 0.05 < high
    assert round(low, 3) == 0.009
    assert round(high, 3) == 0.236

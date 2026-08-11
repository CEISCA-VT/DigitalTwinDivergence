import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from DigitalTwin.motion import (
    DEFAULT_MOTION_FUSION_POLICY,
    estimate_aligned_initial_heading,
    fuse_encoder_imu_motion,
)
from DigitalTwin.kinematics import (
    DifferentialDriveGeometry,
    UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M,
    ugv01_calibrated_geometry,
)


def test_motion_fusion_removes_stationary_gyro_bias():
    encoder = np.zeros((8, 2), dtype=float)
    encoder[4:, 1] = 1.0
    raw_gyro = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])

    policy = replace(DEFAULT_MOTION_FUSION_POLICY, gyro_weight=0.05)
    result = fuse_encoder_imu_motion(
        encoder, raw_gyro, mission_start_index=4, policy=policy
    )

    assert math.isclose(result.gyro_bias_radps, 0.1)
    assert np.allclose(result.corrected_gyro_radps[:4], 0.0)
    assert np.all(result.controls[4:, 1] < encoder[4:, 1])
    assert np.all(result.controls[4:, 1] > result.corrected_gyro_radps[4:])


def test_motion_fusion_reports_encoder_imu_disagreement_as_slip_indicator():
    encoder = np.array([[0.2, 0.0], [0.2, 1.0], [0.2, 1.0]], dtype=float)
    raw_gyro = np.array([0.0, 0.0, 0.0])

    result = fuse_encoder_imu_motion(encoder, raw_gyro, mission_start_index=1)

    assert result.yaw_disagreement_radps[1] == 1.0
    assert 0.9 < result.slip_indicator[1] <= 1.0


def test_initial_heading_alignment_recovers_rotated_straight_path():
    elapsed = np.arange(18, dtype=float)
    controls = np.zeros((18, 2), dtype=float)
    controls[1:, 0] = 0.1
    heading = math.radians(35.0)
    distances = np.arange(18, dtype=float) * 0.1
    gps = np.column_stack(
        [distances * math.cos(heading), distances * math.sin(heading)]
    )

    estimated, end = estimate_aligned_initial_heading(
        gps, controls, elapsed, start_index=0
    )

    assert end == DEFAULT_MOTION_FUSION_POLICY.initialization_updates
    assert math.isclose(estimated, heading, abs_tol=1e-9)


def test_frozen_motion_config_matches_code_defaults():
    payload = json.loads(
        Path("DigitalTwin/configs/motion_fusion.json").read_text(encoding="utf-8")
    )
    for key, value in asdict(DEFAULT_MOTION_FUSION_POLICY).items():
        assert payload[key] == value


def test_calibrated_geometry_preserves_nominal_width_separately():
    nominal = DifferentialDriveGeometry()
    calibrated = ugv01_calibrated_geometry()

    assert nominal.wheel_base_m == 0.141
    assert nominal.turn_width_m == 0.141
    assert calibrated.wheel_base_m == 0.141
    assert calibrated.turn_width_m == UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M
    assert calibrated.turn_width_m == 0.192

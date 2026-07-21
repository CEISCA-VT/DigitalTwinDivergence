from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from DigitalTwin.uncertainty import (
    DEFAULT_ADAPTIVE_POLICY,
    DEFAULT_EVIDENCE_GATE_POLICY,
    DEFAULT_FIXED_POLICY,
    FixedUncertaintyEstimator,
    GPSIndependentUncertaintyEstimator,
    NaiveAdaptiveUncertaintyEstimator,
    TelemetryDrivenUncertaintyEstimator,
    TelemetryStatisticsWindow,
)


def test_rolling_uncertainty_features_match_proposal_contract():
    window = TelemetryStatisticsWindow(maxlen=3)
    window.observe(
        dead_reckoning_residual_m=1.0,
        accel_z=9.8,
        gyro_z=0.1,
        velocity_mps=0.2,
        packet_dt_s=0.1,
    )
    window.observe(
        dead_reckoning_residual_m=2.0,
        accel_z=10.2,
        gyro_z=0.3,
        velocity_mps=0.4,
        packet_dt_s=0.2,
    )

    features = window.features(gps_hdop=1.2, gps_satellites=10, fallback_dt_s=0.1)
    assert features.dead_reckoning_residual_m == 1.5
    assert features.model_vector().shape == (5,)


def test_process_covariance_is_positive_diagonal():
    window = TelemetryStatisticsWindow()
    features = window.features(gps_hdop=1.2, gps_satellites=10, fallback_dt_s=0.1)
    Q = TelemetryDrivenUncertaintyEstimator().process_covariance(features, 0.1)
    assert Q.shape == (3, 3)
    assert Q[0, 0] > 0
    assert Q[1, 1] > 0
    assert Q[2, 2] > 0


def test_frozen_uncertainty_config_matches_code_defaults():
    payload = json.loads(Path("DigitalTwin/configs/uncertainty_policies.json").read_text(encoding="utf-8"))
    assert payload["fixed"] == asdict(DEFAULT_FIXED_POLICY)
    naive = payload["naive_adaptive"].copy()
    naive.pop("gps_coordinate_residual_in_process_features")
    assert naive == asdict(DEFAULT_ADAPTIVE_POLICY)
    evidence = payload["evidence_gated"]
    for key, value in asdict(DEFAULT_EVIDENCE_GATE_POLICY).items():
        assert evidence[key] == value
    learned = payload["learned_gps_independent"]
    assert learned["target_status"] == "frozen"
    assert learned["enabled_in_primary_campaign"] is False


def test_frozen_variants_have_distinct_residual_feedback_contracts():
    window = TelemetryStatisticsWindow()
    base = window.features(gps_hdop=1.2, gps_satellites=10, fallback_dt_s=0.1)
    attacked = replace(base, dead_reckoning_residual_m=10.0)

    fixed = FixedUncertaintyEstimator()
    independent = GPSIndependentUncertaintyEstimator()
    naive = NaiveAdaptiveUncertaintyEstimator()

    assert np.array_equal(fixed.process_covariance(base, 0.1), fixed.process_covariance(attacked, 0.1))
    assert np.array_equal(
        independent.process_covariance(base, 0.1),
        independent.process_covariance(attacked, 0.1),
    )
    assert naive.process_covariance(attacked, 0.1)[0, 0] > naive.process_covariance(base, 0.1)[0, 0]

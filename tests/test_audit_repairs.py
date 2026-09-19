import math

import numpy as np
import pandas as pd

from DigitalTwin.analysis.i2nav_fidelity_evaluator import evaluate_fidelity_frames
from DigitalTwin.analysis.i2nav_loso_ablation import build_fold_split as build_v1_fold_split
from DigitalTwin.analysis.i2nav_v2_full_loso import causal_backward_features, restricted_fold_split
from DigitalTwin.analysis.ugv01_fidelity_protocol import _derived_rate_metrics
from DigitalTwin.dashboard.server import TwinStream


def test_rpe_excludes_pairs_crossing_reference_gap():
    time = np.r_[np.arange(20) * .1, 22.0 + np.arange(20) * .1]
    frame = pd.DataFrame({
        "time_s": time,
        "gt_east_m": time,
        "gt_north_m": 0.0,
        "gt_heading_rad": 0.0,
        "estimate_east_m": time,
        "estimate_north_m": 0.0,
        "estimate_heading_rad": 0.0,
    })
    profile, series = evaluate_fidelity_frames(frame, horizons_s=(1.0,))
    assert profile["RPE_1s_pairs"] == 20
    assert profile["RPEp_1s_m"] == 0.0
    metrics = _derived_rate_metrics(frame, series)
    assert metrics["Dv_derived_RMSE_mps"] == 0.0
    assert np.nanmax(series["derived_Dv_mps"]) == 0.0


def test_restricted_fold_excludes_outer_sequence_everywhere():
    training, validation = restricted_fold_split("parking02", 2, "parking01")
    assert "parking02" not in training + validation
    assert "parking01" not in training + validation
    assert len(training) == 6
    assert len(validation) == 2
    v1_training, v1_validation = build_v1_fold_split("parking02", 2, "parking01")
    assert (v1_training, v1_validation) == (training, validation)


def test_causal_backward_features_do_not_use_next_sample():
    grid = np.arange(5, dtype=float) * 0.1
    speed = np.zeros(5)
    omega = np.zeros(5)
    before = causal_backward_features(speed, omega, grid)
    speed[-1] = 1.0
    omega[-1] = 2.0
    after = causal_backward_features(speed, omega, grid)
    np.testing.assert_allclose(after[:-1], before[:-1])
    assert after[-1, 2] == 10.0
    assert after[-1, 3] == 20.0


def test_live_stream_rotates_position_into_gps_frame_and_keeps_elapsed_time(tmp_path, monkeypatch):
    stream = TwinStream(mode="csv", csv_path=None, rover_url="http://127.0.0.1/js",
                        poll_hz=5.0, output_dir=tmp_path)
    monkeypatch.setattr(stream, "_write_log_record", lambda *_: None)
    common = {"gps_valid": True, "sat": 10, "hdop": 1.0,
              "lat": 37.0, "lon": -80.0, "gps_speed_mps": .5,
              "gps_course_deg": 0.0, "gps_age_ms": 50, "gz": 0.0}
    stream._append_sample({**common, "source_sample_time_s": 10.0,
                           "enc_left": 0, "enc_right": 0}, edge_arrival_s=1010.0, latency_ms=10)
    distance = 1000 * stream.geometry.meters_per_tick * stream.distance_scale
    stream._append_sample({**common, "source_sample_time_s": 11.0,
                           "lat": 37.0 + math.degrees(distance / 6371000.0),
                           "enc_left": -1000, "enc_right": -1000},
                          edge_arrival_s=1011.0, latency_ms=10)
    assert stream.points[-1]["common_frame_valid"]
    assert stream.points[-1]["gps_agreement_m"] < .01
    assert abs(stream.points[-1]["t"] - 1.0) < 1e-9

    stream._append_sample({**common, "source_sample_time_s": 16.0,
                           "enc_left": -1000, "enc_right": -1000},
                          edge_arrival_s=1016.0, latency_ms=10)
    assert stream.points[-1]["t"] == 6.0
    assert stream.points[-1]["motion_gap_s"] == 5.0
    assert not stream.points[-1]["contract_reference_valid"]


def test_live_contracts_expire_without_new_packet(tmp_path, monkeypatch):
    stream = TwinStream(mode="csv", csv_path=None, rover_url="http://127.0.0.1/js",
                        poll_hz=5.0, output_dir=tmp_path)
    monkeypatch.setattr(stream, "_write_log_record", lambda *_: None)
    stream._append_sample({"source_sample_time_s": 10.0, "enc_left": 0,
                           "enc_right": 0, "gps_valid": False},
                          edge_arrival_s=1010.0, latency_ms=10)
    stream._latest_contracts = [{"service_id": "global_state_tracking", "status": "qualified",
                                 "maximum_aoi_s": .6}]
    received = stream._last_receive_monotonic
    monkeypatch.setattr("DigitalTwin.dashboard.server.time.monotonic", lambda: received + 2.0)
    payload = stream.payload()
    assert payload["contracts"][0]["status"] == "unobservable"
    assert payload["policy"]["decision"]["aoi_s"] >= 2.0

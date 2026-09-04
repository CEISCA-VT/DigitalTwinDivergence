from DigitalTwin.dashboard.server import (
    TwinStream,
    build_replay_payload,
    build_rover_request_url,
    list_logs,
    sanitize_rover_url,
)


def test_dashboard_lists_and_replays_an_accepted_run() -> None:
    logs = list_logs()
    assert len(logs) == 20

    payload = build_replay_payload(str(logs[0]["id"]))
    assert payload["schema"] == "ugv01_dashboard_replay_v2"
    assert payload["summary"]["updates"] == len(payload["points"])
    assert payload["summary"]["updates"] > 100
    assert payload["summary"]["packet_loss"] >= 0

    first = payload["points"][0]
    required = {
        "gps_x",
        "gps_y",
        "ekf_x",
        "ekf_y",
        "security_x",
        "security_y",
        "nis",
        "threshold",
        "velocity",
        "encoder_omega",
        "imu_omega",
        "gyro_bias_deg_s",
        "yaw_disagreement",
        "slip_indicator",
        "satellites",
        "hdop",
        "latency_ms",
    }
    assert required <= first.keys()


def test_sanitize_rover_url_accepts_common_paste_forms() -> None:
    assert sanitize_rover_url("192.168.4.1/js") == "http://192.168.4.1/js"
    assert sanitize_rover_url("[http://192.168.4.1/js]") == "http://192.168.4.1/js"
    assert (
        sanitize_rover_url("[http://192.168.4.1/js](http://192.168.4.1/js)")
        == "http://192.168.4.1/js"
    )


def test_build_rover_request_url_supports_stream_and_legacy_modes() -> None:
    assert (
        build_rover_request_url("192.168.4.1/telemetry", {"T": 147}, "stream")
        == "http://192.168.4.1/telemetry"
    )
    assert (
        build_rover_request_url("http://192.168.4.1/js", {"T": 147}, "cmd")
        == "http://192.168.4.1/js?cmd=%7B%22T%22%3A147%7D"
    )
    assert (
        build_rover_request_url("http://192.168.4.1/js", {"T": 147}, "json")
        == "http://192.168.4.1/js?json=%7B%22T%22%3A147%7D"
    )


def test_twin_stream_records_experiment_metadata(tmp_path) -> None:
    stream = TwinStream(
        mode="csv",
        csv_path=None,
        rover_url="[http://192.168.4.1/js](http://192.168.4.1/js)",
        rover_request_mode="cmd",
        poll_hz=5.0,
        stream_only=True,
        output_dir=tmp_path,
        experiment_metadata={
            "run_label": "carpet_contract_trial_1",
            "physical_condition": "carpet",
            "wireless_condition": "baseline",
            "trial": 1,
        },
    )
    payload = stream.payload()
    assert stream.rover_url == "http://192.168.4.1/js"
    assert "carpet_contract_trial_1" in stream.log_path.name
    assert payload["metadata"]["experiment"]["physical_condition"] == "carpet"
    assert payload["metadata"]["rover_request_mode"] == "cmd"
    assert payload["metadata"]["stream_only"] is True

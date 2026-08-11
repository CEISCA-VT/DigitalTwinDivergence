from DigitalTwin.dashboard.server import build_replay_payload, list_logs


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

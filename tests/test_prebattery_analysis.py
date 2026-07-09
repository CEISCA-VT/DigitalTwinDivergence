import csv
import json
from pathlib import Path

from DigitalTwin.analysis.analyze_bench_telemetry import analyze_bench_telemetry_log
from DigitalTwin.analysis.analyze_stationary import analyze_stationary_log
from DigitalTwin.analysis.calibration_prep import route_reference_template
from DigitalTwin.analysis.review_hardware_replay import _seq_health


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_analyze_stationary_log_handles_missing_gps(tmp_path: Path):
    path = tmp_path / "static_missing_gps.csv"
    _write_csv(
        path,
        ["t_edge_rx_ns", "t_wall_unix_s", "cycle_ok"],
        [
            {"t_edge_rx_ns": 0, "t_wall_unix_s": 1.0, "cycle_ok": "True"},
            {"t_edge_rx_ns": 2_000_000_000, "t_wall_unix_s": 2.0, "cycle_ok": "True"},
        ],
    )
    summary = analyze_stationary_log(path)
    assert summary["gps_valid_rows"] == 0
    assert summary["local_x_span_m"] is None
    assert summary["update_rate_hz_effective"] == 0.5


def test_analyze_bench_telemetry_uses_estimate_latency_when_available(tmp_path: Path):
    path = tmp_path / "bench.csv"
    _write_csv(
        path,
        [
            "cycle_ok",
            "seq",
            "source_sample_time_s",
            "edge_arrival_time_s",
            "estimate_time_s",
            "http_latency_ms",
            "queue_depth",
            "gps_valid",
            "lat",
            "lon",
            "sat",
            "hdop",
            "enc_left",
            "enc_right",
        ],
        [
            {
                "cycle_ok": "True",
                "seq": 0,
                "source_sample_time_s": 10.0,
                "edge_arrival_time_s": 10.5,
                "estimate_time_s": 10.6,
                "http_latency_ms": 500,
                "queue_depth": 0,
                "gps_valid": "True",
                "lat": 37.0,
                "lon": -80.0,
                "sat": 8,
                "hdop": 1.2,
                "enc_left": 0,
                "enc_right": 0,
            },
            {
                "cycle_ok": "True",
                "seq": 2,
                "source_sample_time_s": 11.0,
                "edge_arrival_time_s": 11.4,
                "estimate_time_s": 11.5,
                "http_latency_ms": 400,
                "queue_depth": 1,
                "gps_valid": "True",
                "lat": 37.0,
                "lon": -80.0,
                "sat": 9,
                "hdop": 1.1,
                "enc_left": 10,
                "enc_right": 12,
            },
        ],
    )
    summary = analyze_bench_telemetry_log(path)
    assert summary["pipeline_latency_metric"] == "estimate_time_s - source_sample_time_s"
    assert summary["packet_loss_total"] == 1
    assert summary["queue_depth_p95"] == 0.95
    assert summary["enc_left_total_change"] == 10


def test_seq_health_flags_non_monotonic_and_gap():
    monotonic, gaps = _seq_health([{"seq": "1"}, {"seq": "4"}, {"seq": "3"}])
    assert monotonic is False
    assert gaps == 2


def test_route_reference_template_writes_outputs(tmp_path: Path):
    out_prefix = tmp_path / "route_template"
    route_reference_template(out_prefix)
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["reference_type"] == "overhead_video_or_fiducial"

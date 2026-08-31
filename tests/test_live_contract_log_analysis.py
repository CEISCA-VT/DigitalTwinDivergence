import json

from DigitalTwin.analysis.analyze_live_contract_logs import analyze


def _record(t: float, policy: str, status: str) -> dict[str, object]:
    return {
        "schema": "ugv01_live_contract_record_v1",
        "policy": policy,
        "experiment": {
            "run_label": "unit_live_trial",
            "physical_condition": "turning_intensive",
            "wireless_condition": "wifi_baseline",
            "trial": 1,
        },
        "point": {
            "t": t,
            "gps_valid": True,
            "aoi_s": 0.05,
            "latency_ms": 20.0,
            "payload_bytes": 1000,
            "requested_update_rate_hz": 5.0,
            "resource_mode": "normal",
            "packet_gap": 0,
            "stale": False,
            "queue_depth": 0,
            "gps_agreement_m": 0.03,
            "gps_heading_agreement_deg": 1.0,
            "contracts": [
                {
                    "service_id": "local_1s_tight",
                    "label": "Immediate motion",
                    "horizon_s": 1.0,
                    "position_tolerance_m": 0.1,
                    "heading_tolerance_deg": 2.0,
                    "maximum_aoi_s": 0.6,
                    "status": status,
                    "position_error_m": 0.02,
                    "heading_error_deg": 0.5,
                    "normalized_margin": 0.5,
                }
            ],
        },
        "events": [],
    }


def test_analyze_live_contract_logs(tmp_path) -> None:
    input_dir = tmp_path / "logs"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    path = input_dir / "ugv01_live_contract_unit.jsonl"
    rows = [_record(0.0, "contract-aware", "qualified"), _record(0.2, "contract-aware", "at_risk")]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = analyze(input_dir, output_dir)

    assert manifest["runs_analyzed"] == 1
    assert (output_dir / "live_run_summary.csv").is_file()
    assert (output_dir / "live_service_summary.csv").is_file()
    assert (output_dir / "live_policy_summary.csv").is_file()
    report = (output_dir / "live_contract_experiment_report.md").read_text(encoding="utf-8")
    assert "contract-aware" in report
    assert "Immediate motion" in report

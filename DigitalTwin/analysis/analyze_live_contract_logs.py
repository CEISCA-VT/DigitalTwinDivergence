"""Summarize UGV01 live service-contract experiment logs.

The input is the JSONL stream written by ``DigitalTwin.dashboard.server`` in
live or CSV mode. The analyzer is deliberately descriptive: it does not retune
contracts, alter policy decisions, or reinterpret unobservable service windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "raw_logs" / "live_validation"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "ugv01_live_contract_experiment"
POLICIES = ("static-low", "static-high", "aoi-only", "contract-aware")
STATUSES = ("qualified", "at_risk", "withdrawn", "unobservable")


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values: Iterable[float]) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
            if isinstance(payload, dict) and isinstance(payload.get("point"), dict):
                records.append(payload)
    return records


def _metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        experiment = record.get("experiment")
        if isinstance(experiment, dict):
            return experiment
    return {}


def _point(record: dict[str, Any]) -> dict[str, Any]:
    point = record.get("point")
    return point if isinstance(point, dict) else {}


def _status_rows(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta = _metadata(records)
    policy = str(records[0].get("policy", meta.get("policy", ""))) if records else ""
    for service_id, service_records in _contracts_by_service(records).items():
        statuses = Counter(str(item.get("status", "unobservable")) for item in service_records)
        observable = len(service_records) - statuses["unobservable"]
        position_errors = [_finite(item.get("position_error_m")) for item in service_records]
        heading_errors = [_finite(item.get("heading_error_deg")) for item in service_records]
        margins = [_finite(item.get("normalized_margin")) for item in service_records]
        position_clean = [v for v in position_errors if v is not None]
        heading_clean = [v for v in heading_errors if v is not None]
        margin_clean = [v for v in margins if v is not None]
        sample = service_records[-1] if service_records else {}
        total = len(service_records)
        rows.append(
            {
                "log_file": str(path),
                "run_label": meta.get("run_label", path.stem),
                "physical_condition": meta.get("physical_condition", ""),
                "wireless_condition": meta.get("wireless_condition", ""),
                "trial": meta.get("trial", ""),
                "policy": policy,
                "service_id": service_id,
                "service_label": sample.get("label", service_id),
                "horizon_s": sample.get("horizon_s", ""),
                "position_tolerance_m": sample.get("position_tolerance_m", ""),
                "heading_tolerance_deg": sample.get("heading_tolerance_deg", ""),
                "aoi_limit_s": sample.get("maximum_aoi_s", ""),
                "service_samples": total,
                "observable_fraction": observable / total if total else None,
                "qualified_fraction": statuses["qualified"] / total if total else None,
                "at_risk_fraction": statuses["at_risk"] / total if total else None,
                "withdrawn_fraction": statuses["withdrawn"] / total if total else None,
                "unobservable_fraction": statuses["unobservable"] / total if total else None,
                "satisfied_observable_fraction": (
                    (statuses["qualified"] + statuses["at_risk"]) / observable if observable else None
                ),
                "position_error_p95_m": _quantile(position_clean, 0.95),
                "heading_error_p95_deg": _quantile(heading_clean, 0.95),
                "minimum_normalized_margin": min(margin_clean) if margin_clean else None,
            }
        )
    return rows


def _contracts_by_service(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        contracts = _point(record).get("contracts")
        if not isinstance(contracts, list):
            continue
        for item in contracts:
            if isinstance(item, dict):
                output[str(item.get("service_id", "unknown"))].append(item)
    return output


def _run_row(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    meta = _metadata(records)
    points = [_point(record) for record in records]
    policy = str(records[0].get("policy", meta.get("policy", ""))) if records else ""
    times = [_finite(point.get("t")) for point in points]
    times_clean = [value for value in times if value is not None]
    duration = max(times_clean) - min(times_clean) if len(times_clean) >= 2 else 0.0
    payload_bytes = [_finite(point.get("payload_bytes")) for point in points]
    payload_clean = [value for value in payload_bytes if value is not None]
    latency = [_finite(point.get("latency_ms")) for point in points]
    latency_clean = [value for value in latency if value is not None]
    aoi = [_finite(point.get("aoi_s")) for point in points]
    aoi_clean = [value for value in aoi if value is not None]
    rates = [_finite(point.get("requested_update_rate_hz")) for point in points]
    rates_clean = [value for value in rates if value is not None]
    gps_agreement = [_finite(point.get("gps_agreement_m")) for point in points]
    gps_agreement_clean = [value for value in gps_agreement if value is not None]
    gps_heading = [_finite(point.get("gps_heading_agreement_deg")) for point in points]
    gps_heading_clean = [value for value in gps_heading if value is not None]
    queue_depth = [_finite(point.get("queue_depth")) for point in points]
    queue_clean = [value for value in queue_depth if value is not None]
    stale_count = sum(1 for point in points if bool(point.get("stale")))
    gps_valid_count = sum(1 for point in points if bool(point.get("gps_valid")))
    contract_samples = [
        item
        for service_records in _contracts_by_service(records).values()
        for item in service_records
    ]
    contract_statuses = Counter(str(item.get("status", "unobservable")) for item in contract_samples)
    observable_contracts = len(contract_samples) - contract_statuses["unobservable"]
    resource_modes = Counter(str(point.get("resource_mode", "")) for point in points if point.get("resource_mode"))
    packet_loss = sum(int(_finite(point.get("packet_gap")) or 0) for point in points)
    return {
        "log_file": str(path),
        "run_label": meta.get("run_label", path.stem),
        "physical_condition": meta.get("physical_condition", ""),
        "wireless_condition": meta.get("wireless_condition", ""),
        "trial": meta.get("trial", ""),
        "policy": policy,
        "records": len(records),
        "duration_s": duration,
        "request_count": len(points),
        "actual_update_rate_hz": ((len(points) - 1) / duration if duration > 0.0 and len(points) > 1 else None),
        "requested_update_rate_mean_hz": _mean(rates_clean),
        "requested_update_rate_p95_hz": _quantile(rates_clean, 0.95),
        "payload_bytes_total": sum(payload_clean),
        "payload_bytes_per_s": (sum(payload_clean) / duration if duration > 0.0 else None),
        "latency_p50_ms": _quantile(latency_clean, 0.50),
        "latency_p95_ms": _quantile(latency_clean, 0.95),
        "aoi_p50_s": _quantile(aoi_clean, 0.50),
        "aoi_p95_s": _quantile(aoi_clean, 0.95),
        "packet_loss_total": packet_loss,
        "stale_count": stale_count,
        "stale_fraction": stale_count / len(points) if points else None,
        "queue_depth_mean": _mean(queue_clean),
        "queue_depth_p95": _quantile(queue_clean, 0.95),
        "queue_depth_max": max(queue_clean) if queue_clean else None,
        "gps_valid_fraction": gps_valid_count / len(points) if points else None,
        "gps_agreement_p95_m": _quantile(gps_agreement_clean, 0.95),
        "gps_heading_agreement_p95_deg": _quantile(gps_heading_clean, 0.95),
        "contract_sample_count": len(contract_samples),
        "contract_observable_fraction": observable_contracts / len(contract_samples) if contract_samples else None,
        "contract_qualified_fraction": contract_statuses["qualified"] / len(contract_samples) if contract_samples else None,
        "contract_at_risk_fraction": contract_statuses["at_risk"] / len(contract_samples) if contract_samples else None,
        "contract_withdrawn_fraction": contract_statuses["withdrawn"] / len(contract_samples) if contract_samples else None,
        "contract_unobservable_fraction": contract_statuses["unobservable"] / len(contract_samples) if contract_samples else None,
        "economy_fraction": resource_modes["economy"] / len(points) if points else None,
        "normal_fraction": resource_modes["normal"] / len(points) if points else None,
        "high_fraction": resource_modes["high"] / len(points) if points else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("physical_condition"), row.get("wireless_condition"), row.get("policy"))


def _policy_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[_group_key(row)].append(row)
    output: list[dict[str, Any]] = []
    metrics = [
        "actual_update_rate_hz",
        "requested_update_rate_mean_hz",
        "payload_bytes_total",
        "payload_bytes_per_s",
        "latency_p95_ms",
        "aoi_p95_s",
        "gps_valid_fraction",
        "gps_agreement_p95_m",
        "contract_observable_fraction",
        "contract_qualified_fraction",
        "contract_at_risk_fraction",
        "contract_withdrawn_fraction",
        "contract_unobservable_fraction",
        "economy_fraction",
        "normal_fraction",
        "high_fraction",
    ]
    for (physical, wireless, policy), rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        summary = {
            "physical_condition": physical,
            "wireless_condition": wireless,
            "policy": policy,
            "run_count": len(rows),
        }
        for metric in metrics:
            values = [_finite(row.get(metric)) for row in rows]
            clean = [value for value in values if value is not None]
            summary[f"{metric}_mean"] = _mean(clean)
            summary[f"{metric}_p95"] = _quantile(clean, 0.95)
        output.append(summary)
    return output


def _format(value: Any, digits: int = 3) -> str:
    number = _finite(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _write_report(path: Path, run_rows: list[dict[str, Any]], service_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> None:
    total_runs = len(run_rows)
    policies_seen = sorted({str(row.get("policy")) for row in run_rows if row.get("policy")})
    observable_runs = sum(1 for row in run_rows if (_finite(row.get("contract_observable_fraction")) or 0.0) > 0.0)
    lines = [
        "# UGV01 Live Contract Experiment Summary",
        "",
        "This report summarizes JSONL logs produced by the live UGV01 service-contract dashboard.",
        "Unobservable service windows are retained as unobservable; they are not counted as successful contract satisfaction.",
        "",
        "## Campaign Status",
        "",
        f"- Runs analyzed: {total_runs}",
        f"- Policies present: {', '.join(policies_seen) if policies_seen else 'none'}",
        f"- Runs with any observable contract samples: {observable_runs}",
        f"- Required final policy set: {', '.join(POLICIES)}",
        "",
        "## Policy-Level Summary",
        "",
        "| Physical condition | Wireless condition | Policy | Runs | Observable | Qualified | Withdrawn | p95 AoI (s) | Bytes/s | Requests/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("physical_condition") or ""),
                    str(row.get("wireless_condition") or ""),
                    str(row.get("policy") or ""),
                    str(row.get("run_count") or 0),
                    _format(row.get("contract_observable_fraction_mean")),
                    _format(row.get("contract_qualified_fraction_mean")),
                    _format(row.get("contract_withdrawn_fraction_mean")),
                    _format(row.get("aoi_p95_s_mean")),
                    _format(row.get("payload_bytes_per_s_mean"), 1),
                    _format(row.get("actual_update_rate_hz_mean"), 2),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Service Summary",
            "",
            "| Run | Policy | Service | Observable | Qualified | At risk | Withdrawn | Unobservable | p95 position (m) | p95 heading (deg) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in service_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_label") or Path(str(row.get("log_file", ""))).stem),
                    str(row.get("policy") or ""),
                    str(row.get("service_label") or row.get("service_id") or ""),
                    _format(row.get("observable_fraction")),
                    _format(row.get("qualified_fraction")),
                    _format(row.get("at_risk_fraction")),
                    _format(row.get("withdrawn_fraction")),
                    _format(row.get("unobservable_fraction")),
                    _format(row.get("position_error_p95_m")),
                    _format(row.get("heading_error_p95_deg")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `qualified` and `at_risk` are observable service windows that remain within the declared contract.",
            "- `withdrawn` means an observable contract exceeded position, heading, or AoI limits.",
            "- `unobservable` usually means GPS position/course quality was unavailable for that service.",
            "- The final IoT-J claim requires repeated matched runs across all four policies, not a single dashboard smoke test.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    logs = sorted(input_dir.glob("*.jsonl"))
    run_rows: list[dict[str, Any]] = []
    service_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in logs:
        try:
            records = _read_jsonl(path)
            if not records:
                skipped.append({"log_file": str(path), "reason": "empty"})
                continue
            run_rows.append(_run_row(path, records))
            service_rows.extend(_status_rows(path, records))
        except Exception as exc:
            skipped.append({"log_file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    policy_rows = _policy_rows(run_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "live_run_summary.csv", run_rows)
    _write_csv(output_dir / "live_service_summary.csv", service_rows)
    _write_csv(output_dir / "live_policy_summary.csv", policy_rows)
    if skipped:
        _write_csv(output_dir / "live_skipped_logs.csv", skipped)
    manifest = {
        "schema": "ugv01_live_contract_analysis_manifest_v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "logs_found": len(logs),
        "runs_analyzed": len(run_rows),
        "service_rows": len(service_rows),
        "skipped": skipped,
        "policy_set_required_for_final_experiment": list(POLICIES),
        "outputs": [
            "live_run_summary.csv",
            "live_service_summary.csv",
            "live_policy_summary.csv",
            "live_contract_experiment_report.md",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(output_dir / "live_contract_experiment_report.md", run_rows, service_rows, policy_rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = analyze(args.input_dir, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

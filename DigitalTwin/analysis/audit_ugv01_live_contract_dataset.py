"""Audit the packaged UGV01 live-contract dataset for paper readiness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "public_datasets" / "ugv01_live_contract"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "ugv01_live_contract_dataset_audit"


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def quantile(values: list[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: {exc.msg}")
                continue
            if not isinstance(value, dict) or not isinstance(value.get("point"), dict):
                errors.append(f"line {line_number}: missing point object")
                continue
            rows.append(value)
    return rows, errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    number = finite(value)
    return "" if number is None else f"{number:.{digits}f}"


def audit(dataset: Path, output: Path) -> dict[str, Any]:
    manifest_path = dataset / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))

    output.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    service_rows: list[dict[str, Any]] = []
    file_hashes: dict[str, list[str]] = defaultdict(list)
    telemetry_schemas: set[tuple[str, ...]] = set()
    json_schemas: set[str] = set()

    for item in manifest:
        policy = item["policy"]
        trial = int(item["trial"])
        trial_dir = dataset / "trials" / policy.replace("-", "_") / f"trial_{trial:02d}"
        video = trial_dir / "video.mp4"
        telemetry = trial_dir / "telemetry.csv"
        log = trial_dir / "live_contract.jsonl"
        missing = [path.name for path in (video, telemetry, log) if not path.exists()]

        telemetry_headers: list[str] = []
        telemetry_count = 0
        if telemetry.exists():
            with telemetry.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                telemetry_headers = reader.fieldnames or []
                telemetry_count = sum(1 for _ in reader)
                telemetry_schemas.add(tuple(telemetry_headers))

        records, json_errors = read_jsonl(log) if log.exists() else ([], ["missing log"])
        points = [record["point"] for record in records]
        metadata = records[0].get("experiment", {}) if records else {}
        json_schemas.update(str(record.get("schema", "")) for record in records)
        times = [value for point in points if (value := finite(point.get("t"))) is not None]
        source_times = [finite(point.get("source_time_s")) for point in points]
        if len(source_times) > 1 and all(value is not None for value in source_times) and all(
            b > a for a, b in zip(source_times, source_times[1:])
        ):
            duration = source_times[-1] - source_times[0]
        else:
            duration = max(times) - min(times) if len(times) > 1 else 0.0
        monotonic = all(b > a for a, b in zip(times, times[1:]))
        seqs = [int(value) for point in points if (value := finite(point.get("seq"))) is not None]
        seq_gaps = sum(max(0, b - a - 1) for a, b in zip(seqs, seqs[1:]))
        gps_valid = [bool(point.get("gps_valid")) for point in points]
        rates = [value for point in points if (value := finite(point.get("requested_update_rate_hz"))) is not None]
        payload = [value for point in points if (value := finite(point.get("payload_bytes"))) is not None]
        aoi = [value for point in points if (value := finite(point.get("aoi_s"))) is not None]
        latency = [value for point in points if (value := finite(point.get("latency_ms"))) is not None]
        modes = Counter(str(point.get("resource_mode", "")) for point in points)

        contracts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for point in points:
            for contract in point.get("contracts", []):
                if isinstance(contract, dict):
                    contracts[str(contract.get("service_id", "unknown"))].append(contract)
        all_contracts = [contract for values in contracts.values() for contract in values]
        statuses = Counter(str(contract.get("status", "unobservable")) for contract in all_contracts)

        for path in (video, telemetry, log):
            if path.exists():
                file_hashes[sha256(path)].append(str(path.relative_to(dataset)))

        run_rows.append(
            {
                "policy": policy,
                "trial": trial,
                "physical_condition": metadata.get("physical_condition", ""),
                "wireless_condition": metadata.get("wireless_condition", ""),
                "video_present": video.exists(),
                "telemetry_present": telemetry.exists(),
                "jsonl_present": log.exists(),
                "missing_files": ";".join(missing),
                "telemetry_rows_manifest": item.get("telemetry_rows", ""),
                "telemetry_rows_observed": telemetry_count,
                "telemetry_column_count": len(telemetry_headers),
                "json_records": len(records),
                "json_parse_errors": len(json_errors),
                "duration_s": duration,
                "timestamps_strictly_increasing": monotonic,
                "sequence_gap_total": seq_gaps,
                "gps_valid_fraction": mean(gps_valid) if gps_valid else None,
                "actual_update_rate_hz": (len(points) - 1) / duration if duration > 0 and len(points) > 1 else None,
                "requested_update_rate_mean_hz": mean(rates) if rates else None,
                "requested_update_rate_p95_hz": quantile(rates, 0.95),
                "payload_bytes_total": sum(payload),
                "payload_bytes_per_s": sum(payload) / duration if duration > 0 else None,
                "aoi_p95_s": quantile(aoi, 0.95),
                "latency_p95_ms": quantile(latency, 0.95),
                "contract_observable_fraction": (
                    1.0 - statuses["unobservable"] / len(all_contracts) if all_contracts else None
                ),
                "contract_qualified_fraction": statuses["qualified"] / len(all_contracts) if all_contracts else None,
                "contract_at_risk_fraction": statuses["at_risk"] / len(all_contracts) if all_contracts else None,
                "contract_withdrawn_fraction": statuses["withdrawn"] / len(all_contracts) if all_contracts else None,
                "economy_fraction": modes["economy"] / len(points) if points else None,
                "normal_fraction": modes["normal"] / len(points) if points else None,
                "high_fraction": modes["high"] / len(points) if points else None,
            }
        )

        for service_id, contracts_for_service in contracts.items():
            service_statuses = Counter(str(contract.get("status", "unobservable")) for contract in contracts_for_service)
            reasons = Counter(
                str(contract.get("reason", ""))
                for contract in contracts_for_service
                if contract.get("status") == "unobservable"
            )
            observable = len(contracts_for_service) - service_statuses["unobservable"]
            service_rows.append(
                {
                    "policy": policy,
                    "trial": trial,
                    "service_id": service_id,
                    "samples": len(contracts_for_service),
                    "observable_fraction": observable / len(contracts_for_service),
                    "qualified_fraction": service_statuses["qualified"] / len(contracts_for_service),
                    "at_risk_fraction": service_statuses["at_risk"] / len(contracts_for_service),
                    "withdrawn_fraction": service_statuses["withdrawn"] / len(contracts_for_service),
                    "satisfied_given_observable": (
                        (service_statuses["qualified"] + service_statuses["at_risk"]) / observable
                        if observable
                        else None
                    ),
                    "dominant_unobservable_reason": reasons.most_common(1)[0][0] if reasons else "",
                }
            )

    policy_rows: list[dict[str, Any]] = []
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_policy[str(row["policy"])].append(row)
    policy_metrics = (
        "actual_update_rate_hz",
        "requested_update_rate_mean_hz",
        "payload_bytes_per_s",
        "aoi_p95_s",
        "latency_p95_ms",
        "contract_observable_fraction",
        "contract_qualified_fraction",
        "contract_withdrawn_fraction",
        "economy_fraction",
        "normal_fraction",
        "high_fraction",
    )
    for policy, rows in sorted(by_policy.items()):
        summary: dict[str, Any] = {"policy": policy, "run_count": len(rows)}
        for metric in policy_metrics:
            values = [value for row in rows if (value := finite(row.get(metric))) is not None]
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_sd"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        policy_rows.append(summary)

    service_policy_rows: list[dict[str, Any]] = []
    grouped_services: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in service_rows:
        grouped_services[(str(row["policy"]), str(row["service_id"]))].append(row)
    for (policy, service_id), rows in sorted(grouped_services.items()):
        result: dict[str, Any] = {"policy": policy, "service_id": service_id, "run_count": len(rows)}
        for metric in ("observable_fraction", "qualified_fraction", "at_risk_fraction", "withdrawn_fraction", "satisfied_given_observable"):
            values = [value for row in rows if (value := finite(row.get(metric))) is not None]
            result[f"{metric}_mean"] = mean(values) if values else None
        reasons = Counter(str(row["dominant_unobservable_reason"]) for row in rows if row["dominant_unobservable_reason"])
        result["dominant_unobservable_reason"] = reasons.most_common(1)[0][0] if reasons else ""
        service_policy_rows.append(result)

    duplicate_groups = [paths for paths in file_hashes.values() if len(paths) > 1]
    write_csv(output / "per_run_audit.csv", run_rows)
    write_csv(output / "per_service_audit.csv", service_rows)
    write_csv(output / "policy_summary.csv", policy_rows)
    write_csv(output / "service_policy_summary.csv", service_policy_rows)

    actual_rates = [float(row["actual_update_rate_hz_mean"]) for row in policy_rows]
    rate_spread = max(actual_rates) - min(actual_rates) if actual_rates else None
    complete_runs = sum(
        bool(row["video_present"] and row["telemetry_present"] and row["jsonl_present"])
        for row in run_rows
    )
    parse_errors = sum(int(row["json_parse_errors"]) for row in run_rows)
    seq_gaps = sum(int(row["sequence_gap_total"]) for row in run_rows)
    total_records = sum(int(row["json_records"]) for row in run_rows)
    total_json_duration = sum(float(row["duration_s"]) for row in run_rows)

    lines = [
        "# UGV01 Live-Contract Dataset Audit",
        "",
        "## Audit Verdict",
        "",
        "The dataset is structurally complete and suitable for demonstrating a live UGV01 digital-twin contract prototype. "
        "It does not yet support the stronger causal claim that the contract-aware policy preserves service satisfaction "
        "while reducing delivered communication cost relative to the baselines.",
        "",
        "## Structural Quality",
        "",
        f"- Balanced matrix: 4 policies x 5 trials = {len(run_rows)} runs.",
        f"- Complete video/telemetry/JSONL triplets: {complete_runs}/{len(run_rows)}.",
        f"- JSONL parse errors: {parse_errors}.",
        f"- JSONL records: {total_records} across {total_json_duration / 60.0:.1f} minutes of live traces.",
        f"- Total unexplained sequence gaps: {seq_gaps} ({100.0 * seq_gaps / max(1, total_records + seq_gaps):.3f}% of expected sequence steps).",
        f"- Distinct telemetry schemas: {len(telemetry_schemas)}; distinct JSON record schemas: {len(json_schemas)}.",
        f"- Duplicate file-content groups across trial assets: {len(duplicate_groups)}.",
        f"- Physical conditions: {', '.join(sorted({str(row['physical_condition']) for row in run_rows}))}.",
        f"- Wireless conditions: {', '.join(sorted({str(row['wireless_condition']) for row in run_rows}))}.",
        "",
        "## Policy-Level Evidence",
        "",
        "| Policy | Runs | Requested rate (Hz) | Delivered rate (Hz) | Bytes/s | p95 AoI (s) | Observable | Qualified | Withdrawn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        lines.append(
            f"| {row['policy']} | {row['run_count']} | {fmt(row['requested_update_rate_mean_hz_mean'], 2)} | "
            f"{fmt(row['actual_update_rate_hz_mean'], 2)} | {fmt(row['payload_bytes_per_s_mean'], 1)} | "
            f"{fmt(row['aoi_p95_s_mean'])} | {fmt(row['contract_observable_fraction_mean'])} | "
            f"{fmt(row['contract_qualified_fraction_mean'])} | {fmt(row['contract_withdrawn_fraction_mean'])} |"
        )
    lines.extend(
        [
            "",
            f"Requested policy rates differ strongly, but the mean delivered-rate spread is only {fmt(rate_spread, 3)} Hz. "
            "The rover/HTTP stream therefore acted as the bottleneck and did not realize the requested 2/5/10 Hz policy actions.",
            "",
            "## Service Observability",
            "",
            "| Policy | Service | Observable | Satisfied given observable | Withdrawn | Main reason for missing evaluation |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in service_policy_rows:
        lines.append(
            f"| {row['policy']} | {row['service_id']} | {fmt(row['observable_fraction_mean'])} | "
            f"{fmt(row['satisfied_given_observable_mean'])} | {fmt(row['withdrawn_fraction_mean'])} | "
            f"{row['dominant_unobservable_reason']} |"
        )
    lines.extend(
        [
            "",
            "GPS validity is high, but GPS course is usually unavailable at the rover's low speed. This makes the 1 s, "
            "5 s, and 10 s position-plus-heading contracts mostly unobservable. The global contract is evaluated more often, "
            "but it is usually withdrawn because GPS-to-twin disagreement exceeds the frozen tolerance.",
            "",
            "## Claims Supported Now",
            "",
            "- The complete stack ran online on physical UGV01 hardware and produced reproducible sensor, twin, contract, policy, and resource traces.",
            "- The four frozen policies produce different requested rates and resource-mode decisions.",
            "- The system explicitly marks services unobservable when the operational reference cannot support evaluation.",
            "- Five repetitions per policy support descriptive variability estimates for this smooth-floor, baseline-Wi-Fi condition.",
            "",
            "## Claims Not Supported By This Dataset Alone",
            "",
            "- Contract-aware operation reduces delivered request rate, bytes, latency, or energy relative to static-high.",
            "- Contract-aware operation preserves service satisfaction better than static-low or AoI-only.",
            "- The result generalizes across surfaces, motion regimes, or wireless conditions.",
            "- GPS-referenced online disagreement is independent physical ground truth.",
            "- The paired videos provide independent-reference source material, but they are not synchronized AprilTag ground-truth trajectories in this package yet.",
            "",
            "## Paper Readiness",
            "",
            "Use this dataset as a live prototype and policy-execution study in the paper. Pair it with the existing AprilTag "
            "physical-validation results and the i2Nav condition/fidelity analyses. Describe resource adaptation using requested "
            "rates and policy states. Do not describe the near-identical delivered rates as demonstrated communication savings.",
            "",
            "For the stronger policy-effectiveness claim, the minimum additional evidence is a controlled transport or firmware "
            "path that actually delivers distinct update rates, followed by the same four policies under matched motion. Improving "
            "low-speed heading observability is also required if all finite-horizon live contracts remain part of the claim.",
        ]
    )
    (output / "dataset_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit_manifest = {
        "schema": "ugv01_live_contract_dataset_audit_v1",
        "dataset": str(dataset),
        "run_count": len(run_rows),
        "complete_triplets": complete_runs,
        "policies": {policy: len(rows) for policy, rows in sorted(by_policy.items())},
        "json_parse_errors": parse_errors,
        "sequence_gap_total": seq_gaps,
        "json_record_count": total_records,
        "json_duration_s_total": total_json_duration,
        "telemetry_schema_count": len(telemetry_schemas),
        "json_schema_values": sorted(json_schemas),
        "duplicate_file_groups": duplicate_groups,
        "delivered_rate_spread_hz": rate_spread,
        "verdict": "prototype_ready_policy_effectiveness_not_established",
        "outputs": [
            "per_run_audit.csv",
            "per_service_audit.csv",
            "policy_summary.csv",
            "service_policy_summary.csv",
            "dataset_audit_report.md",
        ],
    }
    (output / "audit_manifest.json").write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")
    return audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

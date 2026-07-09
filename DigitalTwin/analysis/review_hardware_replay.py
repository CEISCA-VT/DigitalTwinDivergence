"""Replay bench telemetry logs and review digital-twin consistency."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .common import parse_bool, parse_float, parse_int, read_rows, write_rows
from .replay_hardware_log import replay_hardware_log


SUMMARY_FIELDS = [
    "log_name",
    "source_successful_gps_rows",
    "replay_rows",
    "row_count_match",
    "raw_seq_monotonic",
    "replay_seq_monotonic",
    "raw_packet_gap_count",
    "replay_packet_gap_count",
    "timing_fields_present",
    "detector_fields_present",
    "detector_values_finite",
    "metrics_nonnegative",
    "detection_count",
    "sustained_detection_warning",
    "warnings",
    "replay_csv",
]


TIMING_FIELDS = [
    "source_sample_time_s",
    "edge_send_time_s",
    "edge_arrival_time_s",
    "queue_release_time_s",
    "estimate_time_s",
    "clock_offset_s",
]
DETECTOR_FIELDS = ["mahalanobis", "threshold", "lambda_max_s", "epsilon_min_m", "confidence"]


def _successful_gps_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if not parse_bool(row.get("cycle_ok", "True")):
            continue
        if not parse_bool(row.get("gps_valid", "False")):
            continue
        if parse_float(row.get("lat", ""), None) is None or parse_float(row.get("lon", ""), None) is None:
            continue
        filtered.append(row)
    return filtered


def _seq_health(rows: list[dict[str, str]]) -> tuple[bool, int]:
    seqs = [seq for seq in (parse_int(row.get("seq", ""), None) for row in rows) if seq is not None]
    if len(seqs) < 2:
        return True, 0
    monotonic = True
    gaps = 0
    prev = seqs[0]
    for seq in seqs[1:]:
        if seq <= prev:
            monotonic = False
        elif seq > prev + 1:
            gaps += seq - prev - 1
        prev = seq
    return monotonic, gaps


def _all_present(rows: list[dict[str, str]], fields: list[str]) -> bool:
    if not rows:
        return False
    return all(any(str(row.get(field, "")).strip() != "" for row in rows) for field in fields)


def _detector_values_ok(rows: list[dict[str, str]]) -> tuple[bool, bool]:
    detector_values_finite = True
    metrics_nonnegative = True
    for row in rows:
        for field in DETECTOR_FIELDS:
            value = parse_float(row.get(field, ""), None)
            if value is None or not math.isfinite(value):
                detector_values_finite = False
                continue
            if field in {"threshold", "lambda_max_s", "epsilon_min_m", "confidence"} and value < 0:
                metrics_nonnegative = False
    return detector_values_finite, metrics_nonnegative


def review_one(path: Path, out_dir: Path) -> dict[str, object]:
    rows = read_rows(path)
    source_rows = _successful_gps_rows(rows)
    replay_path = out_dir / "replayed_csv" / f"{path.stem}_digital_twin.csv"
    if not source_rows:
        return {
            "log_name": path.name,
            "source_successful_gps_rows": 0,
            "replay_rows": 0,
            "row_count_match": False,
            "raw_seq_monotonic": True,
            "replay_seq_monotonic": False,
            "raw_packet_gap_count": 0,
            "replay_packet_gap_count": 0,
            "timing_fields_present": False,
            "detector_fields_present": False,
            "detector_values_finite": False,
            "metrics_nonnegative": False,
            "detection_count": 0,
            "sustained_detection_warning": False,
            "warnings": "no successful GPS-valid source rows",
            "replay_csv": "",
        }
    replay_hardware_log(path, replay_path)
    replay_rows = read_rows(replay_path)

    raw_monotonic, raw_gaps = _seq_health(source_rows)
    replay_monotonic, replay_gaps = _seq_health(replay_rows)
    timing_fields_present = _all_present(replay_rows, TIMING_FIELDS)
    detector_fields_present = _all_present(replay_rows, DETECTOR_FIELDS)
    detector_values_finite, metrics_nonnegative = _detector_values_ok(replay_rows)
    detection_count = sum(1 for row in replay_rows if parse_bool(row.get("detected", "")))
    detection_fraction = (detection_count / len(replay_rows)) if replay_rows else 0.0
    sustained_warning = detection_count > 0 and detection_fraction >= 0.02

    warnings: list[str] = []
    if len(source_rows) != len(replay_rows):
        warnings.append("row-count mismatch")
    if not raw_monotonic:
        warnings.append("raw seq not monotonic")
    if not replay_monotonic:
        warnings.append("replay seq not monotonic")
    if not timing_fields_present:
        warnings.append("timing fields missing in replay")
    if not detector_fields_present:
        warnings.append("detector fields missing in replay")
    if not detector_values_finite:
        warnings.append("detector values not finite")
    if not metrics_nonnegative:
        warnings.append("negative detector metrics")
    if sustained_warning:
        warnings.append("unexpected sustained detections in stationary bench replay")

    return {
        "log_name": path.name,
        "source_successful_gps_rows": len(source_rows),
        "replay_rows": len(replay_rows),
        "row_count_match": len(source_rows) == len(replay_rows),
        "raw_seq_monotonic": raw_monotonic,
        "replay_seq_monotonic": replay_monotonic,
        "raw_packet_gap_count": raw_gaps,
        "replay_packet_gap_count": replay_gaps,
        "timing_fields_present": timing_fields_present,
        "detector_fields_present": detector_fields_present,
        "detector_values_finite": detector_values_finite,
        "metrics_nonnegative": metrics_nonnegative,
        "detection_count": detection_count,
        "sustained_detection_warning": sustained_warning,
        "warnings": "; ".join(warnings),
        "replay_csv": str(replay_path),
    }


def _render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Hardware Replay Consistency Review",
        "",
        "| Log | Source rows | Replay rows | Timing | Detector | Detections | Warnings |",
        "| --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {log} | {src} | {replay} | {timing} | {detector} | {detections} | {warnings} |".format(
                log=row["log_name"],
                src=row["source_successful_gps_rows"],
                replay=row["replay_rows"],
                timing="ok" if row["timing_fields_present"] else "missing",
                detector="ok" if row["detector_fields_present"] and row["detector_values_finite"] else "check",
                detections=row["detection_count"],
                warnings=row["warnings"] or "none",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="raw_logs/telemetry")
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis/hardware_replay_review")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("*.csv")):
        summaries.append(review_one(path, out_dir))

    write_rows(out_dir / "hardware_replay_review.csv", summaries, SUMMARY_FIELDS)
    (out_dir / "hardware_replay_review.md").write_text(_render_markdown(summaries), encoding="utf-8")
    print(out_dir / "hardware_replay_review.csv")
    print(out_dir / "hardware_replay_review.md")


if __name__ == "__main__":
    main()

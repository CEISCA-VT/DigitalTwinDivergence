"""Summarize bench telemetry timing, packet health, and GPS context."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

from .common import cleaned_floats, first_present, parse_bool, parse_float, parse_int, quantile, read_rows, write_rows


SUMMARY_FIELDS = [
    "log_name",
    "schema_variant",
    "total_cycles",
    "successful_cycles",
    "success_fraction",
    "nominal_packet_rate_hz",
    "seq_start",
    "seq_end",
    "packet_loss_total",
    "packet_loss_rate",
    "stale_packet_count",
    "stale_packet_rate",
    "queue_depth_min",
    "queue_depth_mean",
    "queue_depth_p95",
    "queue_depth_max",
    "pipeline_latency_metric",
    "pipeline_latency_p50_ms",
    "pipeline_latency_p95_ms",
    "pipeline_latency_p99_ms",
    "http_latency_p50_ms",
    "http_latency_p95_ms",
    "http_latency_p99_ms",
    "source_to_arrival_p50_ms",
    "source_to_arrival_p95_ms",
    "source_to_arrival_p99_ms",
    "source_to_estimate_p50_ms",
    "source_to_estimate_p95_ms",
    "source_to_estimate_p99_ms",
    "source_to_alarm_p50_ms",
    "source_to_alarm_p95_ms",
    "source_to_alarm_p99_ms",
    "gps_valid_fraction",
    "sat_min",
    "sat_median",
    "sat_max",
    "hdop_min",
    "hdop_median",
    "hdop_max",
    "enc_left_total_change",
    "enc_right_total_change",
]


def _stats_ms(values_s: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values_s:
        return None, None, None
    values_ms = [value * 1000.0 for value in values_s]
    return quantile(values_ms, 0.50), quantile(values_ms, 0.95), quantile(values_ms, 0.99)


def analyze_bench_telemetry_log(path: Path) -> dict[str, object]:
    rows = read_rows(path)
    ok_rows = [row for row in rows if parse_bool(row.get("cycle_ok", "True"))]
    schema_variant = "timing_v2" if ok_rows and "source_sample_time_s" in ok_rows[0] else "legacy"

    sample_times = cleaned_floats(first_present(row, "source_sample_time_s", "rover_millis_s") for row in ok_rows)
    if not sample_times:
        sample_times = [value / 1000.0 for value in cleaned_floats(first_present(row, "sample_ms", "millis") for row in ok_rows)]
    dts = [b - a for a, b in zip(sample_times, sample_times[1:]) if b > a]
    nominal_packet_rate_hz = None
    if dts:
        nominal_packet_rate_hz = 1.0 / median(dts)

    seqs = [seq for seq in (parse_int(row.get("seq", ""), None) for row in ok_rows) if seq is not None]
    packet_loss_total = 0
    stale_packet_count = 0
    if seqs:
        prev_seq = seqs[0]
        for seq in seqs[1:]:
            if seq <= prev_seq:
                stale_packet_count += 1
            else:
                packet_loss_total += max(0, seq - prev_seq - 1)
            prev_seq = seq
    if ok_rows and "stale_packet" in ok_rows[0]:
        stale_packet_count = sum(1 for row in ok_rows if parse_bool(row.get("stale_packet", "")))
    packet_loss_rate = packet_loss_total / len(ok_rows) if ok_rows else None
    stale_packet_rate = stale_packet_count / len(ok_rows) if ok_rows else None

    queue_depths = [depth for depth in (parse_int(row.get("queue_depth", ""), None) for row in ok_rows) if depth is not None]
    queue_depth_mean = (sum(queue_depths) / len(queue_depths)) if queue_depths else None

    http_latency_s = [value / 1000.0 for value in cleaned_floats(row.get("http_latency_ms", "") for row in ok_rows)]
    source_to_arrival_s = []
    source_to_estimate_s = []
    source_to_alarm_s = []
    for row in ok_rows:
        source = parse_float(row.get("source_sample_time_s", ""), None)
        clock_offset_s = parse_float(row.get("clock_offset_s", ""), None)
        arrival = parse_float(row.get("edge_arrival_time_s", ""), None)
        estimate = parse_float(row.get("estimate_time_s", ""), None)
        alarm = parse_float(row.get("alarm_time_s", ""), None)
        calibrated_source = source + clock_offset_s if source is not None and clock_offset_s is not None else source
        if calibrated_source is not None and arrival is not None:
            source_to_arrival_s.append(arrival - calibrated_source)
        if calibrated_source is not None and estimate is not None:
            source_to_estimate_s.append(estimate - calibrated_source)
        if calibrated_source is not None and alarm is not None:
            source_to_alarm_s.append(alarm - calibrated_source)
    if source_to_estimate_s:
        pipeline_latency_metric = "estimate_time_s - calibrated_source_time_s"
        pipeline_values_s = source_to_estimate_s
    elif source_to_arrival_s:
        pipeline_latency_metric = "edge_arrival_time_s - calibrated_source_time_s"
        pipeline_values_s = source_to_arrival_s
    else:
        pipeline_latency_metric = "http_latency_ms"
        pipeline_values_s = http_latency_s

    gps_valid_fraction = None
    gps_valid_rows = [
        row for row in ok_rows
        if parse_bool(first_present(row, "gps_valid")) and parse_float(first_present(row, "lat", "gps_lat"), None) is not None
    ]
    if ok_rows:
        gps_valid_fraction = len(gps_valid_rows) / len(ok_rows)
    sats = cleaned_floats(first_present(row, "sat", "gps_sat") for row in gps_valid_rows)
    hdops = cleaned_floats(first_present(row, "hdop", "gps_hdop") for row in gps_valid_rows)
    left_enc = [value for value in (parse_int(row.get("enc_left", ""), None) for row in ok_rows) if value is not None]
    right_enc = [value for value in (parse_int(row.get("enc_right", ""), None) for row in ok_rows) if value is not None]

    pipeline_p50, pipeline_p95, pipeline_p99 = _stats_ms(pipeline_values_s)
    http_p50, http_p95, http_p99 = _stats_ms(http_latency_s)
    arrival_p50, arrival_p95, arrival_p99 = _stats_ms(source_to_arrival_s)
    estimate_p50, estimate_p95, estimate_p99 = _stats_ms(source_to_estimate_s)
    alarm_p50, alarm_p95, alarm_p99 = _stats_ms(source_to_alarm_s)
    return {
        "log_name": path.name,
        "schema_variant": schema_variant,
        "total_cycles": len(rows),
        "successful_cycles": len(ok_rows),
        "success_fraction": len(ok_rows) / len(rows) if rows else None,
        "nominal_packet_rate_hz": nominal_packet_rate_hz,
        "seq_start": seqs[0] if seqs else None,
        "seq_end": seqs[-1] if seqs else None,
        "packet_loss_total": packet_loss_total,
        "packet_loss_rate": packet_loss_rate,
        "stale_packet_count": stale_packet_count,
        "stale_packet_rate": stale_packet_rate,
        "queue_depth_min": min(queue_depths) if queue_depths else None,
        "queue_depth_mean": queue_depth_mean,
        "queue_depth_p95": quantile(queue_depths, 0.95) if queue_depths else None,
        "queue_depth_max": max(queue_depths) if queue_depths else None,
        "pipeline_latency_metric": pipeline_latency_metric,
        "pipeline_latency_p50_ms": pipeline_p50,
        "pipeline_latency_p95_ms": pipeline_p95,
        "pipeline_latency_p99_ms": pipeline_p99,
        "http_latency_p50_ms": http_p50,
        "http_latency_p95_ms": http_p95,
        "http_latency_p99_ms": http_p99,
        "source_to_arrival_p50_ms": arrival_p50,
        "source_to_arrival_p95_ms": arrival_p95,
        "source_to_arrival_p99_ms": arrival_p99,
        "source_to_estimate_p50_ms": estimate_p50,
        "source_to_estimate_p95_ms": estimate_p95,
        "source_to_estimate_p99_ms": estimate_p99,
        "source_to_alarm_p50_ms": alarm_p50,
        "source_to_alarm_p95_ms": alarm_p95,
        "source_to_alarm_p99_ms": alarm_p99,
        "gps_valid_fraction": gps_valid_fraction,
        "sat_min": min(sats) if sats else None,
        "sat_median": median(sats) if sats else None,
        "sat_max": max(sats) if sats else None,
        "hdop_min": min(hdops) if hdops else None,
        "hdop_median": median(hdops) if hdops else None,
        "hdop_max": max(hdops) if hdops else None,
        "enc_left_total_change": (left_enc[-1] - left_enc[0]) if len(left_enc) >= 2 else None,
        "enc_right_total_change": (right_enc[-1] - right_enc[0]) if len(right_enc) >= 2 else None,
    }


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_markdown(summaries: list[dict[str, object]]) -> str:
    lines = [
        "# Bench Telemetry Health And Latency Summary",
        "",
        "Primary edge-pipeline latency uses the clock-calibrated source time when available: `estimate_time_s - (source_sample_time_s + clock_offset_s)`, otherwise `edge_arrival_time_s - (source_sample_time_s + clock_offset_s)`, and finally `http_latency_ms` for legacy logs.",
        "",
        "| Log | Schema | Success | Rate Hz | Loss | Stale | Queue p95 | Pipeline p95 ms | HTTP p95 ms | GPS valid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {log} | {schema} | {ok}/{total} | {hz} | {loss} | {stale} | {queue} | {pipe} | {http} | {gps} |".format(
                log=row["log_name"],
                schema=row["schema_variant"],
                ok=row["successful_cycles"],
                total=row["total_cycles"],
                hz=_fmt(row["nominal_packet_rate_hz"]),
                loss=row["packet_loss_total"],
                stale=row["stale_packet_count"],
                queue=_fmt(row["queue_depth_p95"]),
                pipe=_fmt(row["pipeline_latency_p95_ms"]),
                http=_fmt(row["http_latency_p95_ms"]),
                gps=_fmt(row["gps_valid_fraction"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="raw_logs/telemetry")
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis/bench_telemetry")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = out_dir / "per_file"

    summaries: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("*.csv")):
        summary = analyze_bench_telemetry_log(path)
        summaries.append(summary)
        write_rows(per_file_dir / f"{path.stem}_summary.csv", [summary], SUMMARY_FIELDS)

    write_rows(out_dir / "bench_telemetry_summary.csv", summaries, SUMMARY_FIELDS)
    (out_dir / "bench_telemetry_report.md").write_text(_render_markdown(summaries), encoding="utf-8")
    print(out_dir / "bench_telemetry_summary.csv")
    print(out_dir / "bench_telemetry_report.md")


if __name__ == "__main__":
    main()

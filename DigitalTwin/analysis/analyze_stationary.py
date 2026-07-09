"""Summarize stationary GPS/base/IMU logs collected before field deployment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import median

from DigitalTwin.telemetry import gps_to_local_xy

from .common import (
    cleaned_floats,
    first_present,
    parse_bool,
    parse_float,
    parse_int,
    quantile,
    read_rows,
    stats_dict,
    write_rows,
)


SUMMARY_FIELDS = [
    "log_name",
    "total_rows",
    "successful_rows",
    "success_fraction",
    "gps_valid_rows",
    "gps_valid_fraction",
    "first_wall_time_s",
    "last_wall_time_s",
    "duration_s",
    "update_rate_hz_median",
    "update_rate_hz_effective",
    "sat_min",
    "sat_median",
    "sat_max",
    "hdop_min",
    "hdop_median",
    "hdop_max",
    "checksum_start",
    "checksum_end",
    "checksum_delta",
    "sentences_start",
    "sentences_end",
    "sentences_delta",
    "checksum_failures_per_sentence",
    "lat_span_deg",
    "lon_span_deg",
    "local_x_span_m",
    "local_y_span_m",
    "local_rms_radius_m",
    "speed_mps_min",
    "speed_mps_median",
    "speed_mps_max",
]


def _valid_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows or "cycle_ok" not in rows[0]:
        return rows
    return [row for row in rows if parse_bool(row.get("cycle_ok", ""))]


def _gps_valid_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    valid_rows: list[dict[str, str]] = []
    for row in rows:
        is_valid = parse_bool(first_present(row, "gps_valid"))
        lat = parse_float(first_present(row, "gps_lat", "lat"), None)
        lon = parse_float(first_present(row, "gps_lon", "lon"), None)
        if is_valid and lat is not None and lon is not None:
            valid_rows.append(row)
    return valid_rows


def analyze_stationary_log(path: Path) -> dict[str, object]:
    rows = read_rows(path)
    ok_rows = _valid_rows(rows)
    gps_rows = _gps_valid_rows(ok_rows)

    wall_times = cleaned_floats(row.get("t_wall_unix_s", "") for row in ok_rows)
    rx_times_ns = cleaned_floats(row.get("t_edge_rx_ns", "") for row in ok_rows)
    rx_dt_s = [
        max((b - a) / 1_000_000_000.0, 0.0)
        for a, b in zip(rx_times_ns, rx_times_ns[1:])
        if b > a
    ]
    effective_rate_hz = None
    if len(rx_times_ns) >= 2 and rx_times_ns[-1] > rx_times_ns[0]:
        effective_rate_hz = (len(rx_times_ns) - 1) / ((rx_times_ns[-1] - rx_times_ns[0]) / 1_000_000_000.0)
    update_rate_hz_median = None
    if rx_dt_s:
        positive = [dt for dt in rx_dt_s if dt > 0]
        if positive:
            update_rate_hz_median = 1.0 / median(positive)

    sats = cleaned_floats(first_present(row, "gps_sat", "sat") for row in gps_rows)
    hdops = cleaned_floats(first_present(row, "gps_hdop", "hdop") for row in gps_rows)
    speeds = cleaned_floats(first_present(row, "gps_speed_mps", "speed_mps") for row in gps_rows)
    latitudes = cleaned_floats(first_present(row, "gps_lat", "lat") for row in gps_rows)
    longitudes = cleaned_floats(first_present(row, "gps_lon", "lon") for row in gps_rows)
    checksums = cleaned_floats(first_present(row, "gps_failed_checksums") for row in ok_rows)
    sentences = cleaned_floats(first_present(row, "gps_sentences") for row in ok_rows)

    local_x: list[float] = []
    local_y: list[float] = []
    if gps_rows:
        origin_lat = parse_float(first_present(gps_rows[0], "gps_lat", "lat"), 0.0)
        origin_lon = parse_float(first_present(gps_rows[0], "gps_lon", "lon"), 0.0)
        assert origin_lat is not None
        assert origin_lon is not None
        for row in gps_rows:
            lat = parse_float(first_present(row, "gps_lat", "lat"), None)
            lon = parse_float(first_present(row, "gps_lon", "lon"), None)
            if lat is None or lon is None:
                continue
            x, y = gps_to_local_xy(lat, lon, origin_lat, origin_lon)
            local_x.append(x)
            local_y.append(y)

    rms_radius_m = None
    if local_x and local_y:
        rms_radius_m = math.sqrt(sum(x * x + y * y for x, y in zip(local_x, local_y)) / len(local_x))

    checksum_delta = None
    sentences_delta = None
    if checksums:
        checksum_delta = checksums[-1] - checksums[0]
    if sentences:
        sentences_delta = sentences[-1] - sentences[0]
    checksum_ratio = None
    if checksum_delta is not None and sentences_delta and sentences_delta > 0:
        checksum_ratio = checksum_delta / sentences_delta

    summary: dict[str, object] = {
        "log_name": path.name,
        "total_rows": len(rows),
        "successful_rows": len(ok_rows),
        "success_fraction": len(ok_rows) / len(rows) if rows else None,
        "gps_valid_rows": len(gps_rows),
        "gps_valid_fraction": len(gps_rows) / len(ok_rows) if ok_rows else None,
        "first_wall_time_s": wall_times[0] if wall_times else None,
        "last_wall_time_s": wall_times[-1] if wall_times else None,
        "duration_s": (wall_times[-1] - wall_times[0]) if len(wall_times) >= 2 else None,
        "update_rate_hz_median": update_rate_hz_median,
        "update_rate_hz_effective": effective_rate_hz,
        "sat_min": min(sats) if sats else None,
        "sat_median": median(sats) if sats else None,
        "sat_max": max(sats) if sats else None,
        "hdop_min": min(hdops) if hdops else None,
        "hdop_median": median(hdops) if hdops else None,
        "hdop_max": max(hdops) if hdops else None,
        "checksum_start": checksums[0] if checksums else None,
        "checksum_end": checksums[-1] if checksums else None,
        "checksum_delta": checksum_delta,
        "sentences_start": sentences[0] if sentences else None,
        "sentences_end": sentences[-1] if sentences else None,
        "sentences_delta": sentences_delta,
        "checksum_failures_per_sentence": checksum_ratio,
        "lat_span_deg": (max(latitudes) - min(latitudes)) if latitudes else None,
        "lon_span_deg": (max(longitudes) - min(longitudes)) if longitudes else None,
        "local_x_span_m": (max(local_x) - min(local_x)) if local_x else None,
        "local_y_span_m": (max(local_y) - min(local_y)) if local_y else None,
        "local_rms_radius_m": rms_radius_m,
        "speed_mps_min": min(speeds) if speeds else None,
        "speed_mps_median": median(speeds) if speeds else None,
        "speed_mps_max": max(speeds) if speeds else None,
    }
    return summary


def _render_markdown(summaries: list[dict[str, object]]) -> str:
    lines = [
        "# Stationary GPS Summary",
        "",
        "This report summarizes pre-battery stationary UGV01 logs. Position variation uses the first valid fix as the local-frame origin and only includes GPS-valid rows.",
        "",
    ]
    if not summaries:
        lines.append("No stationary logs were found.")
        return "\n".join(lines) + "\n"

    aggregate = {
        "logs": len(summaries),
        "total_rows": sum(int(row["total_rows"]) for row in summaries),
        "gps_valid_rows": sum(int(row["gps_valid_rows"]) for row in summaries),
    }
    lines.extend(
        [
            f"- Logs analyzed: `{aggregate['logs']}`",
            f"- Total rows: `{aggregate['total_rows']}`",
            f"- GPS-valid rows: `{aggregate['gps_valid_rows']}`",
            "",
            "| Log | GPS-valid | Sat median | HDOP median | Update Hz | X span m | Y span m | RMS radius m |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        lines.append(
            "| {log} | {valid} | {sat} | {hdop} | {hz} | {xspan} | {yspan} | {rms} |".format(
                log=row["log_name"],
                valid=row["gps_valid_rows"],
                sat=_fmt(row["sat_median"]),
                hdop=_fmt(row["hdop_median"]),
                hz=_fmt(row["update_rate_hz_effective"]),
                xspan=_fmt(row["local_x_span_m"]),
                yspan=_fmt(row["local_y_span_m"]),
                rms=_fmt(row["local_rms_radius_m"]),
            )
        )
    lines.extend(["", "## Notes", ""])
    for row in summaries:
        notes: list[str] = []
        if not row["gps_valid_rows"]:
            notes.append("no valid GPS fixes")
        if row["checksum_delta"] not in {None, 0}:
            notes.append(f"checksum delta {_fmt(row['checksum_delta'])}")
        if row["speed_mps_median"] not in {None, 0}:
            notes.append(f"median GPS speed {_fmt(row['speed_mps_median'])} m/s")
        if not notes:
            notes.append("clean stationary capture")
        lines.append(f"- `{row['log_name']}`: " + ", ".join(notes))
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="raw_logs/static")
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis/stationary")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    per_file_dir = out_dir / "per_file"
    for path in sorted(input_dir.glob("*.csv")):
        summary = analyze_stationary_log(path)
        summaries.append(summary)
        write_rows(per_file_dir / f"{path.stem}_summary.csv", [summary], SUMMARY_FIELDS)

    write_rows(out_dir / "stationary_summary.csv", summaries, SUMMARY_FIELDS)
    (out_dir / "stationary_report.md").write_text(_render_markdown(summaries), encoding="utf-8")
    print(out_dir / "stationary_summary.csv")
    print(out_dir / "stationary_report.md")


if __name__ == "__main__":
    main()

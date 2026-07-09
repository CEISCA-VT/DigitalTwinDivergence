import csv
import json
import time
from pathlib import Path

import requests

from DigitalTwin.timing import SessionClockCalibrator

ROVER_IP = "192.168.4.1"
BASE_URL = f"http://{ROVER_IP}/js"

OUT_DIR = Path("raw_logs/telemetry")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / f"ugv_t147_telemetry_{time.strftime('%Y%m%d_%H%M%S')}.csv"

DURATION_SECONDS = 600       # 10 minutes
POLL_INTERVAL_SECONDS = 1.0  # one T:147 request per second
HTTP_TIMEOUT_SECONDS = 3


FIELDNAMES = [
    "sample_idx",
    "cycle_ok",
    "error",

    "t_wall_unix_s",
    "t_cycle_start_ns",
    "t_edge_send_ns",
    "t_edge_rx_ns",
    "t_edge_mid_ns",
    "http_latency_ms",
    "rover_millis_s",
    "source_sample_time_s",
    "edge_arrival_time_s",
    "estimate_time_s",
    "alarm_time_s",
    "clock_offset_s",
    "clock_calibrated",
    "clock_calibration_samples",
    "packet_loss_count",
    "stale_packet",
    "queue_depth",

    # Firmware response
    "T",
    "seq",
    "sample_ms",
    "send_ms",
    "millis",

    # Base / wheel / voltage
    "L",
    "R",
    "enc_left",
    "enc_right",
    "v",

    # IMU attitude
    "r",
    "p",
    "y",

    # IMU accel / gyro / mag
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "mx",
    "my",
    "mz",
    "temp",

    # GPS
    "gps_valid",
    "gps_age_ms",
    "gps_fix_type",
    "lat",
    "lon",
    "sat",
    "hdop",
    "alt_m",
    "speed_mps",
    "course_deg",
    "gps_chars",
    "gps_sentences",
    "gps_failed_checksums",
]


def query_telemetry() -> tuple[dict, int, int, float]:
    """Request combined UGV telemetry using T:147."""
    command = {"T": 147}
    t0 = time.monotonic_ns()

    response = requests.get(
        BASE_URL,
        params={"cmd": json.dumps(command, separators=(",", ":"))},
        timeout=HTTP_TIMEOUT_SECONDS,
    )

    t1 = time.monotonic_ns()
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        raise RuntimeError("UGV returned an empty response.")

    data = json.loads(text)
    latency_ms = (t1 - t0) / 1_000_000.0

    return data, t0, t1, latency_ms


def main() -> None:
    print(f"Logging for {DURATION_SECONDS} seconds")
    print(f"Output: {OUTPUT_CSV}")
    print("Use only the rover Wi-Fi. Do not keep the browser page open while logging.")
    print("Command used: {\"T\":147}")
    print()

    start_time = time.monotonic()
    end_time = start_time + DURATION_SECONDS

    sample_idx = 0
    successful = 0
    dropped = 0
    prev_rover_millis: int | None = None
    clock = SessionClockCalibrator()

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        while time.monotonic() < end_time:
            row = {key: None for key in FIELDNAMES}
            row["sample_idx"] = sample_idx
            row["t_wall_unix_s"] = time.time()
            row["t_cycle_start_ns"] = time.monotonic_ns()
            row["cycle_ok"] = False

            try:
                telemetry, send_ns, rx_ns, latency_ms = query_telemetry()

                row["t_edge_send_ns"] = send_ns
                row["t_edge_rx_ns"] = rx_ns
                row["t_edge_mid_ns"] = (send_ns + rx_ns) // 2
                row["http_latency_ms"] = latency_ms
                row["cycle_ok"] = True

                for key, value in telemetry.items():
                    if key in row:
                        row[key] = value

                rover_millis = telemetry.get("sample_ms", telemetry.get("millis"))
                if rover_millis is not None:
                    rover_millis = int(rover_millis)
                    rover_millis_s = rover_millis / 1000.0
                    edge_mid_s = row["t_edge_mid_ns"] / 1_000_000_000.0
                    estimate = clock.observe(rover_millis_s, edge_mid_s)
                    row["rover_millis_s"] = rover_millis_s
                    row["source_sample_time_s"] = estimate.remote_time_s
                    row["edge_arrival_time_s"] = rx_ns / 1_000_000_000.0
                    row["estimate_time_s"] = rx_ns / 1_000_000_000.0
                    row["alarm_time_s"] = ""
                    row["clock_offset_s"] = estimate.offset_s
                    row["clock_calibrated"] = estimate.calibrated
                    row["clock_calibration_samples"] = estimate.samples
                    row["stale_packet"] = (
                        prev_rover_millis is not None and rover_millis <= prev_rover_millis
                    )
                    prev_rover_millis = rover_millis

                row["packet_loss_count"] = dropped
                row["queue_depth"] = 0

                successful += 1

                print(
                    f"{sample_idx:04d} | "
                    f"L={row['L']} R={row['R']} | "
                    f"yaw={row['y']} | "
                    f"V={row['v']} | "
                    f"GPS={row['gps_valid']} sat={row['sat']} "
                    f"hdop={row['hdop']} chars={row['gps_chars']} | "
                    f"latency={latency_ms:.1f} ms"
                )

            except Exception as exc:
                dropped += 1
                row["t_edge_rx_ns"] = time.monotonic_ns()
                row["packet_loss_count"] = dropped
                row["stale_packet"] = False
                row["queue_depth"] = 0
                row["error"] = str(exc)

                print(f"{sample_idx:04d} | DROP | {exc}")

            writer.writerow(row)
            file.flush()

            sample_idx += 1
            time.sleep(POLL_INTERVAL_SECONDS)

    print()
    print("Done.")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Total cycles: {sample_idx}")
    print(f"Successful cycles: {successful}")
    print(f"Dropped cycles: {dropped}")

    if sample_idx > 0:
        print(f"Drop rate: {100.0 * dropped / sample_idx:.2f}%")


if __name__ == "__main__":
    main()

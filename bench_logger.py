import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from DigitalTwin.timing import SessionClockCalibrator


# ============================================================================
# UGV01 NETWORK CONFIGURATION
# ============================================================================

# Station-mode IP shown next to "ST" on the UGV01 OLED.
ROVER_IP = "10.0.0.119"
BASE_URL = f"http://{ROVER_IP}/js"


# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

OUT_DIR = Path("raw_logs/telemetry")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / (
    f"ugv_t147_bench_{time.strftime('%Y%m%d_%H%M%S')}.csv"
)

DURATION_SECONDS = 900        # 15 minutes
POLL_INTERVAL_SECONDS = 0.25
HTTP_TIMEOUT_SECONDS = 2.0


# ============================================================================
# OPTIONAL MOTION SCRIPT
# ============================================================================

# Safety default: keep disabled unless you explicitly want the logger to drive
# the rover while recording.
MOTION_SCRIPT_ENABLED = True
MOTION_PLAN = "validation_triplet"

# Official UGV01 nominal geometry from the Waveshare documentation:
# - rail center distance: 170 mm
# - track width: 44 mm
# For the motion model, use the stock UGV01 firmware control constants so the
# logger matches the rover's own vendor baseline exactly.
OFFICIAL_RAIL_CENTER_DISTANCE_M = 0.170
OFFICIAL_TRACK_BELT_WIDTH_M = 0.044
OFFICIAL_RUNNING_SPEED_RANGE_MPS = (0.25, 1.0)

# UGV01 nominal drivetrain constants from the embedded firmware's mainType == 3
# configuration. This keeps the bench motion script aligned with the rover's
# own baseline model while still using the official UGV01 geometry above.
UGV01_FIRMWARE_WHEEL_DIAMETER_M = 0.0523
UGV01_FIRMWARE_ENCODER_COUNTS_PER_REV = 1092
UGV01_FIRMWARE_METERS_PER_COUNT = (
    3.141592653589793 * UGV01_FIRMWARE_WHEEL_DIAMETER_M
    / UGV01_FIRMWARE_ENCODER_COUNTS_PER_REV
)
UGV01_FIRMWARE_COUNTS_PER_METER = 1.0 / UGV01_FIRMWARE_METERS_PER_COUNT

# Conservative motion commands based on the ranges seen in basic_test.csv.
STRAIGHT_FORWARD_CMD = (-0.18, -0.18)
STRAIGHT_BACKWARD_CMD = (0.18, 0.18)
TURN_CCW_CMD = (-0.35, 0.35)
TURN_CW_CMD = (0.35, -0.35)
SQUARE_TURN_CW_CMD = (0.28, -0.28)

STRAIGHT_DISTANCE_M = 1.0
TURN_DEGREES = 360.0
TURN_REPEAT_COUNT = 2
SEQUENCE_REPEAT_COUNT = 3
SQUARE_SIDE_LENGTH_M = 0.25
SQUARE_TURN_DEGREES = 90.0
SQUARE_REPEAT_COUNT = 1
HOLD_SECONDS = 2.0
INITIAL_HOLD_SECONDS = 5.0
FINAL_HOLD_SECONDS = 3.0


FIELDNAMES = [
    # Logger status
    "sample_idx",
    "cycle_ok",
    "error",

    # Host timing
    "t_wall_unix_s",
    "t_cycle_start_ns",
    "t_edge_send_ns",
    "t_edge_rx_ns",
    "t_edge_mid_ns",
    "http_latency_ms",

    # Clock calibration
    "rover_millis_s",
    "source_sample_time_s",
    "edge_arrival_time_s",
    "estimate_time_s",
    "alarm_time_s",
    "clock_offset_s",
    "clock_calibrated",
    "clock_calibration_samples",

    # Transport quality
    "packet_loss_count",
    "request_failure_count",
    "sequence_gap_count",
    "stale_packet",
    "queue_depth",

    # Firmware response
    "T",
    "seq",
    "sample_ms",
    "send_ms",
    "millis",

    # Wheel, encoder and voltage
    "L",
    "R",
    "enc_left",
    "enc_right",
    "v",

    # IMU attitude
    "r",
    "p",
    "y",

    # Accelerometer
    "ax",
    "ay",
    "az",

    # Gyroscope
    "gx",
    "gy",
    "gz",

    # Magnetometer
    "mx",
    "my",
    "mz",

    # Temperature
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


@dataclass(frozen=True)
class MotionStep:
    kind: str
    left_cmd: float
    right_cmd: float
    target: float
    label: str


@dataclass(frozen=True)
class MotionCalibration:
    encoder_counts_per_meter: float
    effective_track_width_m: float
    nominal_track_width_m: float

    @property
    def straight_target_counts(self) -> float:
        return self.encoder_counts_per_meter * STRAIGHT_DISTANCE_M

    @property
    def turn_target_counts(self) -> float:
        return (
            self.encoder_counts_per_meter
            * (3.141592653589793 * self.effective_track_width_m)
            * (TURN_DEGREES / 360.0)
        )

    def distance_target_counts(self, distance_m: float) -> float:
        return self.encoder_counts_per_meter * distance_m

    def turn_target_counts_for_degrees(self, turn_degrees: float) -> float:
        return (
            self.encoder_counts_per_meter
            * (3.141592653589793 * self.effective_track_width_m)
            * (turn_degrees / 360.0)
        )


DEFAULT_MOTION_CALIBRATION = MotionCalibration(
    encoder_counts_per_meter=UGV01_FIRMWARE_COUNTS_PER_METER,
    effective_track_width_m=0.141,
    nominal_track_width_m=0.141,
)


class MotionSequenceController:
    def __init__(self, calibration: MotionCalibration | None = None) -> None:
        self.calibration = calibration or DEFAULT_MOTION_CALIBRATION
        turn_counts = self.calibration.turn_target_counts
        straight_counts = self.calibration.straight_target_counts
        self.steps: list[MotionStep] = []
        for sequence_idx in range(SEQUENCE_REPEAT_COUNT):
            run_label = f"run {sequence_idx + 1}/{SEQUENCE_REPEAT_COUNT}"
            initial_hold_label = "initial hold" if sequence_idx == 0 else f"reset hold ({run_label})"
            self.steps.extend(
                [
                    MotionStep("hold", 0.0, 0.0, INITIAL_HOLD_SECONDS, initial_hold_label),
                    MotionStep("distance", *STRAIGHT_FORWARD_CMD, straight_counts, f"forward 1 m ({run_label})"),
                    MotionStep("hold", 0.0, 0.0, HOLD_SECONDS, f"hold after forward ({run_label})"),
                    MotionStep("distance", *STRAIGHT_BACKWARD_CMD, straight_counts, f"backward 1 m ({run_label})"),
                    MotionStep("hold", 0.0, 0.0, HOLD_SECONDS, f"hold after backward ({run_label})"),
                    MotionStep("turn", *TURN_CW_CMD, turn_counts, f"clockwise 360 ({run_label})"),
                    MotionStep("hold", 0.0, 0.0, HOLD_SECONDS, f"hold after clockwise turn ({run_label})"),
                ]
            )
            for repeat_idx in range(TURN_REPEAT_COUNT):
                self.steps.append(
                    MotionStep(
                        "turn",
                        *TURN_CCW_CMD,
                        turn_counts,
                        f"counterclockwise 360 #{repeat_idx + 1} ({run_label})",
                    )
                )
                self.steps.append(
                    MotionStep(
                        "hold",
                        0.0,
                        0.0,
                        HOLD_SECONDS,
                        f"hold after counterclockwise #{repeat_idx + 1} ({run_label})",
                    )
                )
        self.steps.append(MotionStep("hold", 0.0, 0.0, FINAL_HOLD_SECONDS, "final hold"))
        self.index = 0
        self.step_started_s: float | None = None
        self.start_left: int | None = None
        self.start_right: int | None = None
        self.completed = False

    @property
    def active_step(self) -> MotionStep | None:
        if self.completed or self.index >= len(self.steps):
            return None
        return self.steps[self.index]

    def update(self, telemetry: dict[str, Any], now_s: float) -> tuple[float, float, str]:
        step = self.active_step
        if step is None:
            return 0.0, 0.0, "done"

        left_count = int(float(telemetry.get("enc_left", 0)))
        right_count = int(float(telemetry.get("enc_right", 0)))

        if self.step_started_s is None:
            self.step_started_s = now_s
            self.start_left = left_count
            self.start_right = right_count

        assert self.step_started_s is not None
        assert self.start_left is not None
        assert self.start_right is not None

        if step.kind == "hold":
            if now_s - self.step_started_s >= step.target:
                self._advance(now_s, left_count, right_count)
                return self.update(telemetry, now_s)
            return step.left_cmd, step.right_cmd, step.label

        left_progress = abs(left_count - self.start_left)
        right_progress = abs(right_count - self.start_right)
        progress_counts = 0.5 * (left_progress + right_progress)

        if progress_counts >= step.target:
            self._advance(now_s, left_count, right_count)
            return self.update(telemetry, now_s)

        return step.left_cmd, step.right_cmd, step.label

    def _advance(self, now_s: float, left_count: int, right_count: int) -> None:
        self.index += 1
        if self.index >= len(self.steps):
            self.completed = True
            self.step_started_s = None
            self.start_left = None
            self.start_right = None
            return
        self.step_started_s = now_s
        self.start_left = left_count
        self.start_right = right_count


class SquareSequenceController(MotionSequenceController):
    def __init__(self, calibration: MotionCalibration | None = None) -> None:
        self.calibration = calibration or DEFAULT_MOTION_CALIBRATION
        straight_counts = self.calibration.distance_target_counts(
            SQUARE_SIDE_LENGTH_M
        )
        turn_counts = self.calibration.turn_target_counts_for_degrees(
            SQUARE_TURN_DEGREES
        )
        self.steps: list[MotionStep] = []
        for square_idx in range(SQUARE_REPEAT_COUNT):
            run_label = f"square {square_idx + 1}/{SQUARE_REPEAT_COUNT}"
            initial_hold_label = "initial hold" if square_idx == 0 else f"reset hold ({run_label})"
            self.steps.append(
                MotionStep("hold", 0.0, 0.0, INITIAL_HOLD_SECONDS, initial_hold_label)
            )
            for edge_idx in range(4):
                self.steps.extend(
                    [
                        MotionStep(
                            "distance",
                            *STRAIGHT_FORWARD_CMD,
                            straight_counts,
                            f"side {edge_idx + 1} forward {SQUARE_SIDE_LENGTH_M:.2f} m ({run_label})",
                        ),
                        MotionStep(
                            "hold",
                            0.0,
                            0.0,
                            HOLD_SECONDS,
                            f"hold after side {edge_idx + 1} ({run_label})",
                        ),
                    ]
                )
                if edge_idx < 3:
                    self.steps.extend(
                        [
                            MotionStep(
                                "turn",
                                *SQUARE_TURN_CW_CMD,
                                turn_counts,
                                f"clockwise {int(SQUARE_TURN_DEGREES)} deg corner {edge_idx + 1} ({run_label})",
                            ),
                            MotionStep(
                                "hold",
                                0.0,
                                0.0,
                                HOLD_SECONDS,
                                f"hold after corner {edge_idx + 1} ({run_label})",
                            ),
                        ]
                    )
        self.steps.append(MotionStep("hold", 0.0, 0.0, FINAL_HOLD_SECONDS, "final hold"))
        self.index = 0
        self.step_started_s: float | None = None
        self.start_left: int | None = None
        self.start_right: int | None = None
        self.completed = False


def build_motion_controller() -> MotionSequenceController:
    if MOTION_PLAN == "square_025m":
        return SquareSequenceController()
    return MotionSequenceController()


def query_telemetry() -> tuple[dict[str, Any], int, int, float]:
    """
    Request combined UGV01 telemetry using T:147.

    A short-lived HTTP connection is used so that the logger does not hold
    a persistent connection to the UGV01 web server while manual control is
    also active.
    """

    command = {"T": 147}
    send_ns = time.monotonic_ns()

    response = None

    try:
        response = requests.get(
            BASE_URL,
            params={
                "cmd": json.dumps(
                    command,
                    separators=(",", ":"),
                )
            },
            headers={
                "Connection": "close",
                "Cache-Control": "no-cache",
                "User-Agent": "UGV01-BenchLogger/2.0",
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )

        rx_ns = time.monotonic_ns()
        response.raise_for_status()

        text = response.text.strip()

        if not text:
            raise RuntimeError("UGV01 returned an empty response.")

        try:
            telemetry = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"UGV01 returned invalid JSON: {text[:200]}"
            ) from exc

        if not isinstance(telemetry, dict):
            raise RuntimeError(
                "UGV01 response was not a JSON object."
            )

        latency_ms = (rx_ns - send_ns) / 1_000_000.0

        return telemetry, send_ns, rx_ns, latency_ms

    finally:
        if response is not None:
            response.close()


def send_command(command: dict[str, Any]) -> None:
    response = None
    try:
        response = requests.get(
            BASE_URL,
            params={
                "cmd": json.dumps(
                    command,
                    separators=(",", ":"),
                )
            },
            headers={
                "Connection": "close",
                "Cache-Control": "no-cache",
                "User-Agent": "UGV01-BenchLogger/2.0",
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    finally:
        if response is not None:
            response.close()


def send_stop() -> None:
    send_command({"T": 1, "L": 0.0, "R": 0.0})


def verify_connection() -> None:
    """Check the station-mode connection before beginning the full log."""

    print(f"Checking UGV01 connection at {BASE_URL} ...")

    telemetry, _, _, latency_ms = query_telemetry()

    print("UGV01 connection successful.")
    print(f"Response T value: {telemetry.get('T')}")
    print(f"Initial HTTP latency: {latency_ms:.1f} ms")
    print()


def main() -> None:
    print("=" * 72)
    print("UGV01 T:147 BENCH LOGGER")
    print("=" * 72)
    print(f"Rover station IP: {ROVER_IP}")
    print(f"Telemetry endpoint: {BASE_URL}")
    print(f"Duration: {DURATION_SECONDS} seconds")
    print(f"Polling interval: {POLL_INTERVAL_SECONDS} seconds")
    print(f"HTTP timeout: {HTTP_TIMEOUT_SECONDS} seconds")
    print(f"Output: {OUTPUT_CSV}")
    print()
    print("The logger uses short-lived HTTP connections to reduce interference")
    print("with simultaneous browser-based rover control.")
    print('Command: {"T":147}')
    if MOTION_SCRIPT_ENABLED:
        print()
        print("Motion script is ENABLED.")
        if MOTION_PLAN == "square_025m":
            print(
                "Plan: 0.25 m square with 4 short sides and 90 deg clockwise "
                f"corners, repeated {SQUARE_REPEAT_COUNT} time(s)"
            )
        else:
            print(
                "Sequence: hold -> forward 1 m -> backward 1 m -> clockwise 360 "
                f"-> counterclockwise 360 x{TURN_REPEAT_COUNT}, repeated "
                f"{SEQUENCE_REPEAT_COUNT} times"
            )
        print(
            "Official UGV01 physical rail-center width from Waveshare docs: "
            f"{OFFICIAL_RAIL_CENTER_DISTANCE_M:.3f} m"
        )
        print(
            "Stock UGV01 firmware motion model: "
            f"{DEFAULT_MOTION_CALIBRATION.encoder_counts_per_meter:.1f} counts/m, "
            f"effective track width {DEFAULT_MOTION_CALIBRATION.effective_track_width_m:.4f} m"
        )
        print("Lift the tracks or clear the test area before starting.")
    print()

    try:
        verify_connection()

    except Exception as exc:
        print("Could not connect to the UGV01.")
        print(f"Error: {exc}")
        print()
        print("Check:")
        print(f"  1. OLED still shows ST: {ROVER_IP}")
        print("  2. Laptop and rover are on the same Wi-Fi network")
        print(f"  3. Run: ping {ROVER_IP}")
        print(f"  4. Open: http://{ROVER_IP}")
        return

    clock = SessionClockCalibrator()

    start_time = time.monotonic()
    end_time = start_time + DURATION_SECONDS
    next_poll_time = start_time

    sample_idx = 0
    successful_cycles = 0
    request_failure_count = 0
    sequence_gap_count = 0

    previous_rover_millis: int | None = None
    previous_sequence: int | None = None
    motion = build_motion_controller() if MOTION_SCRIPT_ENABLED else None

    try:
        with OUTPUT_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES,
                extrasaction="ignore",
            )

            writer.writeheader()
            file.flush()

            while time.monotonic() < end_time:
                cycle_start_ns = time.monotonic_ns()

                row: dict[str, Any] = {
                    key: None for key in FIELDNAMES
                }

                row["sample_idx"] = sample_idx
                row["cycle_ok"] = False
                row["error"] = ""
                row["t_wall_unix_s"] = time.time()
                row["t_cycle_start_ns"] = cycle_start_ns
                row["alarm_time_s"] = ""
                row["stale_packet"] = False
                row["queue_depth"] = 0

                try:
                    telemetry, send_ns, rx_ns, latency_ms = query_telemetry()

                    edge_mid_ns = (send_ns + rx_ns) // 2

                    row["t_edge_send_ns"] = send_ns
                    row["t_edge_rx_ns"] = rx_ns
                    row["t_edge_mid_ns"] = edge_mid_ns
                    row["http_latency_ms"] = latency_ms
                    row["edge_arrival_time_s"] = (
                        rx_ns / 1_000_000_000.0
                    )

                    # Copy matching telemetry fields into the CSV row.
                    for key, value in telemetry.items():
                        if key in row:
                            row[key] = value

                    # -----------------------------------------------------------
                    # Rover clock calibration
                    # -----------------------------------------------------------

                    rover_millis_raw = telemetry.get(
                        "sample_ms",
                        telemetry.get("millis"),
                    )

                    if rover_millis_raw is not None:
                        rover_millis = int(rover_millis_raw)
                        rover_millis_s = rover_millis / 1000.0
                        edge_mid_s = edge_mid_ns / 1_000_000_000.0

                        estimate = clock.observe(
                            rover_millis_s,
                            edge_mid_s,
                        )

                        row["rover_millis_s"] = rover_millis_s
                        row["source_sample_time_s"] = (
                            estimate.remote_time_s
                        )
                        row["clock_offset_s"] = estimate.offset_s
                        row["clock_calibrated"] = estimate.calibrated
                        row["clock_calibration_samples"] = (
                            estimate.samples
                        )

                        row["stale_packet"] = (
                            previous_rover_millis is not None
                            and rover_millis <= previous_rover_millis
                        )

                        previous_rover_millis = rover_millis

                    # The estimate is produced after receiving and processing
                    # the telemetry packet.
                    row["estimate_time_s"] = (
                        time.monotonic_ns() / 1_000_000_000.0
                    )

                    # -----------------------------------------------------------
                    # Sequence continuity
                    # -----------------------------------------------------------

                    sequence_raw = telemetry.get("seq")

                    if sequence_raw is not None:
                        sequence = int(sequence_raw)

                        if previous_sequence is not None:
                            if sequence > previous_sequence + 1:
                                missing = sequence - previous_sequence - 1
                                sequence_gap_count += missing

                        previous_sequence = sequence

                    row["request_failure_count"] = request_failure_count
                    row["sequence_gap_count"] = sequence_gap_count
                    row["packet_loss_count"] = (
                        request_failure_count + sequence_gap_count
                    )
                    row["cycle_ok"] = True

                    successful_cycles += 1

                    motion_label = ""
                    if motion is not None:
                        left_cmd, right_cmd, motion_label = motion.update(
                            telemetry,
                            time.monotonic(),
                        )
                        send_command(
                            {
                                "T": 1,
                                "L": left_cmd,
                                "R": right_cmd,
                            }
                        )

                    print(
                        f"{sample_idx:04d} | "
                        f"L={row['L']} R={row['R']} | "
                        f"enc=({row['enc_left']},{row['enc_right']}) | "
                        f"yaw={row['y']} | "
                        f"V={row['v']} | "
                        f"GPS={row['gps_valid']} "
                        f"sat={row['sat']} "
                        f"hdop={row['hdop']} | "
                        f"latency={latency_ms:.1f} ms"
                        + (f" | step={motion_label}" if motion_label else "")
                    )

                except Exception as exc:
                    request_failure_count += 1

                    failure_rx_ns = time.monotonic_ns()

                    row["t_edge_rx_ns"] = failure_rx_ns
                    row["edge_arrival_time_s"] = (
                        failure_rx_ns / 1_000_000_000.0
                    )
                    row["request_failure_count"] = request_failure_count
                    row["sequence_gap_count"] = sequence_gap_count
                    row["packet_loss_count"] = (
                        request_failure_count + sequence_gap_count
                    )
                    row["stale_packet"] = False
                    row["queue_depth"] = 0
                    row["error"] = str(exc)

                    print(
                        f"{sample_idx:04d} | DROP | {exc}"
                    )

                writer.writerow(row)
                file.flush()

                sample_idx += 1

                # Maintain a fixed polling schedule. The logger does not add
                # another full delay after the HTTP request finishes.
                next_poll_time += POLL_INTERVAL_SECONDS
                sleep_seconds = next_poll_time - time.monotonic()

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                else:
                    # If a request exceeded the interval, reset the schedule
                    # rather than creating a burst of back-to-back requests.
                    next_poll_time = time.monotonic()
    finally:
        if MOTION_SCRIPT_ENABLED:
            try:
                send_stop()
                print("Sent final stop command.")
            except Exception as exc:
                print(f"Warning: could not send final stop command: {exc}")

    print()
    print("=" * 72)
    print("LOGGING COMPLETE")
    print("=" * 72)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Total cycles: {sample_idx}")
    print(f"Successful cycles: {successful_cycles}")
    print(f"HTTP request failures: {request_failure_count}")
    print(f"Sequence-gap packets: {sequence_gap_count}")

    total_loss = request_failure_count + sequence_gap_count
    print(f"Total detected loss: {total_loss}")

    if sample_idx > 0:
        request_failure_rate = (
            100.0 * request_failure_count / sample_idx
        )

        print(
            f"HTTP request failure rate: "
            f"{request_failure_rate:.2f}%"
        )


if __name__ == "__main__":
    main()

import csv
import json
import threading
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

# Derived from your current Week 2 calibration pass.
ENCODER_COUNTS_PER_METER = 5632.373
EFFECTIVE_TRACK_WIDTH_M = 0.1818

# Asphalt-tuned commands: still moderate, but less under-rotating than the
# slippery-floor version.
STRAIGHT_FORWARD_CMD = (-0.20, -0.20)
STRAIGHT_BACKWARD_CMD = (0.20, 0.20)
TURN_CCW_CMD = (-0.32, 0.32)
TURN_CW_CMD = (0.32, -0.32)
SQUARE_STRAIGHT_FORWARD_CMD = (-0.14, -0.14)
# With the same timed turn, 0.072 produced 80 degrees and 0.080 produced
# 100 degrees. Linear interpolation gives 0.076 for a 90-degree corner.
SQUARE_TURN_CW_CMD = (0.076, -0.076)

STRAIGHT_DISTANCE_M = 1.0
TURN_DEGREES = 360.0
TURN_REPEAT_COUNT = 2
SEQUENCE_REPEAT_COUNT = 3
SQUARE_SIDE_LENGTH_M = 1.0
SQUARE_TURN_DEGREES = 90.0
SQUARE_REPEAT_COUNT = 1
SQUARE_SIDE_HOLD_SECONDS = 1.0
SQUARE_CORNER_HOLD_SECONDS = 3.0
SQUARE_STRAIGHT_WARMUP_CMD = (-0.08, -0.08)
SQUARE_STRAIGHT_WARMUP_SECONDS = 0.35
SQUARE_STRAIGHT_SECONDS = 4.2
SQUARE_TURN_SECONDS = 1.65
COMMAND_REFRESH_SECONDS = 0.10
HEADING_HOLD_GAIN = 0.012
HEADING_HOLD_MAX_CORRECTION = 0.05
HEADING_HOLD_DEADBAND_DEG = 3.0
HEADING_HOLD_STARTUP_SECONDS = 1.0
STRAIGHT_RAMP_SECONDS = 0.45
STRAIGHT_RAMP_MIN_SCALE = 0.45
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
    direction: str = ""


@dataclass(frozen=True)
class TimedCommandStep:
    duration_s: float
    left_cmd: float
    right_cmd: float
    label: str
    kind: str = "timed"
    target: float = 0.0
    direction: str = ""


@dataclass(frozen=True)
class MotionCalibration:
    encoder_counts_per_meter: float
    effective_track_width_m: float

    def distance_target_counts(self, distance_m: float) -> float:
        return self.encoder_counts_per_meter * distance_m

    def turn_target_counts(self, turn_degrees: float) -> float:
        return (
            self.encoder_counts_per_meter
            * (3.141592653589793 * self.effective_track_width_m)
            * (turn_degrees / 360.0)
        )


DEFAULT_MOTION_CALIBRATION = MotionCalibration(
    encoder_counts_per_meter=ENCODER_COUNTS_PER_METER,
    effective_track_width_m=EFFECTIVE_TRACK_WIDTH_M,
)

_telemetry_state_lock = threading.Lock()
_latest_yaw_deg: float | None = None


def normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg <= -180.0:
        angle_deg += 360.0
    while angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def angular_distance_deg(start_deg: float, current_deg: float) -> float:
    return abs(normalize_angle_deg(current_deg - start_deg))


def signed_yaw_delta_deg(start_deg: float, current_deg: float) -> float:
    return normalize_angle_deg(current_deg - start_deg)


def set_latest_yaw_deg(yaw_deg: float) -> None:
    global _latest_yaw_deg
    with _telemetry_state_lock:
        _latest_yaw_deg = yaw_deg


def get_latest_yaw_deg() -> float | None:
    with _telemetry_state_lock:
        return _latest_yaw_deg


def apply_heading_hold(
    left_cmd: float,
    right_cmd: float,
    target_yaw_deg: float,
    current_yaw_deg: float,
) -> tuple[float, float]:
    yaw_error_deg = normalize_angle_deg(current_yaw_deg - target_yaw_deg)
    if abs(yaw_error_deg) <= HEADING_HOLD_DEADBAND_DEG:
        return left_cmd, right_cmd
    correction = max(
        -HEADING_HOLD_MAX_CORRECTION,
        min(HEADING_HOLD_MAX_CORRECTION, HEADING_HOLD_GAIN * yaw_error_deg),
    )
    corrected_left = left_cmd + correction
    corrected_right = right_cmd - correction
    return corrected_left, corrected_right


def apply_startup_ramp(
    left_cmd: float,
    right_cmd: float,
    elapsed_s: float,
) -> tuple[float, float]:
    if elapsed_s >= STRAIGHT_RAMP_SECONDS:
        return left_cmd, right_cmd
    ramp_fraction = elapsed_s / STRAIGHT_RAMP_SECONDS if STRAIGHT_RAMP_SECONDS > 0 else 1.0
    scale = STRAIGHT_RAMP_MIN_SCALE + (1.0 - STRAIGHT_RAMP_MIN_SCALE) * ramp_fraction
    return left_cmd * scale, right_cmd * scale


class MotionSequenceController:
    def __init__(self, calibration: MotionCalibration | None = None) -> None:
        self.calibration = calibration or DEFAULT_MOTION_CALIBRATION
        turn_counts = self.calibration.turn_target_counts(TURN_DEGREES)
        straight_counts = self.calibration.distance_target_counts(
            STRAIGHT_DISTANCE_M
        )
        self.steps: list[MotionStep] = []
        for sequence_idx in range(SEQUENCE_REPEAT_COUNT):
            run_label = f"run {sequence_idx + 1}/{SEQUENCE_REPEAT_COUNT}"
            initial_hold_label = (
                "initial hold"
                if sequence_idx == 0
                else f"reset hold ({run_label})"
            )
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
        self.steps.append(
            MotionStep("hold", 0.0, 0.0, FINAL_HOLD_SECONDS, "final hold")
        )
        self.index = 0
        self.step_started_s: float | None = None
        self.start_left: int | None = None
        self.start_right: int | None = None
        self.start_yaw_deg: float | None = None
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
        yaw_deg = float(telemetry.get("y", 0.0))

        if self.step_started_s is None:
            self.step_started_s = now_s
            self.start_left = left_count
            self.start_right = right_count
            self.start_yaw_deg = yaw_deg

        assert self.step_started_s is not None
        assert self.start_left is not None
        assert self.start_right is not None
        assert self.start_yaw_deg is not None

        if step.kind == "hold":
            if now_s - self.step_started_s >= step.target:
                self._advance(now_s, left_count, right_count, yaw_deg)
                return self.update(telemetry, now_s)
            return step.left_cmd, step.right_cmd, step.label

        if step.kind == "turn_timed":
            if now_s - self.step_started_s >= step.target:
                self._advance(now_s, left_count, right_count, yaw_deg)
                return self.update(telemetry, now_s)
            return step.left_cmd, step.right_cmd, step.label

        if step.kind == "turn_yaw":
            if angular_distance_deg(self.start_yaw_deg, yaw_deg) >= step.target:
                self._advance(now_s, left_count, right_count, yaw_deg)
                return self.update(telemetry, now_s)
            return step.left_cmd, step.right_cmd, step.label

        left_progress = abs(left_count - self.start_left)
        right_progress = abs(right_count - self.start_right)
        progress_counts = 0.5 * (left_progress + right_progress)

        if progress_counts >= step.target:
            self._advance(now_s, left_count, right_count, yaw_deg)
            return self.update(telemetry, now_s)

        if step.kind == "distance":
            return step.left_cmd, step.right_cmd, step.label

        return step.left_cmd, step.right_cmd, step.label

    def _advance(
        self,
        now_s: float,
        left_count: int,
        right_count: int,
        yaw_deg: float,
    ) -> None:
        self.index += 1
        if self.index >= len(self.steps):
            self.completed = True
            self.step_started_s = None
            self.start_left = None
            self.start_right = None
            self.start_yaw_deg = None
            return
        self.step_started_s = now_s
        self.start_left = left_count
        self.start_right = right_count
        self.start_yaw_deg = yaw_deg


class SquareSequenceController(MotionSequenceController):
    def __init__(self, calibration: MotionCalibration | None = None) -> None:
        self.calibration = calibration or DEFAULT_MOTION_CALIBRATION
        straight_counts = self.calibration.distance_target_counts(
            SQUARE_SIDE_LENGTH_M
        )
        self.steps: list[MotionStep] = []
        for square_idx in range(SQUARE_REPEAT_COUNT):
            run_label = f"square {square_idx + 1}/{SQUARE_REPEAT_COUNT}"
            initial_hold_label = (
                "initial hold"
                if square_idx == 0
                else f"reset hold ({run_label})"
            )
            self.steps.append(
                MotionStep("hold", 0.0, 0.0, INITIAL_HOLD_SECONDS, initial_hold_label)
            )
            for edge_idx in range(4):
                self.steps.extend(
                    [
                        MotionStep(
                            "distance",
                            *SQUARE_STRAIGHT_FORWARD_CMD,
                            straight_counts,
                            f"side {edge_idx + 1} forward {SQUARE_SIDE_LENGTH_M:.2f} m ({run_label})",
                        ),
                        MotionStep(
                            "hold",
                            0.0,
                            0.0,
                            SQUARE_SIDE_HOLD_SECONDS,
                            f"hold after side {edge_idx + 1} ({run_label})",
                        ),
                    ]
                )
                if edge_idx < 3:
                    self.steps.extend(
                        [
                            MotionStep(
                                "turn_timed",
                                *SQUARE_TURN_CW_CMD,
                                SQUARE_TURN_SECONDS,
                                f"clockwise timed {int(SQUARE_TURN_DEGREES)} deg turn ({run_label}) corner {edge_idx + 1}",
                            ),
                            MotionStep(
                                "hold",
                                0.0,
                                0.0,
                                SQUARE_CORNER_HOLD_SECONDS,
                                f"hold after corner {edge_idx + 1} ({run_label})",
                            ),
                        ]
                    )
        self.steps.append(
            MotionStep("hold", 0.0, 0.0, FINAL_HOLD_SECONDS, "final hold")
        )
        self.index = 0
        self.step_started_s: float | None = None
        self.start_left: int | None = None
        self.start_right: int | None = None
        self.completed = False


class TimedSquarePlan:
    def __init__(self) -> None:
        self.steps: list[TimedCommandStep] = []
        for square_idx in range(SQUARE_REPEAT_COUNT):
            run_label = f"square {square_idx + 1}/{SQUARE_REPEAT_COUNT}"
            self.steps.append(
                TimedCommandStep(
                    INITIAL_HOLD_SECONDS if square_idx == 0 else SQUARE_CORNER_HOLD_SECONDS,
                    0.0,
                    0.0,
                    "initial hold" if square_idx == 0 else f"reset hold ({run_label})",
                    "timed",
                )
            )
            for edge_idx in range(4):
                self.steps.extend(
                    [
                        TimedCommandStep(
                            SQUARE_STRAIGHT_WARMUP_SECONDS,
                            *SQUARE_STRAIGHT_WARMUP_CMD,
                            f"side {edge_idx + 1} warmup ({run_label})",
                            "timed",
                        ),
                        TimedCommandStep(
                            SQUARE_STRAIGHT_SECONDS,
                            *SQUARE_STRAIGHT_FORWARD_CMD,
                            f"side {edge_idx + 1} forward {SQUARE_SIDE_LENGTH_M:.2f} m ({run_label})",
                            "timed",
                        ),
                        TimedCommandStep(
                            SQUARE_SIDE_HOLD_SECONDS,
                            0.0,
                            0.0,
                            f"hold after side {edge_idx + 1} ({run_label})",
                            "timed",
                        ),
                    ]
                )
                if edge_idx < 3:
                    self.steps.extend(
                        [
                            TimedCommandStep(
                                0.0,
                                *SQUARE_TURN_CW_CMD,
                                f"clockwise {int(SQUARE_TURN_DEGREES)} deg yaw corner {edge_idx + 1} ({run_label})",
                                "turn_yaw",
                                SQUARE_TURN_DEGREES,
                                "cw",
                            ),
                            TimedCommandStep(
                                SQUARE_CORNER_HOLD_SECONDS,
                                0.0,
                                0.0,
                                f"hold after corner {edge_idx + 1} ({run_label})",
                                "timed",
                            ),
                        ]
                    )
        self.steps.append(
            TimedCommandStep(FINAL_HOLD_SECONDS, 0.0, 0.0, "final hold", "timed")
        )


class MotionCommandStreamer:
    def __init__(self, plan: TimedSquarePlan) -> None:
        self.plan = plan
        self.current_label = "not started"
        self.completed = False
        self._current_step_index = 0
        self._current_step_started_s: float | None = None
        self._straight_target_yaw_deg: float | None = None
        self._turn_target_yaw_deg: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._current_step_started_s = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._current_step_started_s is not None
        while not self._stop_event.is_set():
            if self._current_step_index >= len(self.plan.steps):
                self.current_label = "done"
                self.completed = True
                break
            step = self.plan.steps[self._current_step_index]
            self.current_label = step.label
            now_s = time.monotonic()
            latest_yaw_deg = get_latest_yaw_deg()
            left_cmd = step.left_cmd
            right_cmd = step.right_cmd
            if self._current_step_started_s is None:
                self._current_step_started_s = now_s

            if self._current_step_started_s == now_s:
                if "warmup" in step.label or "forward" in step.label:
                    self._straight_target_yaw_deg = latest_yaw_deg
                elif step.kind == "turn_yaw":
                    self._turn_target_yaw_deg = latest_yaw_deg

            if (
                latest_yaw_deg is not None
                and self._straight_target_yaw_deg is not None
                and ("warmup" in step.label or "forward" in step.label)
                and now_s - self._current_step_started_s <= HEADING_HOLD_STARTUP_SECONDS
            ):
                left_cmd, right_cmd = apply_heading_hold(
                    step.left_cmd,
                    step.right_cmd,
                    self._straight_target_yaw_deg,
                    latest_yaw_deg,
                )

            advance_step = False
            if step.kind == "timed":
                advance_step = now_s - self._current_step_started_s >= step.duration_s
            elif (
                step.kind == "turn_yaw"
                and latest_yaw_deg is not None
                and self._turn_target_yaw_deg is not None
            ):
                signed_delta = normalize_angle_deg(latest_yaw_deg - self._turn_target_yaw_deg)
                if step.direction == "cw":
                    advance_step = signed_delta <= -step.target
                elif step.direction == "ccw":
                    advance_step = signed_delta >= step.target

            try:
                send_command({"T": 1, "L": left_cmd, "R": right_cmd})
            except Exception:
                self.current_label = f"{step.label} (command retry pending)"
            if advance_step:
                self._current_step_index += 1
                self._current_step_started_s = time.monotonic()
                self._straight_target_yaw_deg = None
                self._turn_target_yaw_deg = None
            time.sleep(COMMAND_REFRESH_SECONDS)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        send_stop()


def build_motion_controller() -> MotionSequenceController:
    if MOTION_PLAN == "square_1m":
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
        if MOTION_PLAN == "square_1m":
            print(
                f"Sequence: 1.0 m square with {SQUARE_TURN_SECONDS:g} s clockwise 90 deg corners, repeated {SQUARE_REPEAT_COUNT} times"
            )
        else:
            print(
                "Sequence: hold -> forward 1 m -> backward 1 m -> clockwise 360 "
                f"-> counterclockwise 360 x{TURN_REPEAT_COUNT}, repeated {SEQUENCE_REPEAT_COUNT} times"
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
                    if "y" in telemetry:
                        set_latest_yaw_deg(float(telemetry["y"]))

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

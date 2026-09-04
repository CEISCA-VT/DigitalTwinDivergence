"""Serve a lightweight visual dashboard for UGV01 digital-twin experiments.

The server supports three modes:

``replay``
    Existing accepted benign-log replay through the older GPS/EKF path.
``csv``
    A dummy/live-prototype stream driven from a T:147 CSV file.
``live``
    A live stream that polls the UGV01 firmware telemetry endpoint and propagates
    the same sensor-lightweight twin shown in the paper.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import mimetypes
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
import re
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from DigitalTwin.dashboard.contracts import ContractEngine, ResourcePolicy, load_contract_config
from DigitalTwin.analysis.real_data_study import AttackSpec, _prepare_run, replay
from DigitalTwin.detector import InnovationDetector
from DigitalTwin.kinematics import (
    UGV01_CARPET_DEVELOPMENT_CANDIDATE,
    DifferentialDriveGeometry,
    integrate_unicycle,
    wrap_angle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
MANIFEST_CANDIDATES = (
    REPO_ROOT / "results" / "benign_manifest.csv",
    REPO_ROOT / "DigitalTwin" / "datasets" / "analysis" / "real_data_study" / "benign_manifest.csv",
)

_CACHE: dict[tuple[str, int], dict[str, object]] = {}
_CACHE_LOCK = threading.Lock()
_STREAM: "TwinStream | None" = None

COMMAND_PRESETS = {
    "stop": (0.0, 0.0),
    "forward": (-0.28, -0.28),
    "reverse": (0.28, 0.28),
    "left": (0.22, -0.22),
    "right": (-0.22, 0.22),
    "forward_left": (-0.14, -0.28),
    "forward_right": (-0.28, -0.14),
    "reverse_left": (0.14, 0.28),
    "reverse_right": (0.28, 0.14),
}
SPEED_SCALE = {"slow": 0.65, "medium": 1.0, "fast": 1.35}


def sanitize_rover_url(value: str) -> str:
    """Accept plain URLs and common Markdown-pasted URL forms."""
    text = str(value).strip()
    markdown = re.search(r"\]\((https?://[^)]+)\)", text)
    if markdown:
        text = markdown.group(1)
    elif text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        raise ValueError("rover URL is empty")
    if not text.startswith(("http://", "https://")):
        text = f"http://{text}"
    return text


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _row_text(row: dict[str, object], key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    return str(value)


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", "None", "null", None}:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_number(row, key, float(default))))


def _boolean(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_get(url: str, timeout_s: float) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - local rover URL
        text = response.read().decode("utf-8", errors="replace").strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError("UGV01 response was not a JSON object")
    return payload


def build_rover_request_url(rover_url: str, payload: dict[str, object], mode: str) -> str:
    """Build the rover HTTP URL for direct streaming or legacy command polling."""
    sanitized = sanitize_rover_url(rover_url).rstrip("/")
    if mode == "stream":
        return sanitized
    if mode not in {"cmd", "json"}:
        raise ValueError(f"unsupported rover request mode: {mode}")
    separator = "&" if "?" in sanitized else "?"
    query = urllib.parse.urlencode({
        mode: json.dumps(payload, separators=(",", ":")),
    })
    return f"{sanitized}{separator}{query}"


class TwinStream:
    """Incrementally propagates the paper's sensor-lightweight UGV01 twin."""

    def __init__(
        self,
        *,
        mode: str,
        csv_path: Path | None,
        rover_url: str,
        poll_hz: float,
        rover_request_mode: str = "stream",
        policy: str = "contract-aware",
        stream_only: bool = False,
        contract_config_path: Path | None = None,
        output_dir: Path | None = None,
        experiment_metadata: dict[str, object] | None = None,
        duration_s: float | None = None,
        max_points: int = 2400,
    ) -> None:
        self.mode = mode
        self.csv_path = csv_path
        self.rover_url = sanitize_rover_url(rover_url).rstrip("/")
        self.rover_request_mode = rover_request_mode
        self.poll_hz = max(0.5, float(poll_hz))
        self.contract_config = load_contract_config(contract_config_path or Path(__file__).resolve().parents[1] / "configs" / "ugv01_live_service_contracts.json")
        self.contract_engine = ContractEngine(self.contract_config)
        self.resource_policy = ResourcePolicy(policy, self.contract_config)
        self.policy_name = policy
        self.stream_only = stream_only
        self.max_points = max_points
        self.geometry = DifferentialDriveGeometry(
            effective_track_width_m=(
                UGV01_CARPET_DEVELOPMENT_CANDIDATE.clockwise_effective_track_width_m
                + UGV01_CARPET_DEVELOPMENT_CANDIDATE.counterclockwise_effective_track_width_m
            )
            / 2.0
        )
        self.gyro_weight = UGV01_CARPET_DEVELOPMENT_CANDIDATE.gyro_weight
        self.distance_scale = UGV01_CARPET_DEVELOPMENT_CANDIDATE.distance_scale
        self.points: list[dict[str, object]] = []
        self.summary: dict[str, object] = {}
        self.bounds = {"min_x": -1.0, "max_x": 1.0, "min_y": -1.0, "max_y": 1.0}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = np.array([0.0, 0.0, 0.0], dtype=float)
        self._last_enc: tuple[int, int] | None = None
        self._last_sample_s: float | None = None
        self._last_seq: int | None = None
        self._elapsed_s = 0.0
        self._origin: tuple[float, float] | None = None
        self._gps_translation_offset: tuple[float, float] | None = None
        self._gps_heading_offset_rad: float | None = None
        self._clock_offset_s: float | None = None
        self._last_arrival_s: float | None = None
        self._last_source_for_jitter_s: float | None = None
        self._arrival_jitter_s = 0.0
        self._bytes_window: deque[tuple[float, int]] = deque()
        self._events: list[dict[str, object]] = []
        self._latest_contracts: list[dict[str, object]] = []
        self.experiment_metadata = experiment_metadata or {}
        output_root = output_dir or (REPO_ROOT / "raw_logs" / "live_validation")
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = str(self.experiment_metadata.get("run_label") or "").strip()
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix).strip("_")
        name = f"ugv01_live_contract_{suffix}_{stamp}.jsonl" if suffix else f"ugv01_live_contract_{stamp}.jsonl"
        self.log_path = output_root / name
        self._error = ""
        self._running = False
        self.duration_s = duration_s if duration_s is None else max(0.0, float(duration_s))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        target = self._run_csv if self.mode == "csv" else self._run_live
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def payload(self) -> dict[str, object]:
        with self._lock:
            points = list(self.points)
            summary = dict(self.summary)
            bounds = dict(self.bounds)
            error = self._error
            running = self._running
            events = list(self._events[-80:])
            contracts = list(self._latest_contracts)
            latest_aoi = float(points[-1]["aoi_s"]) if points else None
            policy_decision = self.resource_policy.snapshot(latest_aoi, contracts)
        return {
            "schema": "ugv01_live_twin_stream_v2",
            "mode": self.mode,
            "running": running,
            "error": error,
            "metadata": {
                "label": "CSV prototype" if self.mode == "csv" else "UGV01 live firmware",
                "source": str(self.csv_path) if self.csv_path else self.rover_url,
                "rover_request_mode": self.rover_request_mode,
                "stream_only": self.stream_only,
                "paper_role": "sensor-lightweight physical-virtual fidelity prototype",
                "runtime_inputs": "T:147 encoder counts, IMU yaw rate, firmware yaw, timing, optional GPS",
                "twin_model": "UGV01 deterministic tracked-drive propagation with gyro blending",
                "reference_note": "live GPS operational reference",
                "contract_provenance": self.contract_config["provenance"],
                "experiment": self.experiment_metadata,
            },
            "policy": {
                "name": self.policy_name,
                "resource_mode": self.resource_policy.mode,
                "requested_update_rate_hz": self.resource_policy.update_rate_hz,
                "log_path": display_path(self.log_path),
                "decision": policy_decision,
            },
            "contracts": contracts,
            "events": events,
            "summary": summary,
            "bounds": bounds,
            "points": points,
        }

    def csv_data(self) -> tuple[str, str]:
        """Return the currently collected live-twin samples as CSV."""
        with self._lock:
            points = list(self.points)

        if not points:
            raise RuntimeError("No telemetry samples have been collected yet.")

        # Collect all scalar point fields.
        fieldnames = []
        for point in points:
            for key in point.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        # Keep nested contract information in one CSV cell as JSON.
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for point in points:
            row = {}

            for key in fieldnames:
                value = point.get(key, "")

                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(
                        value,
                        separators=(",", ":"),
                        allow_nan=False,
                    )

                elif value is None:
                    value = ""

                row[key] = value

            writer.writerow(row)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ugv01_live_twin_{timestamp}.csv"

        return output.getvalue(), filename

    def _run_csv(self) -> None:
        assert self.csv_path is not None
        try:
            rows = _read_csv(self.csv_path)
            last_wall = time.monotonic()
            started_wall = time.monotonic()
            for row in rows:
                if self._stop.is_set():
                    break
                if self.duration_s is not None and time.monotonic() - started_wall >= self.duration_s:
                    break
                if _row_text(row, "cycle_ok", "True").lower() not in {"true", "1", "yes"}:
                    continue
                recorded_arrival = _number(row, "edge_arrival_time_s", math.nan)
                self._append_sample(
                    row,
                    edge_arrival_s=recorded_arrival if math.isfinite(recorded_arrival) else time.time(),
                    latency_ms=_number(row, "http_latency_ms"),
                )
                sample_s = self._sample_time(row)
                previous = self.points[-2]["source_time_s"] if len(self.points) > 1 else sample_s
                delay = max(0.0, min(0.25, float(sample_s) - float(previous)))
                now = time.monotonic()
                time.sleep(max(0.0, delay - (now - last_wall)))
                last_wall = time.monotonic()
        except Exception as exc:  # pragma: no cover - surfaced to UI
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._running = False

    def _run_live(self) -> None:
        while not self._stop.is_set():
            period = 1.0 / self.resource_policy.update_rate_hz
            started = time.monotonic()

            try:
                edge_send = time.time()
                row = _json_get(
                    build_rover_request_url(
                        self.rover_url,
                        {"T": 147},
                        self.rover_request_mode,
                    ),
                    timeout_s=1.0,
                )

                edge_arrival = time.time()

                row_bytes = len(
                    json.dumps(row, separators=(",", ":")).encode("utf-8")
                )

                self._append_sample(
                    row,
                    edge_arrival_s=edge_arrival,
                    latency_ms=(edge_arrival - edge_send) * 1000.0,
                    payload_bytes=row_bytes,
                )

                # Clear any previous transient communication error
                # after a successful T:147 request.
                with self._lock:
                    self._error = ""

            except Exception as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"

            elapsed = time.monotonic() - started
            time.sleep(max(0.0, period - elapsed))

    def send_drive_command(self, command: str, speed: str) -> dict[str, object]:
        command = command.strip().lower()
        speed = speed.strip().lower()
        if command not in COMMAND_PRESETS:
            raise ValueError(f"Unsupported command: {command}")
        if speed not in SPEED_SCALE:
            raise ValueError(f"Unsupported speed: {speed}")
        left, right = COMMAND_PRESETS[command]
        scale = SPEED_SCALE[speed]
        if command != "stop":
            left *= scale
            right *= scale
        payload = {"T": 1, "L": round(left, 3), "R": round(right, 3)}
        if self.mode != "live":
            return {"sent": False, "dry_run": True, "payload": payload, "mode": self.mode}
        if self.stream_only:
            return {
                "sent": False,
                "dry_run": True,
                "payload": payload,
                "mode": self.mode,
                "note": "Movement commands are disabled in stream-only dashboard mode.",
            }
        command_url = self.rover_url
        if self.rover_request_mode == "stream":
            parsed = urllib.parse.urlparse(command_url)
            command_url = urllib.parse.urlunparse(parsed._replace(path="/js", query=""))
        started = time.time()
        response = _json_get(build_rover_request_url(command_url, payload, "cmd"), timeout_s=1.0)
        return {
            "sent": True,
            "dry_run": False,
            "payload": payload,
            "response": response,
            "latency_ms": (time.time() - started) * 1000.0,
        }

    def _sample_time(self, row: dict[str, object]) -> float:
        for key in ("source_sample_time_s", "rover_millis_s"):
            value = _number(row, key, math.nan)
            if math.isfinite(value):
                return value
        for key in ("sample_ms", "millis"):
            value = _number(row, key, math.nan)
            if math.isfinite(value):
                return value / 1000.0
        return time.monotonic()

    def _append_sample(self, row: dict[str, object], *, edge_arrival_s: float, latency_ms: float, payload_bytes: int | None = None) -> None:
        evaluation_started = time.perf_counter()
        sample_s = self._sample_time(row)
        source_reset = self._last_sample_s is not None and sample_s < self._last_sample_s - 0.5
        if source_reset:
            with self._lock:
                self.points.clear()
                self._events = [
                    {"t": 0.0, "type": "source_session_reset", "from": "active", "to": "new_session", "reason": "source clock regressed"}
                ]
            self.contract_engine = ContractEngine(self.contract_config)
            self.resource_policy = ResourcePolicy(self.policy_name, self.contract_config)
            self._latest_contracts = []
            self._state = np.array([0.0, 0.0, 0.0], dtype=float)
            self._last_enc = None
            self._last_seq = None
            self._origin = None
            self._gps_translation_offset = None
            self._gps_heading_offset_rad = None
            self._clock_offset_s = None
            self._last_sample_s = None
            self._last_arrival_s = None
            self._last_source_for_jitter_s = None
            self._elapsed_s = 0.0
        seq = _integer(row, "seq", len(self.points))
        enc_left = _integer(row, "enc_left", _integer(row, "enc_l", 0))
        enc_right = _integer(row, "enc_right", _integer(row, "enc_r", 0))
        gyro_z_deg_s = _number(row, "gz", _number(row, "gyro_z", 0.0))
        firmware_yaw_deg = _number(row, "y", _number(row, "yaw", 0.0))
        gps_valid = _boolean(row, "gps_valid")
        lat = _number(row, "lat", math.nan)
        lon = _number(row, "lon", math.nan)
        gps_speed_mps = _number(row, "gps_speed_mps", _number(row, "speed_mps", math.nan))
        gps_course_deg = _number(row, "gps_course_deg", _number(row, "course_deg", math.nan))
        gps_age_s = max(0.0, _number(row, "gps_age_ms", 0.0) / 1000.0)

        observed_offset = edge_arrival_s - sample_s
        if self._clock_offset_s is None or observed_offset < self._clock_offset_s:
            self._clock_offset_s = observed_offset
        aoi_s = max(0.0, observed_offset - float(self._clock_offset_s))
        if self._last_arrival_s is not None and self._last_source_for_jitter_s is not None:
            self._arrival_jitter_s = abs(
                (edge_arrival_s - self._last_arrival_s) - (sample_s - self._last_source_for_jitter_s)
            )
        self._last_arrival_s = edge_arrival_s
        self._last_source_for_jitter_s = sample_s
        bytes_this_update = int(payload_bytes if payload_bytes is not None else len(json.dumps(row, default=str).encode("utf-8")))
        self._bytes_window.append((edge_arrival_s, bytes_this_update))
        while self._bytes_window and edge_arrival_s - self._bytes_window[0][0] > 5.0:
            self._bytes_window.popleft()
        window_span = max(1.0, edge_arrival_s - self._bytes_window[0][0]) if self._bytes_window else 1.0
        bytes_per_s = sum(item[1] for item in self._bytes_window) / window_span

        if self._last_sample_s is None:
            dt = 0.0
        else:
            dt = max(0.0, min(1.0, sample_s - self._last_sample_s))
        if self._last_enc is None:
            delta_left = 0
            delta_right = 0
        else:
            delta_left = enc_left - self._last_enc[0]
            delta_right = enc_right - self._last_enc[1]

        encoder_v, encoder_omega = self.geometry.ticks_to_control(delta_left, delta_right, dt)
        encoder_v *= self.distance_scale
        imu_omega = math.radians(gyro_z_deg_s)
        omega = (1.0 - self.gyro_weight) * encoder_omega + self.gyro_weight * imu_omega
        if dt > 0.0:
            self._state = integrate_unicycle(self._state, encoder_v, omega, dt)
            self._elapsed_s += dt

        gps_x = gps_y = None
        if gps_valid and math.isfinite(lat) and math.isfinite(lon):
            from DigitalTwin.telemetry import gps_to_local_xy

            if self._origin is None:
                self._origin = (lat, lon)
                self._gps_translation_offset = (float(self._state[0]), float(self._state[1]))
            gps_x, gps_y = gps_to_local_xy(lat, lon, self._origin[0], self._origin[1])
            if self._gps_translation_offset is not None:
                gps_x += self._gps_translation_offset[0]
                gps_y += self._gps_translation_offset[1]

        gps_agreement_m = None
        gps_heading_agreement_deg = None
        gps_heading_rad = None
        twin_global_heading = None
        if gps_x is not None and gps_y is not None:
            gps_agreement_m = math.hypot(float(self._state[0]) - gps_x, float(self._state[1]) - gps_y)
            if math.isfinite(gps_course_deg) and gps_course_deg >= 0.0 and gps_speed_mps >= 0.30:
                # NMEA course is clockwise from north. Convert to ENU heading.
                gps_heading_rad = wrap_angle(math.radians(90.0 - gps_course_deg))
                if self._gps_heading_offset_rad is None:
                    self._gps_heading_offset_rad = wrap_angle(gps_heading_rad - float(self._state[2]))
                twin_global_heading = wrap_angle(float(self._state[2]) + self._gps_heading_offset_rad)
                gps_heading_agreement_deg = abs(
                    math.degrees(wrap_angle(twin_global_heading - gps_heading_rad))
                )

        packet_gap = 0 if self._last_seq is None else max(0, seq - self._last_seq - 1)
        yaw_disagreement = abs(encoder_omega - imu_omega)
        slip_indicator = min(10.0, yaw_disagreement / 0.35)
        condition = self._condition_label(abs(encoder_v), abs(omega), yaw_disagreement)
        elapsed_s = self._elapsed_s
        point = {
            "t": elapsed_s,
            "source_time_s": sample_s,
            "edge_arrival_time_s": edge_arrival_s,
            "seq": seq,
            "twin_x": float(self._state[0]),
            "twin_y": float(self._state[1]),
            "twin_theta": float(self._state[2]),
            "gps_x": gps_x,
            "gps_y": gps_y,
            "gps_valid": gps_valid and gps_x is not None and gps_y is not None,
            "gps_speed_mps": gps_speed_mps if math.isfinite(gps_speed_mps) else None,
            "gps_course_deg": gps_course_deg if math.isfinite(gps_course_deg) and gps_course_deg >= 0.0 else None,
            "gps_heading_rad": gps_heading_rad,
            "twin_global_theta": twin_global_heading,
            "gps_age_s": gps_age_s,
            "aoi_s": aoi_s,
            "clock_offset_s": self._clock_offset_s,
            "arrival_jitter_ms": self._arrival_jitter_s * 1000.0,
            "gps_agreement_m": gps_agreement_m,
            "gps_heading_agreement_deg": gps_heading_agreement_deg,
            "firmware_yaw_deg": firmware_yaw_deg,
            "encoder_v": encoder_v,
            "encoder_omega": encoder_omega,
            "imu_omega": imu_omega,
            "omega": omega,
            "yaw_disagreement": yaw_disagreement,
            "slip_indicator": slip_indicator,
            "condition": condition,
            "latency_ms": latency_ms,
            "payload_bytes": bytes_this_update,
            "bytes_per_s": bytes_per_s,
            "packet_gap": packet_gap,
            "stale": _boolean(row, "stale_packet") or dt == 0.0,
            "queue_depth": _integer(row, "queue_depth", 0),
            "enc_left": enc_left,
            "enc_right": enc_right,
            "motor_l": _number(row, "L"),
            "motor_r": _number(row, "R"),
            "voltage": _number(row, "v", _number(row, "voltage", 0.0)),
            "satellites": _integer(row, "sat", _integer(row, "gps_satellites", 0)),
            "hdop": _number(row, "hdop", 99.99),
        }

        quality = self.contract_config["reference_quality"]
        point["contract_reference_valid"] = bool(
            point["gps_valid"]
            and point["satellites"] >= int(quality["minimum_satellites"])
            and point["hdop"] <= float(quality["maximum_hdop"])
            and gps_age_s <= float(quality["maximum_gps_age_s"])
            and gps_heading_rad is not None
            and twin_global_heading is not None
        )

        self._last_sample_s = sample_s
        self._last_enc = (enc_left, enc_right)
        self._last_seq = seq
        with self._lock:
            self.points.append(point)
            if len(self.points) > self.max_points:
                self.points = self.points[-self.max_points :]
            contracts, contract_events = self.contract_engine.evaluate(self.points)
            policy_event = self.resource_policy.update(elapsed_s, aoi_s, contracts)
            events = contract_events + ([policy_event] if policy_event else [])
            self._events.extend(events)
            self._events = self._events[-500:]
            self._latest_contracts = contracts
            point["contracts"] = contracts
            point["resource_mode"] = self.resource_policy.mode
            point["requested_update_rate_hz"] = self.resource_policy.update_rate_hz
            point["evaluation_ms"] = (time.perf_counter() - evaluation_started) * 1000.0
            self._update_summary_locked()
            self._write_log_record(point, events)

    def _write_log_record(self, point: dict[str, object], events: list[dict[str, object]]) -> None:
        record = {
            "schema": "ugv01_live_contract_record_v1",
            "policy": self.policy_name,
            "experiment": self.experiment_metadata,
            "point": point,
            "events": events,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    @staticmethod
    def _condition_label(speed: float, yaw_rate: float, disagreement: float) -> str:
        if disagreement > 0.45:
            return "high wheel-IMU disagreement"
        if yaw_rate > 0.55:
            return "turning"
        if speed > 0.12:
            return "translation"
        return "low-motion"

    def _update_summary_locked(self) -> None:
        points = self.points
        latencies = [float(p["latency_ms"]) for p in points]
        disagreements = [float(p["yaw_disagreement"]) for p in points]
        gps_points = [p for p in points if p["gps_valid"]]
        gps_agreements = [float(p["gps_agreement_m"]) for p in gps_points if p.get("gps_agreement_m") is not None]
        gps_heading_agreements = [
            float(p["gps_heading_agreement_deg"])
            for p in gps_points
            if p.get("gps_heading_agreement_deg") is not None
        ]
        xs = [float(p["twin_x"]) for p in points]
        ys = [float(p["twin_y"]) for p in points]
        xs.extend(float(p["gps_x"]) for p in gps_points)
        ys.extend(float(p["gps_y"]) for p in gps_points)
        pad = 0.35
        self.bounds = {
            "min_x": min(xs, default=-1.0) - pad,
            "max_x": max(xs, default=1.0) + pad,
            "min_y": min(ys, default=-1.0) - pad,
            "max_y": max(ys, default=1.0) + pad,
        }
        self.summary = {
            "updates": len(points),
            "duration_s": float(points[-1]["t"]) if points else 0.0,
            "gps_valid_fraction": len(gps_points) / len(points) if points else 0.0,
            "gps_valid_count": len(gps_points),
            "gps_agreement_rmse_m": (
                math.sqrt(sum(value * value for value in gps_agreements) / len(gps_agreements))
                if gps_agreements
                else None
            ),
            "gps_agreement_p95_m": _quantile(gps_agreements, 0.95) if gps_agreements else None,
            "gps_agreement_max_m": max(gps_agreements) if gps_agreements else None,
            "gps_heading_mae_deg": (
                sum(gps_heading_agreements) / len(gps_heading_agreements)
                if gps_heading_agreements
                else None
            ),
            "gps_heading_p95_deg": _quantile(gps_heading_agreements, 0.95) if gps_heading_agreements else None,
            **self._gps_relative_disagreement(points),
            "latency_median_ms": _quantile(latencies, 0.5),
            "latency_p95_ms": _quantile(latencies, 0.95),
            "packet_loss": sum(int(p["packet_gap"]) for p in points),
            "stale_packets": sum(int(bool(p["stale"])) for p in points),
            "max_queue_depth": max((int(p["queue_depth"]) for p in points), default=0),
            "yaw_disagreement_p95_deg_s": math.degrees(_quantile(disagreements, 0.95)),
            "aoi_p95_ms": 1000.0 * _quantile([float(p["aoi_s"]) for p in points], 0.95),
            "jitter_p95_ms": _quantile([float(p["arrival_jitter_ms"]) for p in points], 0.95),
            "bytes_per_s": float(points[-1].get("bytes_per_s", 0.0)) if points else 0.0,
            "evaluation_p95_ms": _quantile([float(p.get("evaluation_ms", 0.0)) for p in points], 0.95),
            "resource_mode": self.resource_policy.mode,
            "requested_update_rate_hz": self.resource_policy.update_rate_hz,
            "actual_update_rate_hz": (
                (len(points) - 1) / float(points[-1]["t"])
                if len(points) > 1 and float(points[-1]["t"]) > 0.0
                else 0.0
            ),
            "contract_qualified_count": sum(item["status"] == "qualified" for item in self._latest_contracts),
            "contract_at_risk_count": sum(item["status"] == "at_risk" for item in self._latest_contracts),
            "contract_withdrawn_count": sum(item["status"] == "withdrawn" for item in self._latest_contracts),
            "contract_unobservable_count": sum(item["status"] == "unobservable" for item in self._latest_contracts),
            "distance_m": self._path_length(points),
            "condition_mode": self._mode_label([str(p["condition"]) for p in points]),
            "fidelity_status": (
                "GPS operational fidelity active" if gps_points else "waiting for valid GPS reference"
            ),
        }

    @staticmethod
    def _gps_relative_disagreement(points: list[dict[str, object]]) -> dict[str, object]:
        """Compute online displacement disagreement without post-hoc alignment."""
        valid = [p for p in points if p.get("gps_valid")]
        output: dict[str, object] = {
            "gps_RPEp_1s_m": None,
            "gps_RPEp_5s_m": None,
            "gps_RPEp_10s_m": None,
        }
        for horizon in (1.0, 5.0, 10.0):
            errors: list[float] = []
            for index, start in enumerate(valid):
                target_t = float(start["t"]) + horizon
                candidates = valid[index + 1 :]
                if not candidates:
                    continue
                end = min(candidates, key=lambda item: abs(float(item["t"]) - target_t))
                if abs(float(end["t"]) - target_t) > 0.35:
                    continue
                gps_dx = float(end["gps_x"]) - float(start["gps_x"])
                gps_dy = float(end["gps_y"]) - float(start["gps_y"])
                twin_dx = float(end["twin_x"]) - float(start["twin_x"])
                twin_dy = float(end["twin_y"]) - float(start["twin_y"])
                errors.append(math.hypot(twin_dx - gps_dx, twin_dy - gps_dy))
            if errors:
                output[f"gps_RPEp_{int(horizon)}s_m"] = math.sqrt(
                    sum(value * value for value in errors) / len(errors)
                )
        return output

    @staticmethod
    def _path_length(points: list[dict[str, object]]) -> float:
        total = 0.0
        for before, after in zip(points, points[1:]):
            total += math.hypot(float(after["twin_x"]) - float(before["twin_x"]), float(after["twin_y"]) - float(before["twin_y"]))
        return total

    @staticmethod
    def _mode_label(values: list[str]) -> str:
        if not values:
            return "--"
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return max(counts, key=counts.get)


def _manifest_path() -> Path:
    for candidate in MANIFEST_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No benign_manifest.csv was found. Run the real-data study first.")


def list_logs() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for row in _read_csv(_manifest_path()):
        source = (REPO_ROOT / row["source_csv"]).resolve()
        try:
            source.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if not source.exists():
            continue
        run_id = row["run_id"]
        entries.append(
            {
                "id": run_id,
                "label": (
                    f"{row['speed'].title()} | {row['surface'].replace('_', ' ')} "
                    f"| trial {row['trial']}"
                ),
                "speed": row["speed"],
                "surface": row["surface"],
                "route": row["route"],
                "trial": int(row["trial"]),
                "rows": int(row["rows"]),
                "path": str(source.relative_to(REPO_ROOT)),
                "modified_ns": source.stat().st_mtime_ns,
            }
        )
    return sorted(
        entries,
        key=lambda item: (str(item["surface"]), str(item["speed"]), int(item["trial"])),
    )


def _resolve_run(run_id: str) -> tuple[dict[str, object], Path]:
    for entry in list_logs():
        if entry["id"] == run_id:
            return entry, (REPO_ROOT / str(entry["path"])).resolve()
    raise KeyError(f"Unknown run id: {run_id}")


def _filtered_source_rows(path: Path) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(path)
        if row.get("cycle_ok") == "True" and _boolean(row, "gps_valid")
    ]


def _heading_from_path(points: list[dict[str, object]], index: int) -> float:
    if not points:
        return 0.0
    current = points[index]
    for offset in range(1, min(8, len(points))):
        before = points[max(0, index - offset)]
        dx = float(current["ekf_x"]) - float(before["ekf_x"])
        dy = float(current["ekf_y"]) - float(before["ekf_y"])
        if math.hypot(dx, dy) > 0.015:
            return math.atan2(dy, dx)
    return float(current.get("ekf_theta", 0.0))


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability)) if values else 0.0


def build_replay_payload(run_id: str) -> dict[str, object]:
    entry, source = _resolve_run(run_id)
    cache_key = (run_id, source.stat().st_mtime_ns)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    prepared = _prepare_run(source)
    source_rows = prepared.rows
    # The previous run-level lock predates the independent security branch.
    # Show the nominal per-update chi-square limit until benign refreezing.
    threshold = InnovationDetector().threshold
    result = replay(
        source,
        "evidence_gated",
        AttackSpec(),
        threshold=threshold,
        prepared=prepared,
    )
    detector = InnovationDetector(threshold=threshold)

    points: list[dict[str, object]] = []
    for index, (replay_row, raw) in enumerate(zip(result.rows, source_rows)):
        innovation_x = float(result.innovations[index, 0])
        innovation_y = float(result.innovations[index, 1])
        detection = detector.evaluate(result.innovations[index], result.s_matrices[index])
        point = {
            "t": float(result.elapsed_s[index]),
            "seq": _integer(raw, "seq", index),
            "gps_x": float(result.clean_gps_xy[index, 0]),
            "gps_y": float(result.clean_gps_xy[index, 1]),
            "ekf_x": float(result.states_xy[index, 0]),
            "ekf_y": float(result.states_xy[index, 1]),
            "security_x": float(result.security_states_xy[index, 0]),
            "security_y": float(result.security_states_xy[index, 1]),
            "ekf_theta": 0.0,
            "innovation": math.hypot(innovation_x, innovation_y),
            "nis": float(result.scores[index]),
            "threshold": threshold,
            "epsilon_min": detection.epsilon_min_m,
            "confidence": detection.confidence,
            "region": detection.envelope_region,
            "q_trace": float(result.q_trace[index]),
            "s_trace": float(result.s_trace[index]),
            "velocity": float(prepared.controls[index, 0]),
            "omega": float(prepared.controls[index, 1]),
            "encoder_omega": float(prepared.encoder_controls[index, 1]),
            "imu_omega": float(prepared.corrected_gyro_radps[index]),
            "gyro_bias_deg_s": math.degrees(prepared.gyro_bias_radps),
            "yaw_disagreement": float(prepared.yaw_disagreement_radps[index]),
            "slip_indicator": float(prepared.slip_indicator[index]),
            "voltage": _number(raw, "v"),
            "motor_l": _number(raw, "L"),
            "motor_r": _number(raw, "R"),
            "yaw": _number(raw, "y"),
            "gyro_z": _number(raw, "gz"),
            "accel_z": _number(raw, "az"),
            "satellites": _integer(raw, "sat"),
            "hdop": _number(raw, "hdop", 99.99),
            "latency_ms": _number(raw, "http_latency_ms"),
            "packet_loss": _integer(raw, "packet_loss_count"),
            "stale": _boolean(raw, "stale_packet"),
            "queue_depth": _integer(raw, "queue_depth"),
            "gate_allowed": bool(int(replay_row["gate_allowed"])),
            "persistent_bias_score": float(replay_row["persistent_bias_score"]),
        }
        points.append(point)

    for index in range(len(points)):
        points[index]["path_heading"] = _heading_from_path(points, index)

    differences = [
        math.hypot(float(p["ekf_x"]) - float(p["gps_x"]), float(p["ekf_y"]) - float(p["gps_y"]))
        for p in points
    ]
    security_differences = [
        math.hypot(
            float(p["security_x"]) - float(p["gps_x"]),
            float(p["security_y"]) - float(p["gps_y"]),
        )
        for p in points
    ]
    latencies = [float(p["latency_ms"]) for p in points]
    hdops = [float(p["hdop"]) for p in points if float(p["hdop"]) < 90.0]
    satellites = [int(p["satellites"]) for p in points]
    xs = [float(p[key]) for p in points for key in ("gps_x", "ekf_x", "security_x")]
    ys = [float(p[key]) for p in points for key in ("gps_y", "ekf_y", "security_y")]
    pad = 0.25
    bounds = {
        "min_x": min(xs, default=-1.0) - pad,
        "max_x": max(xs, default=1.0) + pad,
        "min_y": min(ys, default=-1.0) - pad,
        "max_y": max(ys, default=1.0) + pad,
    }
    metadata = {
        **entry,
        "side_length_m": _number(source_rows[0], "square_side_length_m", 0.5) if source_rows else 0.5,
        "repeats": _integer(source_rows[0], "square_repeats", 3) if source_rows else 3,
        "network": source_rows[0].get("network_condition", "wifi_baseline") if source_rows else "",
    }
    summary = {
        "updates": len(points),
        "duration_s": float(points[-1]["t"]) if points else 0.0,
        "agreement_rmse_m": math.sqrt(sum(value * value for value in differences) / len(differences))
        if differences
        else 0.0,
        "agreement_median_m": _quantile(differences, 0.5),
        "agreement_p95_m": _quantile(differences, 0.95),
        "security_agreement_rmse_m": (
            math.sqrt(
                sum(value * value for value in security_differences)
                / len(security_differences)
            )
            if security_differences
            else 0.0
        ),
        "gate_pass_fraction": (
            sum(int(bool(p["gate_allowed"])) for p in points) / len(points)
            if points
            else 0.0
        ),
        "latency_median_ms": _quantile(latencies, 0.5),
        "latency_p95_ms": _quantile(latencies, 0.95),
        "packet_loss": sum(int(p["packet_loss"]) for p in points),
        "stale_packets": sum(int(bool(p["stale"])) for p in points),
        "max_queue_depth": max((int(p["queue_depth"]) for p in points), default=0),
        "satellite_min": min(satellites, default=0),
        "satellite_max": max(satellites, default=0),
        "hdop_median": _quantile(hdops, 0.5),
        "max_nis": max((float(p["nis"]) for p in points), default=0.0),
        "gyro_bias_deg_s": math.degrees(prepared.gyro_bias_radps),
        "slip_indicator_p95": _quantile(
            [float(p["slip_indicator"]) for p in points], 0.95
        ),
        "threshold_status": "provisional_per_update_chi_square_pending_benign_refreeze",
    }
    payload: dict[str, object] = {
        "schema": "ugv01_dashboard_replay_v2",
        "metadata": metadata,
        "summary": summary,
        "bounds": bounds,
        "points": points,
    }
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE[cache_key] = payload
    return payload


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "UGV01Dashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/logs":
                self._json({"logs": list_logs()})
                return
            if parsed.path == "/api/mode":
                self._json({"mode": _STREAM.mode if _STREAM is not None else "replay"})
                return
            if parsed.path == "/api/stream":
                if _STREAM is None:
                    self._json({"error": "Stream mode is not active"}, HTTPStatus.BAD_REQUEST)
                    return
                self._json(_STREAM.payload())
                return
            if parsed.path == "/api/download-csv":
                if _STREAM is None:
                    self._json(
                        {"error": "Live stream is not active"},
                        HTTPStatus.BAD_REQUEST,
                    )
                    return

                csv_text, filename = _STREAM.csv_data()
                body = csv_text.encode("utf-8")

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{filename}"',
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/replay":
                run_id = parse_qs(parsed.query).get("id", [""])[0]
                if not run_id:
                    self._json({"error": "Missing replay id"}, HTTPStatus.BAD_REQUEST)
                    return
                self._json(build_replay_payload(run_id))
                return
            self._static(parsed.path)
        except KeyError as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - surfaced to browser
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/api/command":
                self._json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("command request must be a JSON object")
            command = str(body.get("command", "stop"))
            speed = str(body.get("speed", "medium"))
            if _STREAM is None:
                self._json(
                    {
                        "sent": False,
                        "dry_run": True,
                        "mode": "replay",
                        "payload": {"T": 1, "L": 0.0, "R": 0.0},
                        "note": "Start the dashboard with --mode live to send rover movement commands.",
                    }
                )
                return
            self._json(_STREAM.send_drive_command(command, speed))
        except Exception as exc:  # pragma: no cover - surfaced to browser
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (STATIC_ROOT / relative).resolve()
        try:
            target.relative_to(STATIC_ROOT)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[dashboard] {format_string % args}")


def main() -> None:
    global _STREAM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser")
    parser.add_argument(
        "--mode",
        choices=("replay", "csv", "live"),
        default="replay",
        help="Data source for the dashboard",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "raw_logs" / "telemetry" / "ugv_t147_bench_20260814_143729.csv",
        help="T:147 CSV used by --mode csv",
    )
    parser.add_argument(
        "--rover-url",
        default="http://192.168.4.1/telemetry",
        help="UGV01 firmware URL used by --mode live",
    )
    parser.add_argument(
        "--rover-request-mode",
        choices=("stream", "cmd", "json"),
        default="stream",
        help=(
            "How live telemetry is requested: stream reads --rover-url directly; "
            "cmd/json append a legacy /js query argument."
        ),
    )
    parser.add_argument(
        "--stream-only",
        action="store_true",
        help="Hide dashboard movement controls and reject dashboard drive commands.",
    )
    parser.add_argument("--poll-hz", type=float, default=5.0, help="Live/csv stream polling rate")
    parser.add_argument(
        "--policy",
        choices=("static-low", "static-high", "aoi-only", "contract-aware"),
        default="contract-aware",
        help="Frozen live resource-allocation policy",
    )
    parser.add_argument("--contract-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-label", default="", help="Short run label added to live/csv JSONL filenames")
    parser.add_argument("--physical-condition", default="", help="Prospective experiment condition, for example static, transition, or turning")
    parser.add_argument("--wireless-condition", default="", help="Wireless condition, for example baseline or buffered")
    parser.add_argument("--trial", type=int, default=None, help="Prospective repetition number")
    parser.add_argument("--notes", default="", help="Short operator note stored in the JSONL metadata")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional live/csv trial duration before the stream stops automatically",
    )
    args = parser.parse_args()

    if args.mode in {"csv", "live"}:
        csv_path = (REPO_ROOT / args.csv).resolve() if not args.csv.is_absolute() else args.csv
        if args.mode == "csv" and not csv_path.is_file():
            raise FileNotFoundError(f"CSV stream source not found: {csv_path}")
        experiment_metadata = {
            "run_label": args.run_label,
            "physical_condition": args.physical_condition,
            "wireless_condition": args.wireless_condition,
            "trial": args.trial,
            "notes": args.notes,
        }
        _STREAM = TwinStream(
            mode=args.mode,
            csv_path=csv_path if args.mode == "csv" else None,
            rover_url=args.rover_url,
            rover_request_mode=args.rover_request_mode,
            poll_hz=args.poll_hz,
            policy=args.policy,
            stream_only=args.stream_only,
            contract_config_path=args.contract_config,
            output_dir=args.output_dir,
            experiment_metadata={key: value for key, value in experiment_metadata.items() if value not in {"", None}},
            duration_s=args.duration_s,
        )
        _STREAM.start()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"UGV01 digital-twin dashboard: {url}")
    print(f"mode={args.mode}")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    if args.mode in {"csv", "live"} and args.duration_s is not None:
        shutdown_delay = max(0.1, float(args.duration_s) + 1.0)
        threading.Timer(shutdown_delay, server.shutdown).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if _STREAM is not None:
            _STREAM.stop()
        server.server_close()


if __name__ == "__main__":
    main()

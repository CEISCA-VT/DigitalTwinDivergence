"""Serve a lightweight visual dashboard for accepted UGV01 telemetry logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from DigitalTwin.analysis.real_data_study import AttackSpec, _prepare_run, replay
from DigitalTwin.detector import InnovationDetector


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
MANIFEST_CANDIDATES = (
    REPO_ROOT / "results" / "benign_manifest.csv",
    REPO_ROOT / "DigitalTwin" / "datasets" / "analysis" / "real_data_study" / "benign_manifest.csv",
)

_CACHE: dict[tuple[str, int], dict[str, object]] = {}
_CACHE_LOCK = threading.Lock()


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"UGV01 digital-twin dashboard: {url}")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

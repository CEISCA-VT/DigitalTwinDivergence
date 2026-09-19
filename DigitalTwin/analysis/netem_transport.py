"""Capture measured UDP delivery traces through a Linux ``tc netem`` link.

The network setup itself lives in ``scripts/linux/run_netem_delivery_study.sh``.
This module deliberately does not invoke ``sudo``, ``ip``, or ``tc``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "ugv01_netem_transport_capture_v1"


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def send_packets(
    host: str,
    port: int,
    rate_hz: float,
    duration_s: float,
    output: Path,
    payload_bytes: int,
) -> None:
    if rate_hz <= 0 or duration_s <= 0:
        raise ValueError("rate_hz and duration_s must be positive")
    if payload_bytes < 128:
        raise ValueError("payload_bytes must be at least 128")

    period_ns = int(round(1_000_000_000 / rate_hz))
    packet_count = int(np.ceil(duration_s * rate_hz))
    rows: list[dict] = []
    padding = "x" * payload_bytes
    start_ns = time.monotonic_ns() + 200_000_000

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for packet_id in range(packet_count):
            deadline_ns = start_ns + packet_id * period_ns
            while True:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                time.sleep(min(remaining_ns / 1_000_000_000, 0.002))
            send_ns = time.monotonic_ns()
            message = {
                "kind": "state",
                "packet_id": packet_id,
                "source_elapsed_s": packet_id / rate_hz,
                "send_monotonic_ns": send_ns,
                "padding": padding,
            }
            payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
            sock.sendto(payload, (host, port))
            rows.append(
                {
                    "packet_id": packet_id,
                    "source_elapsed_s": packet_id / rate_hz,
                    "send_monotonic_ns": send_ns,
                    "datagram_bytes": len(payload),
                }
            )

        end = json.dumps({"kind": "end"}).encode("utf-8")
        for _ in range(10):
            sock.sendto(end, (host, port))
            time.sleep(0.02)

    _write_csv(
        output,
        ["packet_id", "source_elapsed_s", "send_monotonic_ns", "datagram_bytes"],
        rows,
    )


def receive_packets(bind: str, port: int, output: Path, idle_timeout_s: float) -> None:
    rows: list[dict] = []
    started = False
    last_packet_s = time.monotonic()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((bind, port))
        sock.settimeout(0.5)
        while True:
            try:
                payload, _ = sock.recvfrom(65_535)
            except socket.timeout:
                if started and time.monotonic() - last_packet_s >= idle_timeout_s:
                    break
                continue
            receive_ns = time.monotonic_ns()
            last_packet_s = time.monotonic()
            message = json.loads(payload.decode("utf-8"))
            if message.get("kind") == "end":
                # Jitter can reorder the marker ahead of a delayed state packet.
                # Drain until the idle timeout instead of truncating the capture.
                continue
            if message.get("kind") != "state":
                continue
            started = True
            rows.append(
                {
                    "packet_id": int(message["packet_id"]),
                    "source_elapsed_s": float(message["source_elapsed_s"]),
                    "send_monotonic_ns": int(message["send_monotonic_ns"]),
                    "receive_monotonic_ns": receive_ns,
                    "datagram_bytes": len(payload),
                }
            )

    _write_csv(
        output,
        [
            "packet_id",
            "source_elapsed_s",
            "send_monotonic_ns",
            "receive_monotonic_ns",
            "datagram_bytes",
        ],
        rows,
    )


def summarize_capture(
    sent_path: Path,
    received_path: Path,
    output_dir: Path,
    condition: str,
    replicate: int,
    rate_hz: float,
    delay_ms: float,
    jitter_ms: float,
    loss_percent: float,
    qdisc_record: Path | None = None,
) -> dict:
    sent = pd.read_csv(sent_path)
    received = pd.read_csv(received_path)
    if sent.empty:
        raise RuntimeError("capture contains no sent packets")
    if sent.packet_id.duplicated().any() or received.packet_id.duplicated().any():
        raise RuntimeError("capture has duplicate packet identifiers")
    merged = sent.merge(
        received[["packet_id", "receive_monotonic_ns"]],
        on="packet_id",
        how="left",
        validate="one_to_one",
    )
    merged["received"] = merged.receive_monotonic_ns.notna()
    merged["measured_delay_ms"] = (
        merged.receive_monotonic_ns - merged.send_monotonic_ns
    ) / 1_000_000
    received_delays = merged.loc[merged.received, "measured_delay_ms"].to_numpy(float)
    if not len(received_delays):
        raise RuntimeError("capture contains no received packets")
    duration_s = float(sent.source_elapsed_s.max()) + 1.0 / rate_hz
    received_order = received.sort_values("receive_monotonic_ns")
    out_of_order = int((np.diff(received_order.packet_id.to_numpy(int)) < 0).sum())
    summary = {
        "schema": SCHEMA,
        "condition": condition,
        "transport_replicate": replicate,
        "requested_rate_hz": rate_hz,
        "configured_delay_ms": delay_ms,
        "configured_jitter_ms": jitter_ms,
        "configured_loss_percent": loss_percent,
        "duration_s": duration_s,
        "sent_packets": int(len(sent)),
        "received_packets": int(merged.received.sum()),
        "lost_packets": int((~merged.received).sum()),
        "realized_loss_fraction": float((~merged.received).mean()),
        "realized_receive_rate_hz": float(merged.received.sum() / duration_s),
        "delay_p50_ms": float(np.quantile(received_delays, 0.50)),
        "delay_p95_ms": float(np.quantile(received_delays, 0.95)),
        "delay_max_ms": float(np.max(received_delays)),
        "out_of_order_transitions": out_of_order,
        "mean_datagram_bytes": float(sent.datagram_bytes.mean()),
        "clock": "Linux CLOCK_MONOTONIC sampled with Python time.monotonic_ns on one host",
        "transport": "UDP over veth pair shaped with Linux tc netem",
        "repository_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "repository_status": _command_output(["git", "status", "--porcelain"]),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if qdisc_record is not None:
        qdisc_text = qdisc_record.read_text(encoding="utf-8")
        summary["tc_qdisc_record"] = qdisc_text.strip()
        summary["tc_qdisc_sha256"] = hashlib.sha256(qdisc_text.encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "packet_delivery_ledger.csv", index=False)
    (output_dir / "capture_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sender = sub.add_parser("send")
    sender.add_argument("--host", required=True)
    sender.add_argument("--port", type=int, required=True)
    sender.add_argument("--rate-hz", type=float, required=True)
    sender.add_argument("--duration-s", type=float, required=True)
    sender.add_argument("--output", type=Path, required=True)
    sender.add_argument("--payload-bytes", type=int, default=1024)

    receiver = sub.add_parser("receive")
    receiver.add_argument("--bind", required=True)
    receiver.add_argument("--port", type=int, required=True)
    receiver.add_argument("--output", type=Path, required=True)
    receiver.add_argument("--idle-timeout-s", type=float, default=5.0)

    summary = sub.add_parser("summarize")
    summary.add_argument("--sent", type=Path, required=True)
    summary.add_argument("--received", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--condition", required=True)
    summary.add_argument("--replicate", type=int, required=True)
    summary.add_argument("--rate-hz", type=float, required=True)
    summary.add_argument("--delay-ms", type=float, required=True)
    summary.add_argument("--jitter-ms", type=float, required=True)
    summary.add_argument("--loss-percent", type=float, required=True)
    summary.add_argument("--qdisc-record", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "send":
        send_packets(args.host, args.port, args.rate_hz, args.duration_s, args.output, args.payload_bytes)
    elif args.command == "receive":
        receive_packets(args.bind, args.port, args.output, args.idle_timeout_s)
    else:
        summarize_capture(
            args.sent,
            args.received,
            args.output_dir,
            args.condition,
            args.replicate,
            args.rate_hz,
            args.delay_ms,
            args.jitter_ms,
            args.loss_percent,
            args.qdisc_record,
        )


if __name__ == "__main__":
    main()

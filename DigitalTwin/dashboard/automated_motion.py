"""Run repeatable UGV01 motion scripts through the dashboard command API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class MotionStep:
    command: str
    duration_s: float
    label: str


PROFILES: dict[str, list[MotionStep]] = {
    "stop_only": [
        MotionStep("stop", 0.2, "stop"),
    ],
    "compact_validation": [
        MotionStep("stop", 6.0, "initial stillness"),
        MotionStep("forward", 8.0, "forward segment"),
        MotionStep("stop", 3.0, "forward hold"),
        MotionStep("reverse", 8.0, "reverse segment"),
        MotionStep("stop", 3.0, "reverse hold"),
        MotionStep("forward_left", 6.0, "left curve"),
        MotionStep("stop", 2.0, "curve hold"),
        MotionStep("forward_right", 6.0, "right curve"),
        MotionStep("stop", 3.0, "curve hold"),
        MotionStep("right", 4.0, "clockwise turn"),
        MotionStep("stop", 3.0, "turn hold"),
        MotionStep("left", 4.0, "counterclockwise turn"),
        MotionStep("stop", 6.0, "final stillness"),
    ],
    "turning_intensive": [
        MotionStep("stop", 6.0, "initial stillness"),
        MotionStep("forward", 6.0, "forward segment"),
        MotionStep("stop", 2.0, "hold"),
        MotionStep("right", 4.0, "clockwise turn 1"),
        MotionStep("stop", 2.0, "hold"),
        MotionStep("forward", 5.0, "short translation"),
        MotionStep("stop", 2.0, "hold"),
        MotionStep("right", 4.0, "clockwise turn 2"),
        MotionStep("stop", 2.0, "hold"),
        MotionStep("left", 4.0, "counterclockwise turn 1"),
        MotionStep("stop", 2.0, "hold"),
        MotionStep("forward_left", 5.0, "left curve"),
        MotionStep("stop", 2.0, "hold"),
        MotionStep("forward_right", 5.0, "right curve"),
        MotionStep("stop", 6.0, "final stillness"),
    ],
    "surface_transition": [
        MotionStep("stop", 6.0, "initial stillness"),
        MotionStep("forward", 10.0, "forward transition"),
        MotionStep("stop", 3.0, "transition hold"),
        MotionStep("reverse", 10.0, "reverse transition"),
        MotionStep("stop", 3.0, "transition hold"),
        MotionStep("forward_left", 5.0, "gentle left curve"),
        MotionStep("stop", 2.0, "hold"),
        MotionStep("forward_right", 5.0, "gentle right curve"),
        MotionStep("stop", 6.0, "final stillness"),
    ],
    "stationary_low_motion": [
        MotionStep("stop", 10.0, "initial stillness"),
        MotionStep("forward", 3.0, "short forward check"),
        MotionStep("stop", 6.0, "stillness"),
        MotionStep("reverse", 3.0, "short reverse check"),
        MotionStep("stop", 10.0, "final stillness"),
    ],
}


def scaled_plan(profile: str, duration_s: float | None) -> list[MotionStep]:
    steps = PROFILES[profile]
    if duration_s is None:
        return steps
    total = sum(step.duration_s for step in steps)
    if total <= 0.0:
        return steps
    scale = max(0.1, float(duration_s) / total)
    return [MotionStep(step.command, step.duration_s * scale, step.label) for step in steps]


def post_command(dashboard_url: str, command: str, speed: str, timeout_s: float) -> dict[str, object]:
    body = json.dumps({"command": command, "speed": speed}, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{dashboard_url.rstrip('/')}/api/command",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - local dashboard URL
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("dashboard command response was not a JSON object")
    return payload


def wait_for_dashboard(dashboard_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{dashboard_url.rstrip('/')}/api/mode", timeout=1.0) as response:  # noqa: S310
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"dashboard did not become ready at {dashboard_url}: {last_error}")


def run_motion(
    *,
    dashboard_url: str,
    profile: str,
    speed: str,
    duration_s: float | None,
    command_timeout_s: float,
) -> None:
    wait_for_dashboard(dashboard_url, timeout_s=20.0)
    try:
        for step in scaled_plan(profile, duration_s):
            print(f"{step.label}: {step.command} for {step.duration_s:.1f} s")
            response = post_command(dashboard_url, step.command, speed, command_timeout_s)
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            time.sleep(max(0.0, step.duration_s))
    finally:
        try:
            post_command(dashboard_url, "stop", speed, command_timeout_s)
        except Exception as exc:  # pragma: no cover - best-effort safety stop
            print(f"warning: final stop command failed: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8765")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="compact_validation")
    parser.add_argument("--speed", choices=("slow", "medium", "fast"), default="slow")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--command-timeout-s", type=float, default=2.0)
    args = parser.parse_args()
    run_motion(
        dashboard_url=args.dashboard_url,
        profile=args.profile,
        speed=args.speed,
        duration_s=args.duration_s,
        command_timeout_s=args.command_timeout_s,
    )


if __name__ == "__main__":
    main()

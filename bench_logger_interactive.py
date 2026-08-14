"""Interactive UGV01 T:147 logger with menu-driven motion commands.

This keeps the same telemetry CSV path/schema as ``bench_logger.py`` but lets
the operator choose the next motion after each completed motion segment.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bench_logger


DISTANCE_PRESETS_M = {
    "1": 0.25,
    "2": 0.50,
    "3": 0.75,
    "4": 1.00,
}

CURVE_RADIUS_PRESETS_M = {
    "1": 0.25,
    "2": 0.50,
    "3": 0.75,
    "4": 1.00,
}

ANGLE_PRESETS_DEG = {
    "1": 45.0,
    "2": 90.0,
    "3": 180.0,
    "4": 360.0,
}

INTERACTIVE_SPEED_PROFILES = {
    "low": {
        "forward_cmd": 0.10,
        "reverse_cmd": 0.10,
        "turn_cmd": 0.038,
        "note": "matches the current low-speed square straight command and slow 90-degree turn setting",
    },
    "medium": {
        "forward_cmd": 0.20,
        "reverse_cmd": 0.20,
        "turn_cmd": 0.045,
        "note": "medium straight command with a modest turn command; use only after a lifted/safe check",
    },
}


@dataclass
class InteractiveStep:
    kind: str
    left_cmd: float
    right_cmd: float
    target: float
    label: str


class InteractiveMotionController:
    def __init__(
        self,
        *,
        forward_cmd: float,
        reverse_cmd: float,
        turn_cmd: float,
        turn_control: str,
        turn_counts_per_90: float,
        turn_target_scale: float,
        min_turn_stop_target_deg: float,
        max_turn_chunk_deg: float,
        hold_after_s: float,
    ) -> None:
        self.forward_cmd = abs(forward_cmd)
        self.reverse_cmd = abs(reverse_cmd)
        self.turn_cmd = abs(turn_cmd)
        self.speed_profile_name = "custom"
        self.turn_control = turn_control
        self.turn_counts_per_90 = turn_counts_per_90
        self.turn_target_scale = turn_target_scale
        self.min_turn_stop_target_deg = min_turn_stop_target_deg
        self.max_turn_chunk_deg = max_turn_chunk_deg
        self.hold_after_s = hold_after_s
        self.calibration = bench_logger.DEFAULT_MOTION_CALIBRATION

        self.step: InteractiveStep | None = None
        self.pending_steps: deque[InteractiveStep] = deque()
        self.step_started_s: float | None = None
        self.start_left: int | None = None
        self.start_right: int | None = None
        self.start_yaw_deg: float | None = None
        self.previous_yaw_deg: float | None = None
        self.accumulated_turn_deg = 0.0
        self.hold_until_s: float | None = None
        self.completed = False
        self.motion_index = 0

    def update(self, telemetry: dict[str, Any], now_s: float) -> tuple[float, float, str]:
        if self.completed:
            return 0.0, 0.0, "done"

        if self.hold_until_s is not None:
            if now_s < self.hold_until_s:
                return 0.0, 0.0, "operator hold"
            self.hold_until_s = None

        if self.step is None:
            if self.pending_steps:
                self.step = self.pending_steps.popleft()
            else:
                self.step = self._prompt_next_step()
            if self.step is None:
                self.completed = True
                return 0.0, 0.0, "done"
            self.step_started_s = None

        left_count = int(float(telemetry.get("enc_left", 0)))
        right_count = int(float(telemetry.get("enc_right", 0)))
        yaw_deg = float(telemetry.get("y", 0.0))

        if self.step_started_s is None:
            self.step_started_s = now_s
            self.start_left = left_count
            self.start_right = right_count
            self.start_yaw_deg = yaw_deg
            self.previous_yaw_deg = yaw_deg
            self.accumulated_turn_deg = 0.0
            print(f"Starting: {self.step.label}")

        assert self.start_left is not None
        assert self.start_right is not None
        assert self.start_yaw_deg is not None

        if self.step.kind == "distance":
            left_progress = abs(left_count - self.start_left)
            right_progress = abs(right_count - self.start_right)
            progress_counts = 0.5 * (left_progress + right_progress)
            if progress_counts >= self.step.target:
                self._finish_step(now_s)
                return 0.0, 0.0, "finished distance"
            return self.step.left_cmd, self.step.right_cmd, self.step.label

        if self.step.kind == "turn_yaw":
            assert self.previous_yaw_deg is not None
            yaw_delta = bench_logger.normalize_angle_deg(yaw_deg - self.previous_yaw_deg)
            self.previous_yaw_deg = yaw_deg
            direction = -1.0 if self.step.left_cmd > self.step.right_cmd else 1.0
            self.accumulated_turn_deg = max(
                0.0,
                self.accumulated_turn_deg + direction * yaw_delta,
            )
            yaw_progress = self.accumulated_turn_deg
            if yaw_progress >= self.step.target:
                self._finish_step(now_s)
                return 0.0, 0.0, "finished turn"
            return self.step.left_cmd, self.step.right_cmd, self.step.label

        if self.step.kind == "turn_encoder":
            left_progress = abs(left_count - self.start_left)
            right_progress = abs(right_count - self.start_right)
            progress_counts = 0.5 * (left_progress + right_progress)
            if progress_counts >= self.step.target:
                self._finish_step(now_s)
                return 0.0, 0.0, "finished turn"
            return self.step.left_cmd, self.step.right_cmd, self.step.label

        self._finish_step(now_s)
        return 0.0, 0.0, "finished"

    def _finish_step(self, now_s: float) -> None:
        assert self.step is not None
        print(f"Completed: {self.step.label}")
        self.step = None
        self.step_started_s = None
        self.start_left = None
        self.start_right = None
        self.start_yaw_deg = None
        self.previous_yaw_deg = None
        self.accumulated_turn_deg = 0.0
        self.hold_until_s = now_s + self.hold_after_s

    def _prompt_next_step(self) -> InteractiveStep | None:
        self.motion_index += 1
        try:
            bench_logger.send_stop()
        except Exception as exc:
            print(f"Warning: could not send stop before prompt: {exc}")

        print()
        print("=" * 72)
        print(f"Choose next motion #{self.motion_index}")
        print("  f = forward distance")
        print("  r = reverse distance")
        print("  lc = left/anti-clockwise curve")
        print("  rc = right/clockwise curve")
        print("  c = clockwise turn")
        print("  a = anti-clockwise turn")
        print("  v = change speed profile")
        print("  s = stop/hold and ask again")
        print("  q = finish logging")
        choice = input("Motion [f/r/lc/rc/c/a/v/s/q]: ").strip().lower()

        if choice in {"q", "quit", "done", "exit"}:
            return None
        if choice in {"s", "stop", "hold", ""}:
            return InteractiveStep("hold", 0.0, 0.0, 0.0, "operator stop")
        if choice in {"v", "speed", "profile"}:
            self._prompt_speed_profile()
            return InteractiveStep("hold", 0.0, 0.0, 0.0, "speed profile change")
        if choice in {"f", "forward"}:
            distance_m = _prompt_distance_m()
            forward_cmd, _, _ = self._prompt_motion_speed(default_kind="forward")
            target_counts = self.calibration.distance_target_counts(distance_m)
            cmd = -forward_cmd
            return InteractiveStep(
                "distance",
                cmd,
                cmd,
                target_counts,
                f"forward {distance_m:g} m at cmd {forward_cmd:g}",
            )
        if choice in {"r", "reverse", "back", "backward"}:
            distance_m = _prompt_distance_m()
            _, reverse_cmd, _ = self._prompt_motion_speed(default_kind="reverse")
            target_counts = self.calibration.distance_target_counts(distance_m)
            cmd = reverse_cmd
            return InteractiveStep(
                "distance",
                cmd,
                cmd,
                target_counts,
                f"reverse {distance_m:g} m at cmd {reverse_cmd:g}",
            )
        if choice in {"lc", "leftcurve", "left", "curveleft"}:
            radius_m = _prompt_curve_radius_m()
            angle_deg = _prompt_angle_deg()
            return self._make_curve_step(radius_m, angle_deg, clockwise=False)
        if choice in {"rc", "rightcurve", "right", "curveright"}:
            radius_m = _prompt_curve_radius_m()
            angle_deg = _prompt_angle_deg()
            return self._make_curve_step(radius_m, angle_deg, clockwise=True)
        if choice in {"c", "cw", "clockwise"}:
            angle_deg = _prompt_angle_deg()
            return self._make_turn_steps(angle_deg, clockwise=True)
        if choice in {"a", "ccw", "anticlockwise", "counterclockwise"}:
            angle_deg = _prompt_angle_deg()
            return self._make_turn_steps(angle_deg, clockwise=False)

        print(f"Unknown choice: {choice!r}; holding.")
        return InteractiveStep("hold", 0.0, 0.0, 0.0, "operator stop")

    def _make_turn_steps(self, requested_deg: float, *, clockwise: bool) -> InteractiveStep:
        chunk_count = max(1, int((requested_deg + self.max_turn_chunk_deg - 1e-9) // self.max_turn_chunk_deg))
        chunk_requested = requested_deg / chunk_count
        target_deg = max(self.min_turn_stop_target_deg, chunk_requested * self.turn_target_scale)
        target_counts = self.turn_counts_per_90 * (chunk_requested / 90.0)
        left_cmd = self.turn_cmd if clockwise else -self.turn_cmd
        right_cmd = -self.turn_cmd if clockwise else self.turn_cmd
        direction_label = "clockwise" if clockwise else "anti-clockwise"
        kind = "turn_encoder" if self.turn_control == "encoder" else "turn_yaw"
        target = target_counts if kind == "turn_encoder" else target_deg

        steps = [
            InteractiveStep(
                kind,
                left_cmd,
                right_cmd,
                target,
                (
                    f"{direction_label} {requested_deg:g} deg part {idx + 1}/{chunk_count} "
                    f"(requested chunk {chunk_requested:g} deg, "
                    f"stop target {target_counts:g} counts)"
                    if kind == "turn_encoder"
                    else (
                        f"{direction_label} {requested_deg:g} deg part {idx + 1}/{chunk_count} "
                        f"(requested chunk {chunk_requested:g} deg, stop target {target_deg:g} deg)"
                    )
                ),
            )
            for idx in range(chunk_count)
        ]
        self.pending_steps.extend(steps[1:])
        return steps[0]

    def _make_curve_step(
        self,
        radius_m: float,
        angle_deg: float,
        *,
        clockwise: bool,
    ) -> InteractiveStep:
        half_width_m = 0.5 * bench_logger.EFFECTIVE_TRACK_WIDTH_M
        if radius_m <= half_width_m:
            raise ValueError(
                "curve radius must be larger than half the effective track width "
                f"({half_width_m:.3f} m)"
            )

        inner_scale = (radius_m - half_width_m) / radius_m
        outer_scale = (radius_m + half_width_m) / radius_m
        base = self.forward_cmd
        if clockwise:
            # UGV01 T:147 forward commands are negative; on this tracked rover
            # the observed curve direction is opposite the naive differential
            # drive sign convention, so right/clockwise uses the right track as
            # the outer/faster track.
            left_cmd = -base * inner_scale
            right_cmd = -base * outer_scale
            direction_label = "right/clockwise"
        else:
            left_cmd = -base * outer_scale
            right_cmd = -base * inner_scale
            direction_label = "left/anti-clockwise"

        arc_length_m = radius_m * abs(angle_deg) * 3.141592653589793 / 180.0
        target_counts = self.calibration.distance_target_counts(arc_length_m)
        return InteractiveStep(
            "distance",
            left_cmd,
            right_cmd,
            target_counts,
            (
                f"{direction_label} curve radius {radius_m:g} m "
                f"angle {angle_deg:g} deg"
            ),
        )

    def _prompt_speed_profile(self) -> None:
        print("Speed profiles:")
        for idx, name in enumerate(sorted(INTERACTIVE_SPEED_PROFILES), start=1):
            profile = INTERACTIVE_SPEED_PROFILES[name]
            print(
                f"  {idx} = {name} "
                f"(forward={profile['forward_cmd']}, "
                f"reverse={profile['reverse_cmd']}, turn={profile['turn_cmd']})"
            )
        raw = input("Speed [low/medium or 1/2]: ").strip().lower()
        names = sorted(INTERACTIVE_SPEED_PROFILES)
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            raw = names[int(raw) - 1]
        if raw not in INTERACTIVE_SPEED_PROFILES:
            print(f"Unknown speed profile {raw!r}; keeping current speed.")
            return
        profile = INTERACTIVE_SPEED_PROFILES[raw]
        self.forward_cmd = abs(float(profile["forward_cmd"]))
        self.reverse_cmd = abs(float(profile["reverse_cmd"]))
        self.turn_cmd = abs(float(profile["turn_cmd"]))
        self.speed_profile_name = raw
        print(
            f"Changed speed to {raw}: "
            f"forward={self.forward_cmd:g}, "
            f"reverse={self.reverse_cmd:g}, "
            f"turn={self.turn_cmd:g}"
        )

    def _prompt_motion_speed(self, *, default_kind: str) -> tuple[float, float, float]:
        print("Speed for this motion:")
        print(
            f"  enter = current ({self.speed_profile_name}: "
            f"forward={self.forward_cmd:g}, reverse={self.reverse_cmd:g}, "
            f"turn={self.turn_cmd:g})"
        )
        names = sorted(INTERACTIVE_SPEED_PROFILES)
        for idx, name in enumerate(names, start=1):
            profile = INTERACTIVE_SPEED_PROFILES[name]
            print(
                f"  {idx} = {name} "
                f"(forward={profile['forward_cmd']}, "
                f"reverse={profile['reverse_cmd']}, turn={profile['turn_cmd']})"
            )
        print("  or type a custom command like 0.12")
        raw = input("Speed [enter/low/medium/1/2/custom]: ").strip().lower()
        if raw == "":
            return self.forward_cmd, self.reverse_cmd, self.turn_cmd
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            raw = names[int(raw) - 1]
        if raw in INTERACTIVE_SPEED_PROFILES:
            profile = INTERACTIVE_SPEED_PROFILES[raw]
            return (
                abs(float(profile["forward_cmd"])),
                abs(float(profile["reverse_cmd"])),
                abs(float(profile["turn_cmd"])),
            )
        custom = abs(float(raw))
        if default_kind == "forward":
            return custom, self.reverse_cmd, self.turn_cmd
        if default_kind == "reverse":
            return self.forward_cmd, custom, self.turn_cmd
        return self.forward_cmd, self.reverse_cmd, custom


def _prompt_distance_m() -> float:
    print("Distance presets:")
    print("  1 = 25 cm")
    print("  2 = 50 cm")
    print("  3 = 75 cm")
    print("  4 = 1 m")
    raw = input("Distance [1/2/3/4 or meters like 0.35]: ").strip().lower()
    if raw in DISTANCE_PRESETS_M:
        return DISTANCE_PRESETS_M[raw]
    if raw.endswith("cm"):
        return float(raw[:-2].strip()) / 100.0
    if raw.endswith("m"):
        return float(raw[:-1].strip())
    return float(raw)


def _prompt_curve_radius_m() -> float:
    print("Curve radius presets:")
    print("  1 = 25 cm")
    print("  2 = 50 cm")
    print("  3 = 75 cm")
    print("  4 = 1 m")
    raw = input("Radius [1/2/3/4 or meters like 0.35]: ").strip().lower()
    if raw in CURVE_RADIUS_PRESETS_M:
        return CURVE_RADIUS_PRESETS_M[raw]
    if raw.endswith("cm"):
        return float(raw[:-2].strip()) / 100.0
    if raw.endswith("m"):
        return float(raw[:-1].strip())
    return float(raw)


def _prompt_angle_deg() -> float:
    print("Angle presets:")
    print("  1 = 45 deg")
    print("  2 = 90 deg")
    print("  3 = 180 deg")
    print("  4 = 360 deg")
    raw = input("Angle [1/2/3/4 or degrees like 135]: ").strip().lower()
    if raw in ANGLE_PRESETS_DEG:
        return ANGLE_PRESETS_DEG[raw]
    if raw.endswith("deg"):
        return float(raw[:-3].strip())
    return float(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=bench_logger.ROVER_IP)
    parser.add_argument("--duration", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=0.10)
    parser.add_argument(
        "--speed",
        choices=sorted(INTERACTIVE_SPEED_PROFILES),
        default="low",
        help="named speed preset for manual data collection; explicit command arguments override it",
    )
    parser.add_argument("--forward-cmd", type=float, default=None)
    parser.add_argument("--reverse-cmd", type=float, default=None)
    parser.add_argument("--turn-cmd", type=float, default=None)
    parser.add_argument(
        "--turn-control",
        choices=("encoder", "yaw"),
        default="encoder",
        help="turn stop signal; encoder is more repeatable for manual 90-degree tests, yaw is useful for diagnostics",
    )
    parser.add_argument(
        "--turn-counts-per-90",
        type=float,
        default=575.0,
        help="average opposite-track encoder counts used for one 90-degree turn in encoder control mode",
    )
    parser.add_argument(
        "--turn-target-scale",
        type=float,
        default=0.34,
        help=(
            "scale requested turn angle before stopping; default 0.34 with a "
            "slow turn command to reduce coast/latency overshoot on smooth floors"
        ),
    )
    parser.add_argument(
        "--min-turn-stop-target",
        type=float,
        default=22.0,
        help="minimum yaw-progress stop target for a turn chunk, useful because tiny yaw targets can stop before the tracked rover breaks static friction",
    )
    parser.add_argument(
        "--max-turn-chunk",
        type=float,
        default=90.0,
        help="split larger requested turns into chunks no larger than this angle so 180/360 turns do not rely on wrapped yaw",
    )
    parser.add_argument("--hold-after", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional CSV path; default is raw_logs/telemetry/ugv_t147_interactive_<timestamp>.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speed_profile = INTERACTIVE_SPEED_PROFILES[args.speed]
    forward_cmd = (
        speed_profile["forward_cmd"]
        if args.forward_cmd is None
        else abs(args.forward_cmd)
    )
    reverse_cmd = (
        speed_profile["reverse_cmd"]
        if args.reverse_cmd is None
        else abs(args.reverse_cmd)
    )
    turn_cmd = (
        speed_profile["turn_cmd"]
        if args.turn_cmd is None
        else abs(args.turn_cmd)
    )

    bench_logger.ROVER_IP = args.ip
    bench_logger.BASE_URL = f"http://{args.ip}/js"
    bench_logger.DURATION_SECONDS = args.duration
    bench_logger.POLL_INTERVAL_SECONDS = args.poll_interval
    bench_logger.MOTION_SCRIPT_ENABLED = True
    bench_logger.MOTION_PLAN = "interactive"
    bench_logger.STOP_WHEN_MOTION_COMPLETE = True
    bench_logger.OUTPUT_CSV = args.output or (
        bench_logger.OUT_DIR
        / f"ugv_t147_interactive_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    bench_logger.set_run_metadata(
        route="interactive_manual_sequence",
        attack_type="none",
        network_condition="wifi_baseline",
        speed_label=args.speed,
        speed_profile_note=speed_profile["note"],
    )
    print(
        "Interactive speed profile: "
        f"{args.speed} "
        f"(forward={forward_cmd:g}, reverse={reverse_cmd:g}, turn={turn_cmd:g})"
    )

    controller = InteractiveMotionController(
        forward_cmd=forward_cmd,
        reverse_cmd=reverse_cmd,
        turn_cmd=turn_cmd,
        turn_control=args.turn_control,
        turn_counts_per_90=args.turn_counts_per_90,
        turn_target_scale=args.turn_target_scale,
        min_turn_stop_target_deg=args.min_turn_stop_target,
        max_turn_chunk_deg=args.max_turn_chunk,
        hold_after_s=args.hold_after,
    )
    bench_logger.build_motion_controller = lambda: controller
    bench_logger.main()


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import time

import bench_logger


ROUTE_PRESETS = {
    "figure8": [
        ("hold", 0.0, 0.0, 4.0, "initial hold"),
        ("timed", -0.075, -0.120, 2.8, "left lobe arc"),
        ("timed", -0.095, -0.095, 0.5, "center crossing"),
        ("timed", -0.120, -0.075, 2.8, "right lobe arc"),
        ("timed", -0.095, -0.095, 0.5, "center crossing"),
    ],
    "s_curve": [
        ("hold", 0.0, 0.0, 4.0, "initial hold"),
        ("timed", -0.085, -0.120, 1.7, "left bend"),
        ("timed", -0.100, -0.100, 0.6, "short straight"),
        ("timed", -0.120, -0.085, 1.7, "right bend"),
        ("timed", -0.100, -0.100, 0.6, "short straight"),
    ],
}


NETWORK_LABELS = {
    "baseline": "wifi_baseline",
    "buffered": "wifi_buffered_delay",
    "wifi_baseline": "wifi_baseline",
    "wifi_buffered_delay": "wifi_buffered_delay",
}


def positive_int(value: str) -> int:
    repeats = int(value)
    if repeats < 1:
        raise argparse.ArgumentTypeError("repeats must be at least 1")
    return repeats


class TimedRouteController:
    def __init__(self, steps: list[bench_logger.MotionStep]) -> None:
        self.steps = steps
        self.index = 0
        self.step_started_s: float | None = None
        self.completed = False

    def update(self, telemetry: dict, now_s: float) -> tuple[float, float, str]:
        if self.completed:
            return 0.0, 0.0, "done"

        step = self.steps[self.index]
        if self.step_started_s is None:
            self.step_started_s = now_s

        if now_s - self.step_started_s >= step.target:
            self.index += 1
            self.step_started_s = now_s
            if self.index >= len(self.steps):
                self.completed = True
                return 0.0, 0.0, "done"
            return self.update(telemetry, now_s)

        return step.left_cmd, step.right_cmd, step.label


def build_steps(route: str, repeats: int, scale: float) -> list[bench_logger.MotionStep]:
    steps: list[bench_logger.MotionStep] = []
    for repeat_idx in range(repeats):
        for kind, left, right, duration_s, label in ROUTE_PRESETS[route]:
            if kind == "hold" and repeat_idx > 0 and label == "initial hold":
                duration_s = 2.0
                label = "repeat settle hold"
            steps.append(
                bench_logger.MotionStep(
                    "turn_timed",
                    left,
                    right,
                    duration_s * scale,
                    f"{label} ({route} {repeat_idx + 1}/{repeats})",
                )
            )
    steps.append(bench_logger.MotionStep("hold", 0.0, 0.0, 3.0, "final hold"))
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run timed benign curved routes for AprilTag/video ground-truth collection."
    )
    parser.add_argument("--route", choices=sorted(ROUTE_PRESETS), required=True)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--surface", default="smooth_kitchen_floor")
    parser.add_argument("--speed", choices=("low", "medium"), default="low")
    parser.add_argument("--network", choices=sorted(NETWORK_LABELS), default="baseline")
    parser.add_argument("--trial", type=positive_int, default=None)
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="scale all route segment durations without changing commands",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    network_label = NETWORK_LABELS[args.network]
    route_label = f"{args.route}x{args.repeats}"
    trial_label = f"_trial-{args.trial}" if args.trial is not None else ""
    run_id = (
        f"speed-{args.speed}_"
        f"surface-{args.surface}_"
        f"latency-{network_label}_"
        f"route-{route_label}_"
        f"attack-none"
        f"{trial_label}"
    )

    bench_logger.MOTION_SCRIPT_ENABLED = True
    bench_logger.STOP_WHEN_MOTION_COMPLETE = True
    bench_logger.OUTPUT_CSV = (
        Path("raw_logs/telemetry")
        / f"{run_id}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    bench_logger.set_run_metadata(
        run_id=run_id,
        route=route_label,
        surface=args.surface,
        speed_label=args.speed,
        network_condition=network_label,
        trial_id=args.trial,
        attack_type="none",
    )

    controller = TimedRouteController(
        build_steps(args.route, args.repeats, args.duration_scale)
    )
    bench_logger.build_motion_controller = lambda: controller
    bench_logger.main()


if __name__ == "__main__":
    main()

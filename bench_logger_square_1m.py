import argparse
from pathlib import Path
import time

import bench_logger


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benign-dataset continuous 1 m square loops."
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=3,
        help="number of continuous square loops (default: 3)",
    )
    parser.add_argument(
        "--surface",
        choices=sorted(bench_logger.SQUARE_TERRAIN_PROFILES),
        default="smooth",
        help="surface/turn calibration profile (default: smooth)",
    )
    parser.add_argument(
        "--speed",
        choices=sorted(bench_logger.SQUARE_SPEED_PROFILES),
        default="low",
        help="straight-line speed profile (default: low)",
    )
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_LABELS),
        default="baseline",
        help="network condition label for this run (default: baseline)",
    )
    parser.add_argument(
        "--trial",
        type=positive_int,
        default=None,
        help="benign trial number for the filename and CSV metadata",
    )
    parser.add_argument(
        "--turn-seconds",
        type=float,
        default=None,
        help="override every corner's 90 degree turn duration",
    )
    parser.add_argument(
        "--turn-schedule",
        default=None,
        help="comma-separated per-corner turn durations, e.g. 1.80,1.90,1.75,1.80",
    )
    parser.add_argument(
        "--turn-left",
        type=float,
        default=None,
        help="override the selected surface profile's left turn command",
    )
    parser.add_argument(
        "--turn-right",
        type=float,
        default=None,
        help="override the selected surface profile's right turn command",
    )
    parser.add_argument(
        "--turn-counts-per-90",
        type=float,
        default=None,
        help="override the selected surface profile's encoder-count target for a 90 degree turn",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    turn_cmd = None
    turn_schedule = None
    if args.turn_left is not None or args.turn_right is not None:
        if args.turn_left is None or args.turn_right is None:
            raise SystemExit("--turn-left and --turn-right must be provided together")
        turn_cmd = (args.turn_left, args.turn_right)
    if args.turn_schedule is not None:
        try:
            turn_schedule = tuple(float(part.strip()) for part in args.turn_schedule.split(","))
        except ValueError as exc:
            raise SystemExit("--turn-schedule must contain four numeric seconds values") from exc
        if len(turn_schedule) != 4:
            raise SystemExit("--turn-schedule must contain exactly four values")

    surface_profile = bench_logger.apply_square_terrain_profile(
        args.surface,
        turn_seconds=args.turn_seconds,
        turn_cmd=turn_cmd,
        turn_schedule=turn_schedule,
    )
    if args.turn_counts_per_90 is not None:
        bench_logger.SQUARE_TURN_COUNTS_PER_90 = args.turn_counts_per_90
    speed_profile = bench_logger.apply_square_speed_profile(args.speed)
    network_label = NETWORK_LABELS[args.network]

    bench_logger.MOTION_SCRIPT_ENABLED = True
    bench_logger.MOTION_PLAN = "square_1m"
    bench_logger.SQUARE_SIDE_LENGTH_M = 1.0
    bench_logger.SQUARE_REPEAT_COUNT = args.repeats
    bench_logger.STOP_WHEN_MOTION_COMPLETE = True
    route_label = f"square1mx{args.repeats}"
    trial_label = f"_trial-{args.trial}" if args.trial is not None else ""
    run_id = (
        f"speed-{speed_profile.name}_"
        f"surface-{surface_profile.surface_label}_"
        f"latency-{network_label}_"
        f"route-{route_label}_"
        f"attack-none"
        f"{trial_label}"
    )
    bench_logger.OUTPUT_CSV = (
        Path("raw_logs/telemetry")
        / f"{run_id}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    bench_logger.set_run_metadata(
        run_id=run_id,
        route=route_label,
        surface=surface_profile.surface_label,
        surface_profile=surface_profile.name,
        speed_label=speed_profile.name,
        network_condition=network_label,
        trial_id=args.trial,
        attack_type="none",
        square_side_length_m=bench_logger.SQUARE_SIDE_LENGTH_M,
        square_repeats=args.repeats,
        square_turn_profile=surface_profile.name,
        square_turn_left_cmd=bench_logger.SQUARE_TURN_CW_CMD[0],
        square_turn_right_cmd=bench_logger.SQUARE_TURN_CW_CMD[1],
        square_turn_seconds=bench_logger.SQUARE_TURN_SECONDS,
        square_turn_schedule_s=(
            ""
            if bench_logger.SQUARE_TURN_SECONDS_BY_CORNER is None
            else ";".join(str(value) for value in bench_logger.SQUARE_TURN_SECONDS_BY_CORNER)
        ),
        square_turn_counts_per_90=bench_logger.SQUARE_TURN_COUNTS_PER_90,
        square_straight_left_cmd=bench_logger.SQUARE_STRAIGHT_FORWARD_CMD[0],
        square_straight_right_cmd=bench_logger.SQUARE_STRAIGHT_FORWARD_CMD[1],
    )
    bench_logger.main()


if __name__ == "__main__":
    main()

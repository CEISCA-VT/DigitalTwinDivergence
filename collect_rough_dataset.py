import argparse
import subprocess
import sys
import time
from dataclasses import dataclass


DEFAULT_TURN_LEFT = 0.064
DEFAULT_TURN_RIGHT = -0.064
DEFAULT_TURN_SCHEDULE = "2.10,2.10,2.10,2.10"


@dataclass(frozen=True)
class TrialCommand:
    speed: str
    network: str
    trial: int
    command: list[str]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect rough permeable-concrete benign square-loop trials with "
            "the current rough-terrain tuning."
        )
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        choices=["low", "medium"],
        default=["low"],
        help="speed labels to collect in order (default: low)",
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        choices=["baseline", "buffered", "wifi_baseline", "wifi_buffered_delay"],
        default=["baseline"],
        help="network labels to collect in order (default: baseline)",
    )
    parser.add_argument(
        "--start-trial",
        type=positive_int,
        default=1,
        help="first trial number to collect for each condition (default: 1)",
    )
    parser.add_argument(
        "--trials",
        type=positive_int,
        default=5,
        help="number of trials to collect for each speed/network condition (default: 5)",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=3,
        help="square loops per trial (default: 3)",
    )
    parser.add_argument(
        "--turn-left",
        type=float,
        default=DEFAULT_TURN_LEFT,
        help=f"rough turn left command override (default: {DEFAULT_TURN_LEFT})",
    )
    parser.add_argument(
        "--turn-right",
        type=float,
        default=DEFAULT_TURN_RIGHT,
        help=f"rough turn right command override (default: {DEFAULT_TURN_RIGHT})",
    )
    parser.add_argument(
        "--turn-schedule",
        default=DEFAULT_TURN_SCHEDULE,
        help=(
            "comma-separated per-corner turn durations "
            f"(default: {DEFAULT_TURN_SCHEDULE})"
        ),
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="continue automatically between trials after the countdown",
    )
    parser.add_argument(
        "--countdown",
        type=positive_int,
        default=8,
        help="seconds to wait before each trial in --auto mode (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without running them",
    )
    return parser.parse_args()


def build_commands(args: argparse.Namespace) -> list[TrialCommand]:
    commands: list[TrialCommand] = []
    trial_stop = args.start_trial + args.trials
    for network in args.networks:
        for speed in args.speeds:
            for trial in range(args.start_trial, trial_stop):
                command = [
                    sys.executable,
                    "bench_logger_square_0_5m.py",
                    "--surface",
                    "rough",
                    "--speed",
                    speed,
                    "--network",
                    network,
                    "--trial",
                    str(trial),
                    "--repeats",
                    str(args.repeats),
                    "--turn-left",
                    str(args.turn_left),
                    "--turn-right",
                    str(args.turn_right),
                    "--turn-schedule",
                    args.turn_schedule,
                ]
                commands.append(TrialCommand(speed, network, trial, command))
    return commands


def wait_for_trial(command: TrialCommand, args: argparse.Namespace) -> bool:
    print()
    print("=" * 72)
    print(
        f"Next rough trial: speed={command.speed}, "
        f"network={command.network}, trial={command.trial}"
    )
    print("Place/reset the rover on rough permeable concrete and clear the area.")
    print("Command:")
    print(" ".join(command.command))
    print("=" * 72)

    if args.dry_run:
        return True

    if args.auto:
        for remaining in range(args.countdown, 0, -1):
            print(f"Starting in {remaining}...", end="\r", flush=True)
            time.sleep(1.0)
        print()
        return True

    response = input("Press Enter to start, or type q to stop: ").strip().lower()
    return response not in {"q", "quit", "stop", "exit"}


def main() -> None:
    args = parse_args()
    commands = build_commands(args)

    print("Rough permeable-concrete dataset collector")
    print(f"Trials queued: {len(commands)}")
    print(f"Turn command: ({args.turn_left}, {args.turn_right})")
    print(f"Turn schedule: {args.turn_schedule}")
    print("Surface label: rough_permeable_concrete")
    print("Route: square0p5x3 unless --repeats changes it")

    completed = 0
    for command in commands:
        if not wait_for_trial(command, args):
            print("Stopped before running the next trial.")
            break
        if args.dry_run:
            completed += 1
            continue
        result = subprocess.run(command.command, check=False)
        if result.returncode != 0:
            print()
            print(f"Trial failed with exit code {result.returncode}.")
            response = input("Press Enter to continue, or type q to stop: ").strip().lower()
            if response in {"q", "quit", "stop", "exit"}:
                break
        else:
            completed += 1

    print()
    print("=" * 72)
    print(f"Collector finished. Completed command count: {completed}/{len(commands)}")
    print("=" * 72)


if __name__ == "__main__":
    main()

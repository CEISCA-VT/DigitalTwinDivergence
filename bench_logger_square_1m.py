import argparse
from pathlib import Path
import time

import bench_logger


def positive_int(value: str) -> int:
    repeats = int(value)
    if repeats < 1:
        raise argparse.ArgumentTypeError("repeats must be at least 1")
    return repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous 1 m square loops.")
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=1,
        help="number of continuous square loops (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench_logger.MOTION_SCRIPT_ENABLED = True
    bench_logger.MOTION_PLAN = "square_1m"
    bench_logger.SQUARE_SIDE_LENGTH_M = 1.0
    bench_logger.SQUARE_REPEAT_COUNT = args.repeats
    bench_logger.STOP_WHEN_MOTION_COMPLETE = True
    bench_logger.OUTPUT_CSV = (
        Path("raw_logs/telemetry")
        / f"ugv_t147_square_1m_x{args.repeats}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    bench_logger.main()


if __name__ == "__main__":
    main()

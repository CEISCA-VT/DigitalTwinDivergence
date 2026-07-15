from pathlib import Path
import time

import bench_logger


bench_logger.MOTION_SCRIPT_ENABLED = True
bench_logger.MOTION_PLAN = "square_1m"
bench_logger.SQUARE_SIDE_LENGTH_M = 0.5
bench_logger.SQUARE_REPEAT_COUNT = 1
bench_logger.OUTPUT_CSV = (
    Path("raw_logs/telemetry")
    / f"ugv_t147_square_0_5m_{time.strftime('%Y%m%d_%H%M%S')}.csv"
)


if __name__ == "__main__":
    bench_logger.main()

"""Benchmark replay cost for the frozen detector variants on existing logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import tracemalloc

from DigitalTwin.analysis.common import read_rows, write_rows
from DigitalTwin.analysis.real_data_study import VARIANTS, AttackSpec, _prepare_run, replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis/real_data_study")
    parser.add_argument("--max-runs", type=int, default=4)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest = read_rows(out_dir / "benign_manifest.csv")[: args.max_runs]
    rows: list[dict[str, object]] = []
    for manifest_row in manifest:
        path = Path(manifest_row["source_csv"])
        prepared = _prepare_run(path)
        for variant in args.variants:
            tracemalloc.start()
            start = time.perf_counter()
            result = replay(path, variant, AttackSpec(), prepared=prepared)
            elapsed = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append(
                {
                    "run_id": manifest_row["run_id"],
                    "detector_variant": variant,
                    "updates": len(result.scores),
                    "elapsed_s": elapsed,
                    "mean_ms_per_update": 1000.0 * elapsed / max(1, len(result.scores)),
                    "peak_python_memory_kb": peak / 1024.0,
                }
            )
    write_rows(out_dir / "runtime_benchmark.csv", rows, rows[0].keys())
    print(out_dir / "runtime_benchmark.csv")


if __name__ == "__main__":
    main()

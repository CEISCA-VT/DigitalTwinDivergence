"""Generate false-alarm-versus-detection curves by replaying existing logs.

This is intentionally separate from the main frozen campaign. It sweeps score
thresholds after the fact to study operating-point tradeoffs. It does not
collect new data or alter raw logs, but it can be long-running because it
replays attacks for every detector variant.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from DigitalTwin.alarm import operational_run_statistic
from DigitalTwin.analysis.common import read_rows, write_rows
from DigitalTwin.analysis.real_data_study import (
    ATTACK_START_FRACTIONS,
    DRIFT_RATES_MPS,
    STEP_MAGNITUDES_M,
    VARIANTS,
    AttackSpec,
    _alarm_config,
    _attack_specs,
    _prepare_run,
    replay,
)


FINE_STEP_MAGNITUDES_M = (0.5, 1.0, 2.0, 3.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0)
EXPANDED_DRIFT_RATES_MPS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)


def _expanded_attack_specs() -> list[AttackSpec]:
    specs: list[AttackSpec] = []
    for direction in ("along", "cross"):
        specs.extend(AttackSpec("step", direction, magnitude_m=value) for value in FINE_STEP_MAGNITUDES_M)
    specs.extend([AttackSpec("freeze"), AttackSpec("replay", replay_delay_s=5.0)])
    for direction in ("along", "cross"):
        specs.extend(AttackSpec("drift", direction, rate_mps=value) for value in EXPANDED_DRIFT_RATES_MPS)
    for direction in ("along", "cross"):
        specs.append(AttackSpec("strategic_drift", direction, rate_mps=0.03))
    return specs


def _manifest(out_dir: Path) -> list[dict[str, str]]:
    return read_rows(out_dir / "benign_manifest.csv")


def _statistic(path: Path, mode: str, attack: AttackSpec, start_fraction: float | None, prepared) -> float:
    if start_fraction is not None and attack.kind != "none":
        attack = replace(attack, start_fraction=start_fraction)
    result = replay(path, mode, attack, prepared=prepared)
    return operational_run_statistic(result.scores, result.alarm_enabled, _alarm_config(mode))


def _threshold_grid(clean_stats: list[float], attack_stats: list[float], points: int) -> np.ndarray:
    values = np.asarray(clean_stats + attack_stats, dtype=float)
    if len(values) == 0:
        return np.asarray([0.0])
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        return np.asarray([low])
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, points))
    return np.unique(np.concatenate(([low - 1e-9], quantiles, [high + 1e-9])))


def run_sweep(
    out_dir: Path,
    *,
    expanded_attacks: bool,
    grid_points: int,
    variants: tuple[str, ...],
) -> list[dict[str, object]]:
    manifest = _manifest(out_dir)
    attack_specs = _expanded_attack_specs() if expanded_attacks else _attack_specs()[1:]
    clean_stats: dict[str, list[float]] = defaultdict(list)
    attack_stats: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)

    for row in manifest:
        path = Path(row["source_csv"])
        prepared = _prepare_run(path)
        for mode in variants:
            clean_stats[mode].append(_statistic(path, mode, AttackSpec(), None, prepared))
            for attack in attack_specs:
                for start in ATTACK_START_FRACTIONS:
                    key = (
                        mode,
                        attack.kind,
                        attack.direction,
                        str(attack.magnitude_m or ""),
                        str(attack.rate_mps or ""),
                        str(attack.replay_delay_s if attack.kind == "replay" else ""),
                    )
                    attack_stats[key].append(_statistic(path, mode, attack, start, prepared))

    rows: list[dict[str, object]] = []
    for key, stats in sorted(attack_stats.items(), key=lambda item: tuple(map(str, item[0]))):
        mode = key[0]
        thresholds = _threshold_grid(clean_stats[mode], stats, grid_points)
        for threshold in thresholds:
            fa = sum(value > threshold for value in clean_stats[mode]) / len(clean_stats[mode])
            pd = sum(value > threshold for value in stats) / len(stats)
            rows.append(
                {
                    "detector_variant": mode,
                    "attack": key[1],
                    "direction": key[2],
                    "magnitude_m": key[3],
                    "rate_mps": key[4],
                    "replay_delay_s": key[5],
                    "threshold": float(threshold),
                    "run_false_alarm_probability": fa,
                    "run_detection_probability": pd,
                    "benign_runs": len(clean_stats[mode]),
                    "attack_evaluations": len(stats),
                }
            )
    filename = "threshold_sweep_expanded.csv" if expanded_attacks else "threshold_sweep.csv"
    write_rows(out_dir / filename, rows, rows[0].keys())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis/real_data_study")
    parser.add_argument("--expanded-attacks", action="store_true")
    parser.add_argument("--grid-points", type=int, default=50)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(VARIANTS),
        help="Optional detector variants to sweep; default is all variants.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    run_sweep(
        out_dir,
        expanded_attacks=args.expanded_attacks,
        grid_points=args.grid_points,
        variants=tuple(args.variants),
    )
    print(out_dir / ("threshold_sweep_expanded.csv" if args.expanded_attacks else "threshold_sweep.csv"))


if __name__ == "__main__":
    main()

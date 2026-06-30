"""Batch experiment runner.

Examples:
    python -m DigitalTwin.experiments.experiment --quick
    python -m DigitalTwin.experiments.experiment --full-matrix --nominal-only
    python -m DigitalTwin.experiments.experiment --full-matrix --step-sweep
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from DigitalTwin.attack import AttackConfig
from DigitalTwin.simulator import SimulationConfig, run_simulation


STEP_MAGNITUDES_M = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]


def build_attack(kind: str, magnitude_m: float | None = None, start_s: float = 30.0) -> AttackConfig:
    if kind == "none":
        return AttackConfig(kind="none")
    if kind == "step":
        magnitude = 1.2 if magnitude_m is None else magnitude_m
        return AttackConfig(kind="step", start_s=start_s, epsilon_x_m=magnitude, epsilon_y_m=0.0)
    if kind == "freeze":
        return AttackConfig(kind="freeze", start_s=start_s)
    if kind == "replay":
        return AttackConfig(kind="replay", start_s=start_s, replay_delay_s=20.0)
    if kind == "random_drift":
        return AttackConfig(kind="random_drift", start_s=start_s, drift_x_mps=0.035, drift_y_mps=-0.015)
    raise ValueError(f"unknown attack {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="DigitalTwin/datasets")
    parser.add_argument("--quick", action="store_true", help="small smoke test")
    parser.add_argument("--full-matrix", action="store_true", help="proposal 2^3 speed/terrain/latency matrix")
    parser.add_argument("--nominal-only", action="store_true", help="only run benign trials")
    parser.add_argument("--step-sweep", action="store_true", help="run step attacks from 0.5 m to 10 m")
    parser.add_argument("--include-random-drift", action="store_true")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    if args.full_matrix:
        speeds = [0.2, 0.8]
        terrains = [0.0, 1.0]
        latencies = [10.0, 200.0]
        trials = range(args.trials if args.trials is not None else 5)
        duration_s = args.duration if args.duration is not None else 60.0
    elif args.quick:
        speeds = [0.2]
        terrains = [0.0]
        latencies = [10.0, 200.0]
        trials = range(args.trials if args.trials is not None else 1)
        duration_s = args.duration if args.duration is not None else 25.0
    else:
        speeds = [0.2, 0.8]
        terrains = [0.0, 1.0]
        latencies = [10.0, 200.0]
        trials = range(args.trials if args.trials is not None else 1)
        duration_s = args.duration if args.duration is not None else 60.0

    attack_specs: list[tuple[str, float | None]] = [("none", None)]
    if not args.nominal_only:
        if args.step_sweep:
            attack_specs.extend(("step", magnitude) for magnitude in STEP_MAGNITUDES_M)
        else:
            attack_specs.append(("step", 1.2))
        attack_specs.extend([("freeze", None), ("replay", None)])
        if args.include_random_drift:
            attack_specs.append(("random_drift", None))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for speed, terrain, latency, (attack_kind, magnitude), trial in product(
        speeds,
        terrains,
        latencies,
        attack_specs,
        trials,
    ):
        magnitude_tag = "" if magnitude is None else f"_eps-{magnitude:.1f}"
        name = (
            f"speed-{speed:.2f}_terrain-{terrain:.1f}_latency-{int(latency)}"
            f"_attack-{attack_kind}{magnitude_tag}_trial-{trial}.csv"
        )
        config = SimulationConfig(
            speed_mps=speed,
            terrain_index=terrain,
            latency_ms=latency,
            seed=1000 + 37 * trial + int(speed * 100) + int(terrain * 10) + int(latency),
            duration_s=duration_s,
            trajectory="figure8" if attack_kind == "random_drift" else "square",
        )
        attack_start_s = min(30.0, max(5.0, 0.48 * duration_s))
        path = run_simulation(config, build_attack(attack_kind, magnitude, start_s=attack_start_s), out_dir / name)
        print(path)


if __name__ == "__main__":
    main()

"""Controlled centered-bank availability experiment, without twin retraining.

Run: python -m scripts.run_buffered_centered_heldout
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from DigitalTwin.analysis import service_timing_budget_study as base
from DigitalTwin.analysis.service_evidence_from_bank import validate_bank_sources
from scripts.check_delivery_grid_robustness import (
    build_predictions, validate_table, verify_full_reproduction,
)
from scripts.compare_surface_alternatives import operating_points


ROOT = base.ROOT
RUN_KEY = ["outer_sequence", "sequence", "replicate", "service", "rate_hz", "delay_ms"]


def replay_entry(entry, availability_ms):
    path = ROOT / entry["trajectory"]
    data = base.load(path)
    if not all(np.isfinite(values).all() for values in data.values()):
        raise ValueError(f"Nonfinite required signals: {path}")
    features = base.motion_features(data)
    rows = []
    for service in base.SERVICES:
        for rate in base.RATES:
            for delay in base.DELAYS_MS:
                # Buffering changes arrival selection, not source timestamps or cell identity.
                effective_delay = delay + availability_ms
                metrics = base.evaluate(data, service, rate, effective_delay)
                calibration = base.evaluate(data, service, rate, effective_delay,
                                            start_s=0., end_s=base.PREFIX_S)
                qualified = (metrics["coverage"] >= base.MIN_COVERAGE
                             and metrics["joint_satisfaction"] >= base.TARGET)
                rows.append(dict(outer_sequence=entry["outer"], sequence=entry["target"],
                                 replicate=path.parent.parent.name,
                                 source=str(path.relative_to(ROOT)), **service,
                                 rate_hz=rate, interval_s=1 / rate, delay_ms=delay,
                                 **features, **metrics, qualified=int(qualified),
                                 **{f"calibration_{k}": calibration[k] for k in
                                    ("coverage", "physical_satisfaction",
                                     "freshness_satisfaction", "joint_satisfaction")}))
    return pd.DataFrame(rows)


def aggregate(runs):
    frames = []
    for outer, block in runs.groupby("outer_sequence"):
        if not block.groupby(["sequence", "service", "rate_hz", "delay_ms"]).size().eq(3).all():
            raise AssertionError("Expected exactly three seeds per sequence/cell")
        frame = base.sequence_rows(block)
        frame.insert(0, "outer_sequence", outer)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    validate_table(table)
    return table


def paired_risks(control, buffered):
    rows = []
    for (method, target), a in control[control.scope.eq("pooled")].groupby(
            ["method", "target_acceptance"]):
        a = a.set_index("heldout_sequence").sort_index()
        b = buffered[buffered.scope.eq("pooled") & buffered.method.eq(method)
                     & buffered.target_acceptance.eq(target)].set_index("heldout_sequence").reindex(a.index)
        indices = np.random.default_rng(20261109).integers(0, len(a), (5000, len(a)))
        risks = []
        for frame in (a, b):
            counts = frame.accepted_count.to_numpy()[indices].sum(axis=1)
            false = frame.false_count.to_numpy()[indices].sum(axis=1)
            risks.append(np.divide(false, counts, out=np.full(5000, np.nan), where=counts > 0))
        differences = 100 * (risks[1] - risks[0])
        valid = differences[np.isfinite(differences)]
        lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        observed = (100 * (b.false_count.sum() / b.accepted_count.sum()
                          - a.false_count.sum() / a.accepted_count.sum())
                    if a.accepted_count.sum() and b.accepted_count.sum() else np.nan)
        rows.append(dict(method=method, target_acceptance=target,
                         risk_difference_percentage_points=observed,
                         ci_low=lo, ci_high=hi, defined_bootstrap_samples=len(valid)))
    return pd.DataFrame(rows)


def ledger(table):
    outer = table[table.sequence.eq(table.outer_sequence)]
    rows = []
    for (sequence, service), group in outer.groupby(["sequence", "service"]):
        selected = group.set_index(["rate_hz", "delay_ms"])
        degraded, practical, ideal = (selected.loc[cell] for cell in ((2., 200), (5., 50), (10., 0)))
        if min(degraded.coverage, practical.coverage, ideal.coverage) < base.MIN_COVERAGE:
            category = "unobservable"
        else:
            category = ("already_qualified" if degraded.qualified else
                        "remediable" if ideal.qualified else "persistent")
        rows.append(dict(sequence=sequence, service=service, category=category,
                         practical_qualified=int(practical.qualified),
                         degraded_joint=degraded.joint_satisfaction,
                         practical_joint=practical.joint_satisfaction,
                         ideal_joint=ideal.joint_satisfaction))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/buffered_centered_heldout_100ms")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "results/doubly_nested_loso_slurm/merged_manifest.json"
    reference_path = ROOT / "results/doubly_nested_loso_evidence/per_run_response_surface.csv"
    manifest = json.loads(manifest_path.read_text())
    if manifest["study"] != "doubly_nested" or len(manifest["trajectories"]) != 300:
        raise ValueError("Expected complete centered nested bank")
    print("Verifying all 300 source trajectory hashes", flush=True)
    validate_bank_sources(manifest)
    reference = pd.read_csv(reference_path).set_index(RUN_KEY).sort_index()
    buffered_runs = []
    maximum_difference = 0.
    for i, entry in enumerate(manifest["trajectories"], 1):
        control = replay_entry(entry, 0).set_index(RUN_KEY).sort_index()
        expected = reference.loc[control.index]
        columns = control.select_dtypes(include="number").columns
        np.testing.assert_allclose(control[columns], expected[columns], rtol=1e-10, atol=1e-10)
        maximum_difference = max(maximum_difference, float(np.max(np.abs(
            control[columns].to_numpy() - expected[columns].to_numpy()))))
        buffered_runs.append(replay_entry(entry, 100))
        if i % 10 == 0:
            print(f"Replayed {i}/300 trajectories; unbuffered control verified", flush=True)
    runs = pd.concat(buffered_runs, ignore_index=True)
    runs.to_csv(output / "per_run_response_surface.csv", index=False)
    buffered = aggregate(runs)
    buffered.to_csv(output / "per_sequence_response_surface.csv", index=False)
    control_table = aggregate(reference.reset_index())
    results = {}
    with threadpool_limits(limits=1):
        for condition, table in (("unbuffered_control", control_table), ("buffered_100ms", buffered)):
            print(f"Refitting seven screening rules: {condition}", flush=True)
            predictions = build_predictions(table)
            per, summary = operating_points(predictions)
            if condition == "unbuffered_control":
                verify_full_reproduction("centered", predictions, summary)
            directory = output / condition
            directory.mkdir(exist_ok=True)
            predictions.to_csv(directory / "predictions.csv", index=False)
            per.to_csv(directory / "per_sequence_operating_points.csv", index=False)
            summary.to_csv(directory / "operating_point_summary.csv", index=False)
            ledger(table).to_csv(directory / "ledger.csv", index=False)
            outer = table[table.outer_sequence.eq(table.sequence)]
            print(f"{condition}: {outer.qualified.sum()}/800 qualified ground-truth cells", flush=True)
            selected = summary[summary.scope.eq("pooled") & summary.target_acceptance.isin([.25, .5])]
            print(selected.to_string(index=False), flush=True)
            results[condition] = (per, summary)
    changes = paired_risks(results["unbuffered_control"][0], results["buffered_100ms"][0])
    changes.to_csv(output / "paired_risk_changes.csv", index=False)
    paths = [manifest_path, reference_path, Path(__file__),
             ROOT / "DigitalTwin/analysis/service_timing_budget_study.py",
             ROOT / "DigitalTwin/analysis/service_evidence_from_bank.py",
             ROOT / "scripts/check_delivery_grid_robustness.py",
             ROOT / "scripts/compare_surface_alternatives.py",
             ROOT / "scripts/compare_structured_qualification.py"]
    provenance = dict(status="complete", availability_delay_ms=100, source_trajectories=300,
                      unbuffered_control_max_difference=maximum_difference,
                      control_seven_method_predictions_and_counts_reproduced=True,
                      network_delays_ms=list(base.DELAYS_MS), rates_hz=list(base.RATES),
                      source_timestamps_unchanged=True, twin_retraining=False,
                      cutoff_selection="eight-sequence cross-fitting within nine development recordings",
                      seeds_aggregated_before_qualification=True, bootstrap_physical_sequences=5000,
                      input_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in paths}, trajectory_hashes=manifest["trajectories"])
    (output / "experiment_manifest.json").write_text(json.dumps(provenance, indent=2))
    lines = ["# 100-ms buffered-centered held-out experiment", "",
             "Only availability changes: arrival = original source time + 100 ms + network delay.",
             "Source states, reference, network-cell labels, services, and sensor prefix features remain fixed.",
             "All 300 trajectory hashes verified; unbuffered measurements and seven-method results reproduced.",
             "All development and outer-test labels are replayed. Screening rules and cutoffs are refitted.",
             "Intervals use 5,000 whole-physical-sequence bootstrap samples.", ""]
    for condition, (_, summary) in results.items():
        lines += [f"## {condition}", "", "Target | Method | Accepted | False | Risk (%) | 95% interval (%)",
                  "--- | --- | --- | --- | --- | ---"]
        for row in summary[summary.scope.eq("pooled") & summary.target_acceptance.isin([.25, .5])].itertuples():
            lines.append(f"{row.target_acceptance:.0%} | {row.method} | {row.accepted_count} | "
                         f"{row.false_count} | {100*row.risk:.2f} | {100*row.ci_low:.2f}--{100*row.ci_high:.2f}")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines))
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()

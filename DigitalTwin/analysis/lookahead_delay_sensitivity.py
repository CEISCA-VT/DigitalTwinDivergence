"""Treat centered-gradient feature lookahead as explicit delivery latency.

This is a sensitivity analysis over already saved trajectories.  It does not
replace retraining with causal derivatives and makes no claim that model states
would be identical after causal retraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from DigitalTwin.analysis import service_timing_budget_study as base


DEFAULT_OUT = base.ROOT / "results/lookahead_as_delay_sensitivity"
CONDITIONS = {
    "ideal": (10.0, 0),
    "practical": (5.0, 50),
    "degraded": (2.0, 200),
}


def classify(frame: pd.DataFrame) -> str:
    values = frame.set_index("condition").qualified
    if not bool(values["ideal"]):
        return "persistent_under_ideal"
    if bool(values["degraded"]):
        return "already_qualified_degraded"
    return "delivery_remediable"


def run(output: Path, lookahead_ms: int) -> None:
    paths = base.discover()
    rows = []
    for path in paths:
        data = base.load(path)
        sequence = path.parent.name.split("_", 2)[-1]
        replicate = path.parent.parent.name
        for service in base.SERVICES:
            for scenario, added_ms in (("recorded_clock", 0), ("lookahead_accounted", lookahead_ms)):
                for condition, (rate, delay_ms) in CONDITIONS.items():
                    metrics = base.evaluate(data, service, rate, delay_ms + added_ms)
                    qualified = (metrics["coverage"] >= base.MIN_COVERAGE and
                                 metrics["physical_satisfaction"] >= base.TARGET and
                                 metrics["freshness_satisfaction"] >= base.TARGET and
                                 metrics["joint_satisfaction"] >= base.TARGET)
                    rows.append({"sequence": sequence, "replicate": replicate, "service": service["service"],
                                 "scenario": scenario, "condition": condition, "rate_hz": rate,
                                 "nominal_delay_ms": delay_ms, "feature_availability_delay_ms": added_ms,
                                 "effective_delay_ms": delay_ms + added_ms, **metrics,
                                 "qualified": int(qualified)})
    per_run = pd.DataFrame(rows)
    metrics = ["coverage", "physical_satisfaction", "freshness_satisfaction", "joint_satisfaction"]
    per_sequence = per_run.groupby(
        ["sequence", "service", "scenario", "condition", "rate_hz", "nominal_delay_ms",
         "feature_availability_delay_ms", "effective_delay_ms"], as_index=False
    )[metrics].mean()
    per_sequence["qualified"] = ((per_sequence.coverage >= base.MIN_COVERAGE) &
                                  (per_sequence.physical_satisfaction >= base.TARGET) &
                                  (per_sequence.freshness_satisfaction >= base.TARGET) &
                                  (per_sequence.joint_satisfaction >= base.TARGET)).astype(int)
    ledger_rows = []
    for (sequence, service, scenario), group in per_sequence.groupby(["sequence", "service", "scenario"]):
        ledger_rows.append({"sequence": sequence, "service": service, "scenario": scenario,
                            "classification": classify(group),
                            **{f"{row.condition}_qualified": int(row.qualified) for row in group.itertuples()}})
    ledger = pd.DataFrame(ledger_rows)
    comparison = ledger.pivot(index=["sequence", "service"], columns="scenario", values="classification").reset_index()
    comparison["classification_changed"] = comparison.recorded_clock != comparison.lookahead_accounted
    summary = ledger.groupby(["scenario", "classification"]).size().rename("cases").reset_index()
    output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(output / "per_run_effective_delay.csv", index=False)
    per_sequence.to_csv(output / "per_sequence_effective_delay.csv", index=False)
    ledger.to_csv(output / "remediability_ledger.csv", index=False)
    comparison.to_csv(output / "classification_changes.csv", index=False)
    summary.to_csv(output / "classification_summary.csv", index=False)
    payload = {
        "schema": "centered_gradient_lookahead_as_delay_v1", "status": "complete",
        "lookahead_ms": lookahead_ms, "source_trajectories": len(paths),
        "physical_sequences": int(per_sequence.sequence.nunique()),
        "interpretation": "sensitivity over frozen centered-gradient trajectories; not causal retraining",
        "changed_sequence_service_cases": int(comparison.classification_changed.sum()),
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table = summary.pivot(index="classification", columns="scenario", values="cases").fillna(0).astype(int)
    report = ["# Centered-gradient lookahead as delivery delay", "",
              f"A predeclared `{lookahead_ms} ms` availability penalty was added to every replay condition.",
              "This is a frozen-trajectory sensitivity analysis, not causal-feature retraining.", "",
              "| Classification | Recorded clock | Lookahead accounted |", "|---|---:|---:|"]
    for classification, row in table.iterrows():
        report.append(f"| {classification.replace('_', ' ')} | {row.get('recorded_clock', 0)} | {row.get('lookahead_accounted', 0)} |")
    report += ["", f"Only **{int(comparison.classification_changed.sum())}/40** sequence-service classifications changed.",
               "The causal-derivative Kaggle study remains necessary to test whether causal feature construction changes the trajectories themselves."]
    (output / "lookahead_delay_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lookahead-ms", type=int, default=100,
                        help="Availability penalty for one future 10-Hz sample")
    args = parser.parse_args()
    run(args.output, args.lookahead_ms)


if __name__ == "__main__":
    main()

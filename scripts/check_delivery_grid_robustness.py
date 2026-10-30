"""Refit qualification rules after collapsing timing-equivalent delay cells.

Run with: python -m scripts.check_delivery_grid_robustness
Only saved, seed-aggregated nested evidence is used. Twin models are not retrained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from DigitalTwin.analysis.service_evidence_from_bank import _scores
from DigitalTwin.analysis.service_timing_budget_study import delivered_indices
from scripts.compare_structured_qualification import score as structured_score
from scripts.compare_surface_alternatives import operating_points


ROOT = Path(__file__).resolve().parents[1]
KEY = ["outer_sequence", "sequence", "service", "rate_hz", "delay_ms"]
PREDICTION_KEY = ["outer_sequence", "prediction_role", "sequence", "service",
                  "rate_hz", "delay_ms", "method"]
BANKS = {
    "centered": ROOT / "results/doubly_nested_loso_evidence",
    "causal": ROOT / "results/doubly_nested_causal_loso_slurm/service_evidence",
}
EXPECTED = {
    "centered": {"service_empirical": (200, 4), "service_specific_logistic": (203, 3),
                 "service_specific_tree": (200, 3), "motion_logistic": (199, 55),
                 "aoi_only": (160, 28), "common": (200, 30),
                 "service_identity_only": (200, 75)},
    "causal": {"service_empirical": (168, 19), "service_specific_logistic": (204, 18),
               "service_specific_tree": (190, 19), "motion_logistic": (202, 49),
               "aoi_only": (160, 37), "common": (144, 27),
               "service_identity_only": (200, 92)},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_table(table: pd.DataFrame) -> None:
    if len(table) != 8000 or table.duplicated(KEY).any():
        raise ValueError("Expected 8,000 unique nested sequence-service-delivery rows")
    sequences = set(table.sequence)
    if len(sequences) != 10 or set(table.outer_sequence) != sequences:
        raise ValueError("Expected ten physical recordings and ten matching outer folds")
    grid = {(s, r, d) for s in table.service.unique()
            for r in (1., 2., 5., 10.) for d in (0, 25, 50, 100, 200)}
    if len(grid) != 80:
        raise ValueError("Expected four services")
    for _, block in table.groupby(["outer_sequence", "sequence"]):
        if set(block[["service", "rate_hz", "delay_ms"]].itertuples(index=False, name=None)) != grid:
            raise ValueError("Incomplete nested cell grid")
    if not table.qualified.isin([0, 1]).all():
        raise ValueError("Expected binary qualification labels")


def duplicate_audit(table: pd.DataFrame) -> list[dict]:
    keys = KEY[:-1]
    columns = list(table.select_dtypes(include="number").columns.difference(["delay_ms"]))
    reference = table[table.delay_ms.eq(25)].set_index(keys).sort_index()
    rows = []
    for delay in (50, 100):
        other = table[table.delay_ms.eq(delay)].set_index(keys).sort_index()
        pd.testing.assert_index_equal(reference.index, other.index)
        for column in columns:
            if column in keys:
                continue
            a, b = reference[column].to_numpy(), other[column].to_numpy()
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)
            difference = np.abs(a - b)
            rows.append(dict(reference_delay_ms=25, compared_delay_ms=delay,
                             column=column, paired_rows=len(a),
                             max_absolute_difference=float(np.nanmax(difference))))
    times = np.arange(201, dtype=float) / 10
    for rate in (1, 2, 5, 10):
        reference_indices = delivered_indices(times, rate, .025)
        for delay in (.050, .100):
            np.testing.assert_array_equal(reference_indices, delivered_indices(times, rate, delay))
    return rows


def build_predictions(table: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for outer, fold in table.groupby("outer_sequence", sort=True):
        development = fold[fold.sequence.ne(outer)]
        for held in sorted(fold.sequence.unique()):
            fit = development[development.sequence.ne(held)]
            test = fold[fold.sequence.eq(held)]
            if set(fit.sequence) & {outer, held}:
                raise AssertionError("Excluded physical recording entered a fit")
            expected_fit_count = 9 if held == outer else 8
            if fit.sequence.nunique() != expected_fit_count:
                raise AssertionError("Wrong inner/outer fitting set")
            role = "outer_test" if held == outer else "operating_point_calibration"
            original = _scores(fit, test)
            original["outer_sequence"] = outer
            original["prediction_role"] = role
            blocks.append(original)
            for method in ("service_specific_logistic", "service_specific_tree"):
                block = test[["sequence", "service", "rate_hz", "delay_ms"]].copy()
                block["outer_sequence"] = outer
                block["prediction_role"] = role
                block["actual_qualified"] = test.qualified.to_numpy()
                block["method"] = method
                block["score"] = structured_score(fit, test, method)
                block["training_sequences"] = ";".join(sorted(fit.sequence.unique()))
                blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    if result.duplicated(PREDICTION_KEY).any() or not np.isfinite(result.score).all():
        raise AssertionError("Duplicate predictions or nonfinite scores")
    return result


def verify_full_reproduction(bank: str, predictions: pd.DataFrame,
                             summary: pd.DataFrame) -> float:
    reference_path = ROOT / "results/structured_qualification_baselines" / bank / "predictions.csv"
    reference = pd.read_csv(reference_path).set_index(PREDICTION_KEY).sort_index()
    reproduced = predictions.set_index(PREDICTION_KEY).sort_index()
    pd.testing.assert_index_equal(reference.index, reproduced.index)
    np.testing.assert_array_equal(reference.actual_qualified, reproduced.actual_qualified)
    np.testing.assert_allclose(reference.score, reproduced.score, rtol=1e-8, atol=1e-10)
    selected = summary[summary.scope.eq("pooled") & summary.target_acceptance.eq(.25)].set_index("method")
    for method, (accepted, false) in EXPECTED[bank].items():
        if (int(selected.loc[method, "accepted_count"]),
                int(selected.loc[method, "false_count"])) != (accepted, false):
            raise AssertionError(f"Failed to reproduce {bank} {method}")
    return float(np.max(np.abs(reference.score - reproduced.score)))


def paired_changes(per: pd.DataFrame) -> pd.DataFrame:
    """Use identical recording resamples for full and reduced grids."""
    rows = []
    for (bank, method, target), block in per[per.scope.eq("pooled")].groupby(
            ["bank", "method", "target_acceptance"]):
        full = block[block.grid.eq("full")].set_index("heldout_sequence").sort_index()
        indices = np.random.default_rng(20261030).integers(0, len(full), (5000, len(full)))
        full_a = full.accepted_count.to_numpy()[indices].sum(axis=1)
        full_f = full.false_count.to_numpy()[indices].sum(axis=1)
        full_risk = np.divide(full_f, full_a, out=np.full(5000, np.nan), where=full_a > 0)
        for grid, subset in block[block.grid.ne("full")].groupby("grid"):
            reduced = subset.set_index("heldout_sequence").reindex(full.index)
            a = reduced.accepted_count.to_numpy()[indices].sum(axis=1)
            f = reduced.false_count.to_numpy()[indices].sum(axis=1)
            risk = np.divide(f, a, out=np.full(5000, np.nan), where=a > 0)
            differences = 100 * (risk - full_risk)
            valid = differences[np.isfinite(differences)]
            full_count, reduced_count = full.accepted_count.sum(), reduced.accepted_count.sum()
            observed = (100 * (reduced.false_count.sum() / reduced_count - full.false_count.sum() / full_count)
                        if full_count and reduced_count else np.nan)
            low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
            rows.append(dict(bank=bank, grid=grid, method=method, target_acceptance=target,
                             risk_difference_percentage_points=observed,
                             ci_low=low, ci_high=high, defined_bootstrap_samples=len(valid)))
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, changes: pd.DataFrame, output: Path) -> None:
    labels = {"service_empirical": "Historical surface", "service_specific_logistic": "Service-specific logistic",
              "service_specific_tree": "Service-specific tree", "aoi_only": "AoI only",
              "common": "Common delivery", "service_identity_only": "Service identity",
              "motion_logistic": "Motion logistic"}
    lines = ["# Delivery-grid robustness", "",
             "The 25-, 50-, and 100-ms conditions have identical saved measurements in both banks.",
             "A 10-Hz integer-clock replay with inclusive arrival selection also gives identical packet-index histories.",
             "The full grid has 800 held-out cells. Keeping one of these delays gives 480 cells (10 recordings x 4 services x 4 rates x 3 delays).", "",
             "All seven qualification rules are refitted, and every development-only cutoff is reselected.",
             "The existing full-grid scores and all 25% operating-point counts were reproduced before comparison.",
             "The main reduced grid retains 50 ms because it is the configured practical delay. Retaining 25 or 100 ms is also reported.",
             "No twin model or trajectory was regenerated. All intervals use 5,000 physical-recording bootstrap resamples.", ""]
    for bank in BANKS:
        lines += [f"## {bank.title()} bank", "",
                  "| Method | Full grid: false / accepted (risk) | Unique grid, retain 50 ms: false / accepted (risk) | Reduced acceptance |",
                  "| --- | --- | --- | --- |"]
        chosen = summary[summary.bank.eq(bank) & summary.scope.eq("pooled") & summary.target_acceptance.eq(.25)]
        for method, label in labels.items():
            full = chosen[chosen.grid.eq("full") & chosen.method.eq(method)].iloc[0]
            reduced = chosen[chosen.grid.eq("unique_50ms") & chosen.method.eq(method)].iloc[0]
            lines.append(f"| {label} | {full.false_count}/{full.accepted_count} ({100*full.risk:.2f}%) | "
                         f"{reduced.false_count}/{reduced.accepted_count} ({100*reduced.risk:.2f}%) | {100*reduced.coverage:.2f}% |")
        lines.append("")
        for target in (.25, .50):
            row = changes[changes.bank.eq(bank) & changes.grid.eq("unique_50ms")
                          & changes.method.eq("service_empirical") & changes.target_acceptance.eq(target)].iloc[0]
            reduced = summary[summary.bank.eq(bank) & summary.grid.eq("unique_50ms")
                              & summary.method.eq("service_empirical") & summary.scope.eq("pooled")
                              & summary.target_acceptance.eq(target)].iloc[0]
            lines.append(f"At {100*target:.0f}% requested acceptance, the reduced-grid surface has risk "
                         f"{100*reduced.risk:.2f}% (95% interval {100*reduced.ci_low:.2f}--{100*reduced.ci_high:.2f}%). "
                         f"Reduced minus full risk is {row.risk_difference_percentage_points:.2f} percentage points "
                         f"(paired 95% interval {row.ci_low:.2f}--{row.ci_high:.2f}).")
        lines += ["", "Representative-delay sensitivity at 25% request:", ""]
        for method, label in labels.items():
            variants = chosen[chosen.grid.ne("full") & chosen.method.eq(method)]
            lines.append(f"- {label}: risk {100*variants.risk.min():.2f}--{100*variants.risk.max():.2f}%; "
                         f"accepted {variants.accepted_count.min()}--{variants.accepted_count.max()} of 480.")
        lines.append("")
    lines += ["The reduced grid changes the weighting of delivery conditions and can change fitted scores, cutoffs, acceptance, and risk.",
              "It is a sensitivity analysis of the qualification comparison. It does not change the three-point delivery ledger."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/delivery_grid_robustness")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_summaries, all_per, audits, sources, reproduction = [], [], [], {}, {}
    for bank, evidence in BANKS.items():
        source = evidence / "per_sequence_response_surface.csv"
        table = pd.read_csv(source)
        validate_table(table)
        audits.extend(dict(bank=bank, **row) for row in duplicate_audit(table))
        for path in (source, evidence / "evidence_manifest.json",
                     ROOT / "results/structured_qualification_baselines" / bank / "predictions.csv"):
            sources[str(path.relative_to(ROOT))] = digest(path)
        for grid, representative in [("full", None), ("unique_50ms", 50),
                                     ("unique_25ms", 25), ("unique_100ms", 100)]:
            selected = table if representative is None else table[table.delay_ms.isin([0, representative, 200])]
            print(f"{bank}: fitting {grid} ({len(selected)} nested rows)", flush=True)
            predictions = build_predictions(selected)
            per, summary = operating_points(predictions)
            if grid == "full":
                reproduction[bank] = verify_full_reproduction(bank, predictions, summary)
            directory = args.output / bank / grid
            directory.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(directory / "predictions.csv", index=False)
            for frame in (per, summary):
                frame.insert(0, "grid", grid)
                frame.insert(0, "bank", bank)
            all_per.append(per)
            all_summaries.append(summary)
    per, summary = pd.concat(all_per, ignore_index=True), pd.concat(all_summaries, ignore_index=True)
    changes = paired_changes(per)
    per.to_csv(args.output / "per_sequence_operating_points.csv", index=False)
    summary.to_csv(args.output / "operating_point_summary.csv", index=False)
    changes.to_csv(args.output / "paired_risk_changes.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output / "duplicate_measurement_audit.csv", index=False)
    write_report(summary, changes, args.output)
    code_paths = [Path(__file__), ROOT / "scripts/compare_structured_qualification.py",
                  ROOT / "scripts/compare_surface_alternatives.py",
                  ROOT / "DigitalTwin/analysis/service_evidence_from_bank.py",
                  ROOT / "DigitalTwin/analysis/service_timing_budget_study.py"]
    manifest = dict(status="complete", input_hashes=sources,
                    code_hashes={str(p.relative_to(ROOT)): digest(p) for p in code_paths},
                    python=platform.python_version(), sklearn=sklearn.__version__,
                    full_grid_reproduced=True, maximum_full_score_difference=reproduction,
                    bootstrap="5000 recording resamples; existing operating-point function; paired changes seed 20261030",
                    main_representative_delay_ms=50, sensitivity_representatives_ms=[25, 50, 100],
                    full_heldout_cells=800, reduced_heldout_cells=480,
                    fitting="Refit all seven rules on the selected grid. Inner LOSO acceptance-only cutoffs, ties retained; outer outcomes excluded.",
                    no_twin_retraining=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print((args.output / "report.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

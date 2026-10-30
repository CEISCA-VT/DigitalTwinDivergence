"""Compare neighboring-cell smoothing and service-level shrinkage on saved folds."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEY = ["sequence", "service", "rate_hz", "delay_ms"]
CELL = ["service", "rate_hz", "delay_ms"]
VARIANTS = {"smooth_h0.5": ("smooth", 0.5), "smooth_h1": ("smooth", 1.0),
            "shrink_k2": ("shrink", 2.0), "shrink_k5": ("shrink", 5.0),
            "shrink_k10": ("shrink", 10.0)}
LABELS = {"service_empirical": "Original surface", "service_identity_only": "Service identity",
          "aoi_only": "AoI only", "common": "Common delivery", "motion_logistic": "Motion logistic",
          "smooth_h0.5": "Smoothed surface (h=0.5)", "smooth_h1": "Smoothed surface (h=1)",
          "shrink_k2": "Pooled surface (k=2)", "shrink_k5": "Pooled surface (k=5)",
          "shrink_k10": "Pooled surface (k=10)"}


def scores(train, test, kind, strength):
    """Each sequence contributes one binary label per cell; services stay separate."""
    counts = train.groupby(CELL).qualified.agg(["sum", "count"])
    n = train.sequence.nunique()
    if not counts["count"].eq(n).all():
        raise ValueError("Incomplete or duplicate training grid")
    result = np.empty(len(test))
    for service in test.service.unique():
        block = counts.loc[service]
        mask = test.service.eq(service).to_numpy()
        query = test.loc[mask]
        if kind == "smooth":
            # Distance is in update-rate octaves and 100-ms delay units.
            locations = np.column_stack((np.log2(block.index.get_level_values("rate_hz")),
                                         block.index.get_level_values("delay_ms") / 100.0))
            queries = np.column_stack((np.log2(query.rate_hz), query.delay_ms / 100.0))
            weights = np.exp(-np.sum((queries[:, None] - locations[None, :]) ** 2, axis=2)
                             / (2 * strength ** 2))
            result[mask] = (weights @ (block["sum"] / block["count"]).to_numpy()) / weights.sum(axis=1)
        else:
            # k is a pooling weight in equivalent per-cell sequence observations.
            # The service mean is a training-only shrinkage target, not an
            # independent prior or a claim that grid cells are independent.
            mean = float(train.loc[train.service.eq(service), "qualified"].mean())
            selected = block.reindex(pd.MultiIndex.from_frame(query[["rate_hz", "delay_ms"]]))
            result[mask] = (selected["sum"].to_numpy() + strength * mean) / (n + strength)
    return result


def cutoff(values, target):
    candidates = np.r_[np.inf, np.unique(values)[::-1], -np.inf]
    acceptance = (values[:, None] >= candidates).mean(axis=0)
    best = min(range(len(candidates)), key=lambda i: (abs(acceptance[i] - target), -candidates[i]))
    return candidates[best], acceptance[best]


def build_predictions(table):
    output = []
    for outer, fold in table.groupby("outer_sequence"):
        development = fold[fold.sequence.ne(outer)]
        if development.sequence.nunique() != 9:
            raise ValueError("Expected nine development sequences")
        for held in sorted(fold.sequence.unique()):
            train = development[development.sequence.ne(held)]
            test = fold[fold.sequence.eq(held)]
            role = "outer_test" if held == outer else "operating_point_calibration"
            if set(train.sequence) & {outer, held}:
                raise AssertionError("Sequence leakage")
            for method, (kind, strength) in VARIANTS.items():
                block = test[KEY].copy()
                block["outer_sequence"] = outer
                block["prediction_role"] = role
                block["actual_qualified"] = test.qualified.to_numpy()
                block["method"] = method
                block["score"] = scores(train, test, kind, strength)
                block["training_sequences"] = ";".join(sorted(train.sequence.unique()))
                output.append(block)
    return pd.concat(output, ignore_index=True)


def operating_points(predictions):
    rows = []
    for (outer, method), fold in predictions.groupby(["outer_sequence", "method"]):
        calibration = fold[fold.prediction_role.eq("operating_point_calibration")]
        test = fold[fold.prediction_role.eq("outer_test")]
        for target in np.round(np.arange(.10, .51, .05), 2):
            threshold, training = cutoff(calibration.score.to_numpy(), target)
            for scope, service in [("pooled", "all")] + [("service", s) for s in sorted(test.service.unique())]:
                group = test if service == "all" else test[test.service.eq(service)]
                accepted = group.score.ge(threshold)
                rows.append(dict(heldout_sequence=outer, method=method, target_acceptance=target,
                                 scope=scope, service=service, threshold=threshold,
                                 training_acceptance=training, accepted_count=int(accepted.sum()),
                                 false_count=int((accepted & group.actual_qualified.eq(0)).sum()),
                                 case_count=len(group)))
    per = pd.DataFrame(rows)
    rng = np.random.default_rng(20261016)
    summary = []
    for key, group in per.groupby(["method", "target_acceptance", "scope", "service"]):
        accepted, false = int(group.accepted_count.sum()), int(group.false_count.sum())
        indices = rng.integers(0, len(group), (5000, len(group)))
        a = group.accepted_count.to_numpy()[indices].sum(axis=1)
        f = group.false_count.to_numpy()[indices].sum(axis=1)
        draws = np.divide(f, a, out=np.full(5000, np.nan), where=a > 0)
        valid = draws[np.isfinite(draws)]
        low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        summary.append(dict(zip(["method", "target_acceptance", "scope", "service"], key),
                            accepted_count=accepted, false_count=false, case_count=int(group.case_count.sum()),
                            coverage=accepted / group.case_count.sum(), risk=false / accepted if accepted else np.nan,
                            ci_low=low, ci_high=high))
    return per, pd.DataFrame(summary)


def plot_buckets(predictions, methods, path):
    outer = predictions[predictions.prediction_role.eq("outer_test")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    plot_rows = []
    colors = dict(zip(LABELS, plt.get_cmap("tab10").colors))
    for method in methods:
        cases = outer[outer.method.eq(method)]
        buckets = cases.groupby("score").actual_qualified.agg(["sum", "count"]).sort_index(ascending=False)
        a = buckets["count"].cumsum().to_numpy()
        tp = buckets["sum"].cumsum().to_numpy()
        positive = int(cases.actual_qualified.sum())
        negative = len(cases) - positive
        if (len(cases), positive, negative) != (800, 390, 410):
            raise AssertionError("Truth counts differ across methods")
        x, fp, fn = 100 * a / 800, 100 * (a - tp) / negative, 100 * (positive - tp) / positive
        size = buckets["count"].to_numpy()
        for ax, y in zip(axes, (fp, fn)):
            ax.plot(x, y, color=colors[method], lw=1.7, label=f"{LABELS[method]} ({len(buckets)} groups)")
            ax.scatter(x, y, color=colors[method], s=12 + .7 * size, alpha=.65)
        plot_rows.extend(dict(method=method, score=score, accepted_count=int(accepted), bucket_size=int(count),
                              coverage_percent=coverage, false_positive_percent=fpr, false_negative_percent=fnr)
                         for score, accepted, count, coverage, fpr, fnr in zip(buckets.index, a, size, x, fp, fn))
    for ax, title in zip(axes, ("False accepts / 410 unqualified cases", "Rejected passes / 390 qualified cases")):
        ax.set(xlabel="Held-out cases accepted (%)", title=title, xlim=(0, 101), ylim=(-2, 102))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Error rate (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(.5, .04))
    fig.suptitle("Held-out score rankings: original surface and close alternatives", fontsize=14)
    fig.text(.5, .012, "Marker size shows cases sharing a score. Descriptive pooled cutoff sweep; training-selected results are in the operating-point table.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .19, 1, .95))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return pd.DataFrame(plot_rows)


def verify(table, predictions, baseline):
    if table.duplicated(["outer_sequence"] + KEY).any() or len(table) != 8000:
        raise AssertionError("Expected unique 10-fold, 10-sequence, 80-cell table")
    for outer, fold in table.groupby("outer_sequence"):
        train, test = fold[fold.sequence.ne(outer)], fold[fold.sequence.eq(outer)]
        expected = baseline[(baseline.outer_sequence.eq(outer)) & baseline.method.eq("service_empirical")
                            & baseline.prediction_role.eq("outer_test")].set_index(CELL).score
        empirical = train.groupby(CELL).qualified.mean().reindex(pd.MultiIndex.from_frame(test[CELL]))
        np.testing.assert_allclose(empirical, expected.reindex(pd.MultiIndex.from_frame(test[CELL])))
        for kind, strength in VARIANTS.values():
            original = scores(train, test, kind, strength)
            changed = test.copy()
            changed["qualified"] = 1 - changed.qualified
            np.testing.assert_array_equal(original, scores(train, changed, kind, strength))
            if not (np.isfinite(original).all() and ((original >= 0) & (original <= 1)).all()):
                raise AssertionError("Invalid scores")
        constant = train.copy()
        constant["qualified"] = 1
        np.testing.assert_allclose(scores(constant, test, "smooth", 1), 1)
        np.testing.assert_allclose(scores(constant, test, "shrink", 5), 1)
    for row in predictions.itertuples():
        excluded = {row.outer_sequence, row.sequence}
        if set(row.training_sequences.split(";")) & excluded:
            raise AssertionError("Prediction includes an excluded sequence")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("results/doubly_nested_loso_evidence"))
    parser.add_argument("--output", type=Path, default=Path("results/surface_alternatives"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [args.evidence_root / name for name in ("per_sequence_response_surface.csv", "heldout_predictions.csv")]
    table, baseline = (pd.read_csv(p) for p in sources)
    additions = build_predictions(table)
    verify(table, additions, baseline)
    predictions = pd.concat([baseline, additions], ignore_index=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    per, summary = operating_points(predictions)
    per.to_csv(args.output / "operating_points_per_sequence.csv", index=False)
    summary.to_csv(args.output / "operating_point_summary.csv", index=False)
    selected = summary[(summary.scope.eq("pooled")) & np.isclose(summary.target_acceptance, .25)]
    expected = {"service_empirical": (200, 4), "service_identity_only": (200, 75),
                "aoi_only": (160, 28), "common": (200, 30), "motion_logistic": (199, 55)}
    for method, counts in expected.items():
        row = selected[selected.method.eq(method)].iloc[0]
        if (row.accepted_count, row.false_count) != counts:
            raise AssertionError(f"Baseline mismatch: {method}")
    core = list(LABELS)[:5]
    plotted = plot_buckets(predictions, core + ["smooth_h1", "shrink_k5"], args.output / "all_methods_two_panel_buckets")
    plotted.to_csv(args.output / "plotted_bucket_data.csv", index=False)
    plot_buckets(predictions, core + ["smooth_h0.5", "smooth_h1"], args.output / "smoothing_sensitivity")
    plot_buckets(predictions, core + ["shrink_k2", "shrink_k5", "shrink_k10"], args.output / "pooling_sensitivity")
    manifest = dict(input_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    variants=VARIANTS, smoothing_coordinates="log2(rate_hz), delay_ms/100",
                    smoothing_formula="Gaussian weighted mean of same-service cell pass fractions; bandwidth h",
                    pooling_formula="(cell_passes + k * training_service_mean) / (n_sequences + k)",
                    parameter_selection="Fixed sensitivity grid; h=1 and k=5 displayed in main plot; no selection using test results",
                    cutoff_rule="Inner LOSO development scores; closest acceptance, keep ties, prefer higher cutoff",
                    bootstrap="5000 physical-sequence resamples, seed 20261016",
                    plot_scope="Descriptive sweep of pooled outer-test scores, not development-selected operating points",
                    verification="Original surface scores and 25% baseline counts reproduced; exclusion, target-perturbation, constant-score and range checks passed")
    (args.output / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["Surface alternatives on the saved doubly nested qualification tables", "",
             "25% requested acceptance; counts use frozen development-selected cutoffs:", ""]
    for row in selected.itertuples():
        lines.append(f"{LABELS[row.method]}: {row.false_count}/{row.accepted_count} false accepts; "
                     f"coverage {100*row.coverage:.2f}%; risk {100*row.risk:.2f}% "
                     f"(sequence-bootstrap interval {100*row.ci_low:.2f}--{100*row.ci_high:.2f}%).")
    lines += ["", "The plot is a descriptive pooled score sweep. The table uses separately chosen cutoffs in each outer fold; these are different procedures.",
              "Smoothing assumes neighboring rates/delays share information. Pooling pulls each cell toward its service mean.",
              "All parameter settings are reported. The displayed h=1 and k=5 were fixed before inspecting outputs.",
              "Inputs are existing seed-aggregated labels; no retraining or manuscript edits were performed."]
    (args.output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

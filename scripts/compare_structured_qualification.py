"""Nested comparison with service-specific parametric and tree predictors."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_surface_alternatives import operating_points


def design(data):
    rate, delay = np.log2(data.rate_hz.to_numpy()), data.delay_ms.to_numpy() / 100.
    return np.column_stack([rate, delay, rate * delay, rate ** 2, delay ** 2])


def score(train, test, method):
    output = np.empty(len(test))
    for service in sorted(test.service.unique()):
        fit = train[train.service.eq(service)]
        mask = test.service.eq(service).to_numpy()
        query = test.loc[mask]
        if fit.qualified.nunique() == 1:
            output[mask] = fit.qualified.iloc[0]
            continue
        if method == "service_specific_logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=42))
            model.fit(design(fit), fit.qualified)
            output[mask] = model.predict_proba(design(query))[:, 1]
        else:
            model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2 * train.sequence.nunique(), random_state=42)
            model.fit(fit[["rate_hz", "delay_ms"]], fit.qualified)
            output[mask] = model.predict_proba(query[["rate_hz", "delay_ms"]])[:, 1]
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("results/doubly_nested_loso_evidence"))
    parser.add_argument("--output", type=Path, default=Path("results/structured_qualification_baselines/centered"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.evidence_root / "per_sequence_response_surface.csv"
    baseline_path = args.evidence_root / "heldout_predictions.csv"
    table = pd.read_csv(source)
    if len(table) != 8000 or table.duplicated(["outer_sequence", "sequence", "service", "rate_hz", "delay_ms"]).any():
        raise ValueError("Expected complete nested sequence table")
    blocks = []
    for outer, fold in table.groupby("outer_sequence"):
        print(outer, flush=True)
        development = fold[fold.sequence.ne(outer)]
        if development.sequence.nunique() != 9:
            raise ValueError("Expected nine development recordings")
        for held in sorted(fold.sequence.unique()):
            fit = development[development.sequence.ne(held)]
            test = fold[fold.sequence.eq(held)]
            if {outer, held} & set(fit.sequence):
                raise ValueError("Sequence leakage")
            for method in ["service_specific_logistic", "service_specific_tree"]:
                block = test[["sequence", "service", "rate_hz", "delay_ms"]].copy()
                block["outer_sequence"] = outer
                block["prediction_role"] = "outer_test" if held == outer else "operating_point_calibration"
                block["actual_qualified"] = test.qualified.to_numpy()
                block["method"] = method
                block["score"] = score(fit, test, method)
                block["training_sequences"] = ";".join(sorted(fit.sequence.unique()))
                blocks.append(block)
    baseline = pd.read_csv(baseline_path)
    new = pd.concat(blocks, ignore_index=True)
    reference = baseline[baseline.method.eq("service_empirical")].set_index(["outer_sequence", "prediction_role", "sequence", "service", "rate_hz", "delay_ms"])
    for _, group in new.groupby("method"):
        indexed = group.set_index(reference.index.names)
        np.testing.assert_array_equal(indexed.actual_qualified, reference.reindex(indexed.index).actual_qualified)
    predictions = pd.concat([baseline, new], ignore_index=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    per, summary = operating_points(predictions)
    per.to_csv(args.output / "per_sequence_operating_points.csv", index=False)
    summary.to_csv(args.output / "operating_point_summary.csv", index=False)
    selected = summary[summary.scope.eq("pooled") & np.isclose(summary.target_acceptance, .25)].copy()
    selected.to_csv(args.output / "target25.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    order = ["service_empirical", "service_specific_logistic", "service_specific_tree", "motion_logistic", "aoi_only", "common", "service_identity_only"]
    labels = ["Original surface", "Service-specific logistic", "Service-specific shallow tree", "Motion logistic", "AoI only", "Common delivery", "Service identity"]
    selected = selected.set_index("method").loc[order]
    for i, row in enumerate(selected.itertuples()):
        ax.plot(100 * np.array([row.ci_low, row.ci_high]), [i, i], color="#757575", lw=2)
        ax.scatter(100 * row.risk, i, color="#168477" if i == 0 else "#2479ac", s=45, zorder=3)
        ax.text(1.02, i, f"{row.false_count}/{row.accepted_count}; {100 * row.coverage:.1f}% accepted", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.set(yticks=range(len(order)), yticklabels=labels, xlabel="False qualifications among accepted cases (%)", title="25% requested acceptance; cutoffs chosen on development recordings")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout(rect=(0, 0, .78, 1))
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"target25_comparison.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    manifest = dict(inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, baseline_path, Path(__file__)]},
                    models={"service_specific_logistic": "One per service; standardized log2(rate), delay/100, product, and both squared terms; C=1; balanced classes; max_iter=2000",
                            "service_specific_tree": "One per service; rate/delay; max_depth=3; min_samples_leaf=2*n_training_recordings; unweighted class counts"},
                    parameter_selection="Fixed before outcomes inspected; no hyperparameter selection on outer labels",
                    evaluation="Original inner-LOSO acceptance-only cutoff selection and tie handling; pooled cutoffs; sequence bootstrap 5000",
                    purpose="Qualification-predictor comparison within the framework, not replacement fidelity frameworks")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(selected[["accepted_count", "false_count", "coverage", "risk", "ci_low", "ci_high"]].to_string())


if __name__ == "__main__":
    main()

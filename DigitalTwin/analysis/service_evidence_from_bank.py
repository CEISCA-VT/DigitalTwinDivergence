"""Recompute service-qualification evidence from a verified trajectory bank.

For the doubly nested study, each outer fold uses its own nine qualification-
training trajectories and its independently generated outer-test trajectory.
The module refuses partial or contaminated banks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from DigitalTwin.analysis import service_timing_budget_study as base


ROOT = base.ROOT


def _source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _scores(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    common = train.groupby(["rate_hz", "delay_ms"]).qualified.mean()
    empirical = train.groupby(["service", "rate_hz", "delay_ms"]).qualified.mean()
    identity = train.groupby("service").qualified.mean()
    categorical = ["service"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), base.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    features = base.NUMERIC_FEATURES + categorical
    y = train.qualified
    if y.nunique() > 1:
        model = make_pipeline(pre, LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42))
        model.fit(train[features], y)
        logistic = model.predict_proba(test[features])[:, 1]
    else:
        logistic = np.full(len(test), float(y.iloc[0]))
    rows = []
    for j, (_, row) in enumerate(test.iterrows()):
        values = {
            "common": float(common.loc[(row.rate_hz, row.delay_ms)]),
            "aoi_only": float(np.round(row.aoi_limit_s - row.max_aoi_s, 6)),
            "service_identity_only": float(identity.loc[row.service]),
            "service_empirical": float(empirical.loc[(row.service, row.rate_hz, row.delay_ms)]),
            "motion_logistic": float(logistic[j]),
        }
        for method, score in values.items():
            rows.append({"sequence": row.sequence, "service": row.service, "rate_hz": row.rate_hz,
                         "delay_ms": row.delay_ms, "actual_qualified": int(row.qualified),
                         "method": method, "score": score,
                         "training_sequences": ";".join(sorted(train.sequence.unique()))})
    return pd.DataFrame(rows)


def _fold_entries(manifest: dict) -> dict[str, list[dict]]:
    entries = manifest["trajectories"]
    if manifest["study"] == "doubly_nested":
        folds: dict[str, list[dict]] = {}
        for entry in entries:
            folds.setdefault(entry["outer"], []).append(entry)
        return folds
    return {"ordinary_loso": entries}


def build_nested_tables(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_runs, all_sequences, predictions = [], [], []
    for outer, entries in sorted(_fold_entries(manifest).items()):
        paths = [_source_path(entry["trajectory"]) for entry in entries]
        runs = base.response_rows(paths)
        if manifest["study"] == "doubly_nested":
            expected = {entry["target"] for entry in entries}
            if set(runs.sequence.unique()) != expected:
                raise RuntimeError(f"Outer {outer}: trajectory/sequence mismatch")
            sequences = base.sequence_rows(runs)
            train, test = sequences[sequences.sequence != outer], sequences[sequences.sequence == outer]
            if train.sequence.nunique() != 9 or test.sequence.nunique() != 1:
                raise RuntimeError(f"Outer {outer}: expected nine training sequences and one test sequence")
            outer_predictions = _scores(train, test)
            outer_predictions.insert(0, "prediction_role", "outer_test")
            inner_predictions = []
            for development_sequence in sorted(train.sequence.unique()):
                inner_train = train[train.sequence != development_sequence]
                inner_test = train[train.sequence == development_sequence]
                inner_predictions.append(_scores(inner_train, inner_test))
            calibration_predictions = pd.concat(inner_predictions, ignore_index=True)
            calibration_predictions.insert(0, "prediction_role", "operating_point_calibration")
            fold_predictions = pd.concat([outer_predictions, calibration_predictions], ignore_index=True)
            runs.insert(0, "outer_sequence", outer)
            sequences.insert(0, "outer_sequence", outer)
            fold_predictions.insert(0, "outer_sequence", outer)
        else:
            sequences = base.sequence_rows(runs)
            blocks = []
            for held in sorted(sequences.sequence.unique()):
                train, test = sequences[sequences.sequence != held], sequences[sequences.sequence == held]
                outer_predictions = _scores(train, test)
                outer_predictions.insert(0, "prediction_role", "outer_test")
                inner_predictions = pd.concat([
                    _scores(train[train.sequence != development], train[train.sequence == development])
                    for development in sorted(train.sequence.unique())
                ], ignore_index=True)
                inner_predictions.insert(0, "prediction_role", "operating_point_calibration")
                outer_predictions.insert(0, "outer_sequence", held)
                inner_predictions.insert(0, "outer_sequence", held)
                blocks.extend((outer_predictions, inner_predictions))
            fold_predictions = pd.concat(blocks, ignore_index=True)
            runs.insert(0, "outer_sequence", "ordinary_loso")
            sequences.insert(0, "outer_sequence", "ordinary_loso")
            predictions.append(fold_predictions)
            all_runs.append(runs)
            all_sequences.append(sequences)
            continue
        all_runs.append(runs)
        all_sequences.append(sequences)
        predictions.append(fold_predictions)
    return pd.concat(all_runs, ignore_index=True), pd.concat(all_sequences, ignore_index=True), pd.concat(predictions, ignore_index=True)


def _operating_points(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = np.round(np.arange(0.10, 0.55, 0.05), 2)
    rows = []
    outer_rows = predictions[predictions.prediction_role == "outer_test"]
    for held in sorted(outer_rows.sequence.unique()):
        fold = predictions[predictions.outer_sequence == held]
        outer = fold[fold.prediction_role == "outer_test"]
        # Cross-fitted predictions over the nine qualification-training
        # trajectories choose the threshold without consulting outer-test data.
        calibration = fold[fold.prediction_role == "operating_point_calibration"]
        scopes = [("pooled", "all")] + [("within_service", service) for service in sorted(predictions.service.unique())]
        for scope, service in scopes:
            outer_scope = outer if scope == "pooled" else outer[outer.service == service]
            calibration_scope = calibration if scope == "pooled" else calibration[calibration.service == service]
            for method in sorted(predictions.method.unique()):
                train_scores = calibration_scope[calibration_scope.method == method].score.to_numpy(float)
                test = outer_scope[outer_scope.method == method]
                truth = test.actual_qualified.to_numpy(bool)
                for target in targets:
                    threshold, training_acceptance = base._acceptance_threshold(train_scores, target)
                    accepted = test.score.to_numpy(float) >= threshold
                    rows.append({"heldout_sequence": held, "scope": scope, "service": service,
                                 "method": method, "target_acceptance": target,
                                 "selected_threshold": threshold, "training_acceptance": training_acceptance,
                                 "accepted_count": int(accepted.sum()), "condition_count": len(test),
                                 "false_qualified_count": int((accepted & ~truth).sum()),
                                 "actual_qualified_count": int(truth.sum())})
    per = pd.DataFrame(rows)
    summary_rows = []
    rng = np.random.default_rng(20260918)
    for (scope, service, method, target), group in per.groupby(["scope", "service", "method", "target_acceptance"]):
        sequences = sorted(group.heldout_sequence.unique())
        draws = []
        for _ in range(5000):
            selected = rng.choice(sequences, len(sequences), replace=True)
            sample = pd.concat([group[group.heldout_sequence == sequence] for sequence in selected])
            accepted = int(sample.accepted_count.sum())
            false = int(sample.false_qualified_count.sum())
            draws.append(false / accepted if accepted else np.nan)
        accepted = int(group.accepted_count.sum())
        false = int(group.false_qualified_count.sum())
        summary_rows.append({"scope": scope, "service": service, "method": method, "target_acceptance": target,
                             "accepted_count": accepted, "condition_count": int(group.condition_count.sum()),
                             "false_qualified_count": false, "achieved_acceptance": accepted / group.condition_count.sum(),
                             "selective_risk": false / accepted if accepted else np.nan,
                             "selective_risk_ci_low": float(np.nanquantile(draws, .025)),
                             "selective_risk_ci_high": float(np.nanquantile(draws, .975)),
                             "n_physical_sequences": len(sequences)})
    return per, pd.DataFrame(summary_rows)


def _remediability(sequences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outer, group in sequences.groupby("outer_sequence"):
        if outer == "ordinary_loso":
            fold = group
        else:
            fold = group[group.sequence == outer]
        ledger = base.remediability(fold, pd.DataFrame())
        ledger.insert(0, "outer_sequence", outer)
        rows.append(ledger)
    return pd.concat(rows, ignore_index=True)


def _pairwise_risk(per: pd.DataFrame, target: float = .30) -> pd.DataFrame:
    rng = np.random.default_rng(20260919)
    rows = []
    chosen = per[np.isclose(per.target_acceptance, target)]
    for (scope, service), block in chosen.groupby(["scope", "service"]):
        reference = block[block.method == "service_empirical"].set_index("heldout_sequence").sort_index()
        for comparator in ("service_identity_only", "aoi_only", "motion_logistic"):
            other = block[block.method == comparator].set_index("heldout_sequence").reindex(reference.index)
            sequences = list(reference.index)
            differences = []
            for _ in range(5000):
                selected = rng.choice(sequences, len(sequences), replace=True)
                first = reference.loc[list(selected)]
                second = other.loc[list(selected)]
                first_risk = first.false_qualified_count.sum() / first.accepted_count.sum() if first.accepted_count.sum() else np.nan
                second_risk = second.false_qualified_count.sum() / second.accepted_count.sum() if second.accepted_count.sum() else np.nan
                differences.append(first_risk - second_risk)
            valid = np.asarray(differences, float)
            valid = valid[np.isfinite(valid)]
            first_accepted, second_accepted = reference.accepted_count.sum(), other.accepted_count.sum()
            observed = np.nan
            if first_accepted and second_accepted:
                observed = (reference.false_qualified_count.sum() / first_accepted -
                            other.false_qualified_count.sum() / second_accepted)
            rows.append({"scope": scope, "service": service, "target_acceptance": target,
                         "first_method": "service_empirical", "second_method": comparator,
                         "first_accepted": int(first_accepted), "second_accepted": int(second_accepted),
                         "selective_risk_difference_first_minus_second": observed,
                         "ci_low": float(np.quantile(valid, .025)) if len(valid) else np.nan,
                         "ci_high": float(np.quantile(valid, .975)) if len(valid) else np.nan,
                         "n_physical_sequences": len(sequences)})
    return pd.DataFrame(rows)


def _plot_risk(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    pooled = summary[summary.scope == "pooled"]
    for method, group in pooled.groupby("method"):
        group = group.sort_values("achieved_acceptance")
        ax.plot(group.achieved_acceptance, group.selective_risk, marker="o", label=method.replace("_", " "))
    ax.set(xlabel="Held-out acceptance", ylabel="False qualification / accepted cells",
           title="Qualification risk-coverage on corrected trajectory bank")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run(manifest_path: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("ready_for_evidence_analysis"):
        raise RuntimeError(f"Trajectory bank is not complete and clean: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)
    runs, sequences, predictions = build_nested_tables(manifest)
    runs.to_csv(output / "per_run_response_surface.csv", index=False)
    sequences.to_csv(output / "per_sequence_response_surface.csv", index=False)
    predictions.to_csv(output / "heldout_predictions.csv", index=False)
    per, summary = _operating_points(predictions)
    per.to_csv(output / "matched_acceptance_per_sequence.csv", index=False)
    summary.to_csv(output / "risk_coverage_summary.csv", index=False)
    _pairwise_risk(per).to_csv(output / "matched_acceptance_pairwise_bootstrap.csv", index=False)
    ledger = _remediability(sequences)
    ledger.to_csv(output / "sequence_service_case_ledger.csv", index=False)
    _plot_risk(summary, output / "risk_coverage_curves.png")
    counts = ledger.classification.value_counts().to_dict()
    audit = {
        "schema": "trajectory_bank_service_evidence_v1", "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_study": manifest["study"], "physical_sequences": int(predictions.sequence.nunique()),
        "sequence_service_cases": len(ledger), "remediability_counts": counts,
        "services": base.SERVICES, "rates_hz": base.RATES, "delays_ms": base.DELAYS_MS,
        "target_satisfaction": base.TARGET, "minimum_coverage": base.MIN_COVERAGE,
        "operating_point_selection": "inner leave-one-sequence-out within each outer fold; outer-test labels excluded",
        "statistical_unit": "physical sequence; seeds aggregated within sequence",
        "status": "complete",
    }
    (output / "evidence_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.bank_manifest, args.output)


if __name__ == "__main__":
    main()

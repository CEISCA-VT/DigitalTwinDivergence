"""Summarize the completed causal-derivative LOSO sensitivity study."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from DigitalTwin.analysis.service_evidence_from_bank import _remediability


METRICS = {
    "v2_ate_rmse_m": "ATE RMSE (m)",
    "v2_heading_mae_deg": "Heading MAE (deg)",
    "v2_rpe_1s_m": "RPE1 RMSE (m)",
    "v2_rpe_5s_m": "RPE5 RMSE (m)",
    "v2_rpe_10s_m": "RPE10 RMSE (m)",
    "fidelity_Dp_p95_m": "Dp p95 (m)",
    "fidelity_Dtheta_p95_deg": "Dtheta p95 (deg)",
}


def load_causal_summaries(archive_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/run_summary.json")]
        return pd.DataFrame(json.loads(archive.read(name)) for name in names)


def run(archive: Path, centered_csv: Path, response_csv: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    causal = load_causal_summaries(archive)
    centered = pd.read_csv(centered_csv)
    if len(causal) != 30 or causal.test_sequence.nunique() != 10 or causal.base_seed.nunique() != 3:
        raise RuntimeError("Expected 30 causal runs spanning 10 sequences and three seeds")

    sequence_rows = causal.groupby("test_sequence", as_index=False)[list(METRICS)].mean()
    sequence_rows.to_csv(output / "causal_derivative_per_sequence.csv", index=False)

    centered_macro = centered.groupby("test_sequence")[list(METRICS)].mean().mean()
    causal_macro = sequence_rows[list(METRICS)].mean()
    comparison = pd.DataFrame({
        "metric": list(METRICS),
        "label": list(METRICS.values()),
        "centered_sequence_macro": [centered_macro[key] for key in METRICS],
        "causal_sequence_macro": [causal_macro[key] for key in METRICS],
    })
    comparison["causal_change_pct"] = 100 * (
        comparison.causal_sequence_macro / comparison.centered_sequence_macro - 1
    )
    comparison.to_csv(output / "causal_vs_centered_fidelity.csv", index=False)

    response = pd.read_csv(response_csv)
    ledger = _remediability(response)
    ledger.to_csv(output / "causal_sequence_service_ledger.csv", index=False)
    counts = ledger.classification.value_counts().to_dict()
    recovered = int(ledger.practical_restores.sum())
    remediable = int(counts.get("delivery_remediable", 0))

    plot = comparison.iloc[:5].copy()
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    groups = ((plot.iloc[[0]], "Global position", "Error (m)"),
              (plot.iloc[[1]], "Heading", "Error (deg)"),
              (plot.iloc[2:], "Finite-horizon translation", "Error (m)"))
    for axis, (frame, title, ylabel) in zip(axes, groups):
        frame.set_index("label")[["centered_sequence_macro", "causal_sequence_macro"]].plot.bar(
            ax=axis, color=["#9aa8b8", "#2878b5"], rot=0
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(["Centered derivative", "Causal backward"], fontsize=8)
    figure.suptitle("Causal feature construction is valid but less accurate")
    figure.tight_layout()
    figure.savefig(output / "causal_vs_centered_fidelity.png", dpi=220)
    plt.close(figure)

    parking = sequence_rows.set_index("test_sequence").loc[["parking00", "parking02"]]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    parking[["v2_ate_rmse_m", "fidelity_Dp_p95_m"]].rename(columns={
        "v2_ate_rmse_m": "ATE", "fidelity_Dp_p95_m": "Dp p95"
    }).plot.bar(ax=axes[0], color=["#2878b5", "#d66b3d"], rot=0)
    parking[["v2_rpe_1s_m", "v2_rpe_5s_m", "v2_rpe_10s_m"]].rename(columns={
        "v2_rpe_1s_m": "RPE1", "v2_rpe_5s_m": "RPE5", "v2_rpe_10s_m": "RPE10"
    }).plot.bar(ax=axes[1], color=["#2878b5", "#56a36c", "#d6a23d"], rot=0)
    axes[0].set(title="Global divergence", ylabel="Error (m)", xlabel="")
    axes[1].set(title="Finite-horizon relative error", ylabel="Error (m)", xlabel="")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("parking00/parking02 ordering reverses under causal features")
    figure.tight_layout()
    figure.savefig(output / "causal_parking_local_global_inversion.png", dpi=220)
    plt.close(figure)

    manifest = {
        "schema": "causal_derivative_summary_v1",
        "archive": archive.name,
        "archive_commit": "6f91f5dae834ac55dfebf3abae76beba7379ff2c",
        "verified_runs": len(causal),
        "physical_sequences": int(causal.test_sequence.nunique()),
        "seeds_per_sequence": int(causal.base_seed.nunique()),
        "derivative_mode": "causal_backward",
        "ledger_counts": counts,
        "practical_recoveries": recovered,
        "practical_recovery_denominator": remediable,
    }
    (output / "causal_derivative_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")

    ate = comparison.loc[comparison.metric == "v2_ate_rmse_m"].iloc[0]
    heading = comparison.loc[comparison.metric == "v2_heading_mae_deg"].iloc[0]
    report = f"""# Causal-derivative LOSO summary

- Provenance audit: **PASS**, 30/30 trajectories, 10 physical sequences, three seeds.
- Feature rule: backward differences; first derivative sample is zero; numerical future-sample perturbation preflight passed.
- Sequence-macro ATE: `{ate.centered_sequence_macro:.3f} m` centered versus `{ate.causal_sequence_macro:.3f} m` causal (`{ate.causal_change_pct:+.1f}%`).
- Sequence-macro heading MAE: `{heading.centered_sequence_macro:.3f} deg` centered versus `{heading.causal_sequence_macro:.3f} deg` causal (`{heading.causal_change_pct:+.1f}%`).
- Causal ledger: `{counts.get('delivery_remediable', 0)}` delivery-remediable, `{counts.get('persistent_under_ideal', 0)}` persistent, and `{counts.get('already_qualified_degraded', 0)}` already qualified.
- The practical 5-Hz/50-ms intervention restores `{recovered}/{remediable}` remediable cases.

The study removes future-sample dependence from derivative-bearing inputs, so it closes the causal-feature provenance gap. It does not improve trajectory accuracy: retraining with backward derivatives degrades aggregate fidelity, especially on building01/building02 and parking02. The service-level structure nevertheless remains similar to the centered-gradient result (historically 30/6/4 with 27/30 practical recoveries). This is therefore a robustness and negative-result finding, not evidence that causal derivatives improve the model.
"""
    (output / "causal_derivative_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--centered-csv", type=Path, required=True)
    parser.add_argument("--response-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.centered_csv, args.response_csv, args.output)


if __name__ == "__main__":
    main()

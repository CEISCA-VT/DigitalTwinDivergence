#!/usr/bin/env python3
"""
Stage 1 publication hardening for the service-relative digital-twin fidelity paper.

Run from the DigitalTwinDivergence repository root:

    python stage1_publication_hardening.py

Optional:
    python stage1_publication_hardening.py --bootstrap 20000
    python stage1_publication_hardening.py --allow-download
    python stage1_publication_hardening.py --skip-e3
    python stage1_publication_hardening.py --self-test

Outputs:
    results/stage1_publication_hardening/

Scientific boundary:
- Does NOT retrain any twin.
- Does NOT change the frozen E1/E2/E3 primary results.
- Does NOT select service thresholds to improve an outcome.
- Alternative normalizations are sensitivity analyses only.
- Sequence is the resampling unit for E1 uncertainty.
- Timing audit does not invent missing protocol details; unresolved details are
  explicitly marked for correction before manuscript submission.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import kendalltau, spearmanr
except Exception as exc:
    raise SystemExit(
        "scipy is required. Install with: python -m pip install numpy pandas scipy matplotlib"
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


BOOTSTRAP_SEED = 20260825
DEFAULT_BOOTSTRAP = 10000

SERVICE_SPECS = [
    ("global", "global_synchronization", 0.0, "ate_m"),
    ("local_1s", "local_relative_motion", 1.0, "rpe1_m"),
    ("local_5s", "local_relative_motion", 5.0, "rpe5_m"),
    ("local_10s", "local_relative_motion", 10.0, "rpe10_m"),
]

EXPECTED_E3_DATASETS = {"MAGNET", "FreeTwinEV_1S4P", "TUWien_SNG"}


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _clean(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _rel(path: Path | None, repo: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except Exception:
        return str(path)


def _latest(paths: Iterable[Path]) -> Path | None:
    valid = [p for p in paths if p.exists()]
    if not valid:
        return None
    return max(valid, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def _find_preferred(repo: Path, preferred: list[Path], pattern: str) -> Path | None:
    for p in preferred:
        if p.exists():
            return p
    candidates = list(repo.glob(pattern))
    if not candidates:
        candidates = list(repo.rglob(Path(pattern).name))
    return _latest(candidates)


def _read_csv(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [str(c).strip() for c in d.columns]
    return d


def _require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}. Found: {list(df.columns)}")


def _kendall(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(kendalltau(x, y, nan_policy="omit").statistic)


def _pct_ci(vals: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return (float("nan"), float("nan"))
    return tuple(np.quantile(v, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist())


def _write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _bool(v: object) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"true", "1", "yes", "pass"}


# ---------------------------------------------------------------------------
# E1 discovery
# ---------------------------------------------------------------------------

def discover_e1(repo: Path) -> dict[str, Path]:
    e1_dir = _find_preferred(
        repo,
        [
            repo / "results" / "e1_e2_service_contract_publication" / "E1_i2nav",
        ],
        "results/**/E1_i2nav",
    )
    if e1_dir is None or not e1_dir.is_dir():
        raise FileNotFoundError(
            "Could not locate E1_i2nav output directory. Expected "
            "results/e1_e2_service_contract_publication/E1_i2nav"
        )

    needed = {
        "grid_average": e1_dir / "e1_grid_average_service_validity.csv",
        "rank_summary": e1_dir / "e1_baseline_rank_alignment_summary.csv",
        "parking_dominance": e1_dir / "e1_parking_inversion_dominance_summary.csv",
        "parking_grid": e1_dir / "e1_parking00_parking02_full_grid.csv",
        "pairwise": e1_dir / "e1_pairwise_service_ordering.csv",
    }
    for k, p in needed.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing E1 input {k}: {p}")

    raw_metrics = _find_preferred(
        repo,
        [
            repo / "results" / "service_relative_fidelity" / "raw_recomputation_vs_frozen_summary.csv",
            repo / "raw_recomputation_vs_frozen_summary.csv",
        ],
        "results/**/raw_recomputation_vs_frozen_summary.csv",
    )
    if raw_metrics is not None:
        needed["raw_metrics"] = raw_metrics
    needed["e1_dir"] = e1_dir
    return needed


# ---------------------------------------------------------------------------
# 1A: scalar vs metrics vs contract
# ---------------------------------------------------------------------------

def run_1a(repo: Path, out: Path, e1: dict[str, Path]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rank = _read_csv(e1["rank_summary"])
    dom = _read_csv(e1["parking_dominance"])

    _require_columns(
        rank,
        ["family", "horizon_s", "baseline_metric", "median_kendall_tau"],
        "E1 rank summary",
    )

    rows = []
    for service, family, horizon, matched_metric in SERVICE_SPECS:
        g = rank[(rank["family"] == family) & np.isclose(rank["horizon_s"], horizon)]
        ate = g[g["baseline_metric"] == "ate_m"]
        matched = g[g["baseline_metric"] == matched_metric]
        if ate.empty or matched.empty:
            raise RuntimeError(
                f"Could not construct baseline comparison for {service}. "
                f"Need ate_m and {matched_metric} in {e1['rank_summary']}"
            )
        a = float(ate.iloc[0]["median_kendall_tau"])
        m = float(matched.iloc[0]["median_kendall_tau"])
        rows.append(
            {
                "service": service,
                "family": family,
                "horizon_s": horizon,
                "single_scalar_baseline": "ATE",
                "ate_only_median_kendall_tau_with_contract_ranking": a,
                "service_matched_standard_metric": matched_metric,
                "matched_metric_median_kendall_tau_with_contract_ranking": m,
                "matched_minus_ate_alignment": m - a,
                "contract_output": "explicit service-specific satisfaction / validity decision",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "1A_metric_vs_contract_alignment.csv", index=False)

    actionability = pd.DataFrame(
        [
            {
                "approach": "single scalar (ATE only)",
                "reports_measurement": True,
                "service_quantity_explicit": False,
                "horizon_explicit": False,
                "tolerance_explicit": False,
                "state_age_explicit": False,
                "direct_service_decision": False,
                "interpretation": "one ranking; appropriate mainly for global synchronized error",
            },
            {
                "approach": "standard metric suite (ATE + RPE)",
                "reports_measurement": True,
                "service_quantity_explicit": False,
                "horizon_explicit": True,
                "tolerance_explicit": False,
                "state_age_explicit": False,
                "direct_service_decision": False,
                "interpretation": "multiple complementary errors; application semantics remain implicit",
            },
            {
                "approach": "service-relative contract",
                "reports_measurement": True,
                "service_quantity_explicit": True,
                "horizon_explicit": True,
                "tolerance_explicit": True,
                "state_age_explicit": True,
                "direct_service_decision": True,
                "interpretation": "makes conventional metrics actionable for the specified service",
            },
        ]
    )
    actionability.to_csv(out / "1A_evaluation_layer_actionability.csv", index=False)

    parking_rows = []
    raw = None
    if "raw_metrics" in e1:
        raw = _read_csv(e1["raw_metrics"])
        if {"sequence", "metric", "recomputed"}.issubset(raw.columns):
            mp = raw.pivot_table(index="sequence", columns="metric", values="recomputed", aggfunc="first")
        else:
            mp = None
    else:
        mp = None

    for service, family, horizon, matched_metric in SERVICE_SPECS:
        dg = dom[(dom["family"] == family) & np.isclose(dom["horizon_s"], horizon)]
        if dg.empty:
            continue
        dr = dg.iloc[0]
        if int(dr["parking02_wins"]) == int(dr["n_grid_points"]):
            contract_pref = "parking02"
        elif int(dr["parking00_wins"]) == int(dr["n_grid_points"]):
            contract_pref = "parking00"
        else:
            contract_pref = "mixed"

        row = {
            "service": service,
            "contract_preference_over_full_grid": contract_pref,
            "contract_grid_points": int(dr["n_grid_points"]),
            "parking00_contract_wins": int(dr["parking00_wins"]),
            "parking02_contract_wins": int(dr["parking02_wins"]),
            "mean_contract_satisfaction_difference_parking02_minus_parking00":
                float(dr["mean_validity_difference"]),
            "ate_only_preference": "",
            "matched_standard_metric": matched_metric,
            "matched_metric_preference": "",
        }
        if mp is not None and {"parking00", "parking02"}.issubset(mp.index):
            if "ate_m" in mp.columns:
                row["ate_only_preference"] = (
                    "parking00" if float(mp.loc["parking00", "ate_m"]) < float(mp.loc["parking02", "ate_m"])
                    else "parking02"
                )
            if matched_metric in mp.columns:
                row["matched_metric_preference"] = (
                    "parking00"
                    if float(mp.loc["parking00", matched_metric]) < float(mp.loc["parking02", matched_metric])
                    else "parking02"
                )
        parking_rows.append(row)
    parking = pd.DataFrame(parking_rows)
    parking.to_csv(out / "1A_parking_decision_example.csv", index=False)

    if plt is not None:
        x = np.arange(len(comparison))
        width = 0.36
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        ax.bar(
            x - width / 2,
            comparison["ate_only_median_kendall_tau_with_contract_ranking"],
            width,
            label="ATE-only scalar",
        )
        ax.bar(
            x + width / 2,
            comparison["matched_metric_median_kendall_tau_with_contract_ranking"],
            width,
            label="service-matched standard metric",
        )
        ax.set_xticks(x, ["global", "local 1 s", "local 5 s", "local 10 s"])
        ax.set_ylim(-0.05, 1.0)
        ax.set_ylabel("Median Kendall tau with contract ranking")
        ax.set_title("One scalar does not rank all service contracts equally")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "1A_service_level_baseline_comparison.png", dpi=200)
        plt.close(fig)

    local_gain = comparison[comparison["family"] == "local_relative_motion"]["matched_minus_ate_alignment"]
    return {
        "status": "COMPLETE",
        "evidence": {
            "minimum_local_alignment_gain_matched_over_ate": float(local_gain.min()),
            "maximum_local_alignment_gain_matched_over_ate": float(local_gain.max()),
            "parking_contract_preferences": parking[
                ["service", "contract_preference_over_full_grid"]
            ].to_dict("records"),
        },
        "source_rank_summary": _rel(e1["rank_summary"], repo),
        "source_raw_metrics": _rel(e1.get("raw_metrics"), repo),
    }


# ---------------------------------------------------------------------------
# 1B: sequence-level uncertainty and leave-one-sequence-out robustness
# ---------------------------------------------------------------------------

def _pairwise_inversions(local: np.ndarray, global_: np.ndarray) -> tuple[int, int, int]:
    inv = conc = ties = 0
    n = len(local)
    for i in range(n):
        for j in range(i + 1, n):
            ld = float(local[i] - local[j])
            gd = float(global_[i] - global_[j])
            if abs(ld) < 1e-15 or abs(gd) < 1e-15:
                ties += 1
            elif ld * gd < 0:
                inv += 1
            else:
                conc += 1
    return inv, conc, ties


def run_1b(repo: Path, out: Path, e1: dict[str, Path], n_boot: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    g = _read_csv(e1["grid_average"])
    _require_columns(
        g,
        ["sequence", "family", "horizon_s", "grid_average_valid_fraction"],
        "E1 grid-average validity",
    )
    seqs = sorted(g["sequence"].astype(str).unique())
    if len(seqs) != 10:
        raise RuntimeError(f"E1 uncertainty expects 10 sequence units; found {len(seqs)}: {seqs}")

    piv = g.pivot_table(
        index="sequence",
        columns=["family", "horizon_s"],
        values="grid_average_valid_fraction",
        aggfunc="first",
    ).reindex(seqs)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idx = rng.integers(0, len(seqs), size=(n_boot, len(seqs)))

    mean_rows = []
    for service, family, horizon, _ in SERVICE_SPECS:
        vals = piv[(family, horizon)].to_numpy(float)
        boots = np.mean(vals[boot_idx], axis=1)
        lo, hi = _pct_ci(boots)
        mean_rows.append(
            {
                "service": service,
                "n_sequences": len(vals),
                "sequence_mean_grid_average_contract_satisfaction": float(np.mean(vals)),
                "bootstrap_median": float(np.median(boots)),
                "bootstrap_95ci_low": lo,
                "bootstrap_95ci_high": hi,
                "bootstrap_replicates": n_boot,
                "resampling_unit": "physical sequence",
            }
        )
    mean_df = pd.DataFrame(mean_rows)
    mean_df.to_csv(out / "1B_sequence_bootstrap_contract_satisfaction.csv", index=False)

    global_vals = piv[("global_synchronization", 0.0)].to_numpy(float)
    tau_rows = []
    loo_rows = []

    for horizon in (1.0, 5.0, 10.0):
        local_vals = piv[("local_relative_motion", horizon)].to_numpy(float)
        original_tau = _kendall(local_vals, global_vals)
        tau_boot = np.empty(n_boot, float)
        tau_boot[:] = np.nan
        for b in range(n_boot):
            idx = boot_idx[b]
            tau_boot[b] = _kendall(local_vals[idx], global_vals[idx])
        lo, hi = _pct_ci(tau_boot)
        valid = tau_boot[np.isfinite(tau_boot)]

        inv, conc, ties = _pairwise_inversions(local_vals, global_vals)
        tau_rows.append(
            {
                "local_horizon_s": int(horizon),
                "original_kendall_tau_local_vs_global": original_tau,
                "bootstrap_median_kendall_tau": float(np.median(valid)) if len(valid) else np.nan,
                "bootstrap_95ci_low": lo,
                "bootstrap_95ci_high": hi,
                "valid_bootstrap_replicates": int(len(valid)),
                "bootstrap_replicates_requested": n_boot,
                "full_data_pairwise_inversions": inv,
                "full_data_pairwise_concordant": conc,
                "full_data_pairwise_ties": ties,
                "full_data_pairwise_inversion_fraction_non_tied":
                    inv / max(inv + conc, 1),
            }
        )

        for omit_i, omit_seq in enumerate(seqs):
            mask = np.arange(len(seqs)) != omit_i
            l = local_vals[mask]
            gg = global_vals[mask]
            li, lc, lt = _pairwise_inversions(l, gg)
            loo_rows.append(
                {
                    "local_horizon_s": int(horizon),
                    "omitted_sequence": omit_seq,
                    "n_sequences_remaining": int(mask.sum()),
                    "kendall_tau_local_vs_global": _kendall(l, gg),
                    "inversions": li,
                    "concordant": lc,
                    "ties": lt,
                    "inversion_fraction_non_tied": li / max(li + lc, 1),
                }
            )

    tau_df = pd.DataFrame(tau_rows)
    loo_df = pd.DataFrame(loo_rows)
    tau_df.to_csv(out / "1B_sequence_bootstrap_rank_uncertainty.csv", index=False)
    loo_df.to_csv(out / "1B_leave_one_sequence_out_inversions.csv", index=False)

    loo_summary = (
        loo_df.groupby("local_horizon_s", as_index=False)
        .agg(
            min_loo_kendall_tau=("kendall_tau_local_vs_global", "min"),
            max_loo_kendall_tau=("kendall_tau_local_vs_global", "max"),
            min_loo_inversion_fraction=("inversion_fraction_non_tied", "min"),
            max_loo_inversion_fraction=("inversion_fraction_non_tied", "max"),
        )
    )
    loo_summary.to_csv(out / "1B_leave_one_sequence_out_summary.csv", index=False)

    # Parking threshold-surface robustness is not naively bootstrapped because
    # windows within a trajectory are temporally dependent.
    parking = _read_csv(e1["parking_dominance"])
    parking_note = [
        "# Parking00 / parking02 uncertainty boundary",
        "",
        "The parking00/parking02 result is a two-sequence threshold-surface comparison.",
        "The analysis intentionally does NOT apply an IID bootstrap to frames or overlapping",
        "trajectory windows, because that would treat temporally dependent observations as",
        "independent and can produce artificially narrow confidence intervals.",
        "",
        "The publication-strength robustness statement remains the predeclared full-grid result:",
    ]
    for _, r in parking.iterrows():
        label = "global" if r["family"] == "global_synchronization" else f"local {int(r['horizon_s'])} s"
        parking_note.append(
            f"- {label}: parking02 wins {int(r['parking02_wins'])}/{int(r['n_grid_points'])}; "
            f"parking00 wins {int(r['parking00_wins'])}/{int(r['n_grid_points'])}."
        )
    parking_note += [
        "",
        "Dataset-wide uncertainty is instead evaluated at the physical-sequence level above.",
    ]
    _write_md(out / "1B_parking_uncertainty_boundary.md", parking_note)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        x = np.arange(len(tau_df))
        y = tau_df["original_kendall_tau_local_vs_global"].to_numpy(float)
        lo = tau_df["bootstrap_95ci_low"].to_numpy(float)
        hi = tau_df["bootstrap_95ci_high"].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])
        yerr = np.maximum(yerr, 0)
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=5)
        ax.set_xticks(x, [f"{int(h)} s local vs global" for h in tau_df["local_horizon_s"]])
        ax.set_ylabel("Kendall tau")
        ax.set_ylim(-1.0, 1.0)
        ax.axhline(1.0, linewidth=1)
        ax.set_title("Sequence-bootstrap uncertainty in local/global ranking agreement")
        fig.tight_layout()
        fig.savefig(out / "1B_sequence_bootstrap_rank_uncertainty.png", dpi=200)
        plt.close(fig)

    return {
        "status": "COMPLETE",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": n_boot,
        "n_sequence_units": len(seqs),
        "rank_uncertainty": tau_df.to_dict("records"),
        "leave_one_out_summary": loo_summary.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# 1C: cross-domain normalization sensitivity
# ---------------------------------------------------------------------------

def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("cross_domain_contract_generalization_stage1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scale_factory(mode: str) -> Callable:
    def scale(values, cfg):
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")

        med_abs = float(np.median(np.abs(v)))
        floor = max(1e-9, float(cfg.robust_scale_relative_floor) * max(med_abs, 1e-6))

        if mode == "p95_p05":
            lo, hi = np.quantile(v, [0.05, 0.95])
            s = float(hi - lo)
        elif mode == "p90_p10":
            lo, hi = np.quantile(v, [0.10, 0.90])
            s = float(hi - lo)
        elif mode == "mad":
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            s = 1.4826 * mad
        else:
            raise ValueError(mode)
        return max(s, floor)
    return scale


def _locate_e3_module(repo: Path) -> Path:
    preferred = repo / "DigitalTwin" / "analysis" / "cross_domain_contract_generalization.py"
    if preferred.exists():
        return preferred
    hits = list(repo.rglob("cross_domain_contract_generalization.py"))
    if not hits:
        raise FileNotFoundError(
            "Missing DigitalTwin/analysis/cross_domain_contract_generalization.py. "
            "Re-extract the frozen E3-v4 package first."
        )
    return _latest(hits)


def _run_e3_once(mod, repo: Path, mode: str, allow_download: bool):
    cfg = mod.ProtocolConfig()
    original = mod._robust_scale
    mod._robust_scale = _scale_factory(mode)
    units_parts = []
    errors = []
    try:
        try:
            u, _, _ = mod.load_magnet_contract_units(repo, cfg, allow_download)
            units_parts.append(u)
        except Exception as exc:
            errors.append({"dataset": "MAGNET", "error": repr(exc)})
        try:
            u, _, _, _ = mod.load_freetwinev_contract_units(repo, cfg, allow_download)
            units_parts.append(u)
        except Exception as exc:
            errors.append({"dataset": "FreeTwinEV_1S4P", "error": repr(exc)})
        try:
            u, _, _, _ = mod.load_sng_contract_units(repo, cfg, allow_download)
            units_parts.append(u)
        except Exception as exc:
            errors.append({"dataset": "TUWien_SNG", "error": repr(exc)})
    finally:
        mod._robust_scale = original

    units = pd.concat(units_parts, ignore_index=True) if units_parts else pd.DataFrame()
    if units.empty:
        return (
            units,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(errors),
            pd.DataFrame(),
        )

    grid = mod.build_contract_grid(units, cfg)
    macro = mod.build_dataset_macro(grid)
    gridavg = mod.build_grid_average(macro)
    gates = mod.transfer_gates(units, grid, cfg)
    return units, grid, macro, gridavg, pd.DataFrame(errors), gates


def run_1c(repo: Path, out: Path, allow_download: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    module_path = _locate_e3_module(repo)
    mod = _load_module(module_path)

    modes = ["p95_p05", "p90_p10", "mad"]
    all_units = []
    all_grid = []
    all_macro = []
    all_avg = []
    all_gates = []
    all_errors = []

    for mode in modes:
        print(f"    [1C] normalization={mode}")
        try:
            result = _run_e3_once(mod, repo, mode, allow_download)
            if len(result) == 6:
                units, grid, macro, avg, errors, gates = result
            else:
                raise RuntimeError("Unexpected E3 sensitivity return shape")
        except Exception as exc:
            errors = pd.DataFrame([{"dataset": "ALL", "error": repr(exc)}])
            units = grid = macro = avg = gates = pd.DataFrame()

        if not errors.empty:
            ee = errors.copy()
            ee.insert(0, "normalization_mode", mode)
            all_errors.append(ee)

        for d in (units, grid, macro, avg, gates):
            if not d.empty:
                d.insert(0, "normalization_mode", mode)

        all_units.append(units)
        all_grid.append(grid)
        all_macro.append(macro)
        all_avg.append(avg)
        all_gates.append(gates)

    units_df = pd.concat([x for x in all_units if not x.empty], ignore_index=True) if any(not x.empty for x in all_units) else pd.DataFrame()
    grid_df = pd.concat([x for x in all_grid if not x.empty], ignore_index=True) if any(not x.empty for x in all_grid) else pd.DataFrame()
    macro_df = pd.concat([x for x in all_macro if not x.empty], ignore_index=True) if any(not x.empty for x in all_macro) else pd.DataFrame()
    avg_df = pd.concat([x for x in all_avg if not x.empty], ignore_index=True) if any(not x.empty for x in all_avg) else pd.DataFrame()
    gates_df = pd.concat([x for x in all_gates if not x.empty], ignore_index=True) if any(not x.empty for x in all_gates) else pd.DataFrame()
    errors_df = pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame(columns=["normalization_mode", "dataset", "error"])

    units_df.to_csv(out / "1C_normalization_sensitivity_contract_units.csv", index=False)
    grid_df.to_csv(out / "1C_normalization_sensitivity_contract_grid.csv", index=False)
    macro_df.to_csv(out / "1C_normalization_sensitivity_macro.csv", index=False)
    avg_df.to_csv(out / "1C_normalization_sensitivity_horizon_summary.csv", index=False)
    gates_df.to_csv(out / "1C_normalization_sensitivity_transfer_gates.csv", index=False)
    errors_df.to_csv(out / "1C_normalization_sensitivity_errors.csv", index=False)

    if avg_df.empty or gates_df.empty:
        raise RuntimeError("E3 normalization sensitivity produced no complete outputs; see errors CSV.")

    # Primary-mode reproduction check against the already-frozen E3 result.
    frozen_path = repo / "results" / "cross_domain_contract_generalization" / "cross_domain_horizon_grid_average.csv"
    reproduction = pd.DataFrame()
    max_diff = float("nan")
    if frozen_path.exists():
        frozen = _read_csv(frozen_path)
        primary = avg_df[avg_df["normalization_mode"] == "p95_p05"].copy()
        m = frozen.merge(primary, on=["dataset", "domain", "horizon_s"], suffixes=("_frozen", "_stage1"))
        if len(m):
            m["abs_difference"] = (
                pd.to_numeric(m["grid_average_validity_frozen"], errors="coerce")
                - pd.to_numeric(m["grid_average_validity_stage1"], errors="coerce")
            ).abs()
            reproduction = m
            max_diff = float(m["abs_difference"].max())
            reproduction.to_csv(out / "1C_primary_reproduction_check.csv", index=False)

    # Robustness summary.
    gate_pivot = gates_df.pivot_table(
        index="dataset",
        columns="normalization_mode",
        values="structural_transfer_pass",
        aggfunc="first",
    ).reset_index()
    for c in [x for x in gate_pivot.columns if x != "dataset"]:
        gate_pivot[c] = gate_pivot[c].map(_bool)
    mode_cols = [c for c in gate_pivot.columns if c != "dataset"]
    gate_pivot["passes_all_normalizations"] = gate_pivot[mode_cols].all(axis=1)
    gate_pivot.to_csv(out / "1C_normalization_robustness_summary.csv", index=False)

    horizon_rows = []
    for (mode, dataset), g in avg_df.groupby(["normalization_mode", "dataset"]):
        gg = g.sort_values("horizon_s")
        vals = gg["grid_average_validity"].to_numpy(float)
        hs = gg["horizon_s"].to_numpy(float)
        nonincreasing = bool(np.all(np.diff(vals) <= 1e-12)) if len(vals) > 1 else False
        horizon_rows.append(
            {
                "normalization_mode": mode,
                "dataset": dataset,
                "horizons_s": ";".join(str(int(x)) for x in hs),
                "grid_average_contract_satisfaction": ";".join(f"{x:.6f}" for x in vals),
                "nonincreasing_with_horizon": nonincreasing,
            }
        )
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(out / "1C_horizon_ordering_robustness.csv", index=False)

    # Correlation of the complete 3-dataset x 3-horizon profile with primary.
    corr_rows = []
    base = avg_df[avg_df["normalization_mode"] == "p95_p05"][
        ["dataset", "horizon_s", "grid_average_validity"]
    ].rename(columns={"grid_average_validity": "primary"})
    for mode in ["p90_p10", "mad"]:
        alt = avg_df[avg_df["normalization_mode"] == mode][
            ["dataset", "horizon_s", "grid_average_validity"]
        ].rename(columns={"grid_average_validity": "alternative"})
        m = base.merge(alt, on=["dataset", "horizon_s"]).dropna()
        rho = float(spearmanr(m["primary"], m["alternative"]).statistic) if len(m) >= 3 else np.nan
        corr_rows.append(
            {
                "alternative_normalization": mode,
                "n_dataset_horizon_points": len(m),
                "spearman_rho_vs_primary_profile": rho,
                "max_absolute_contract_satisfaction_change":
                    float(np.max(np.abs(m["alternative"] - m["primary"]))) if len(m) else np.nan,
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out / "1C_profile_rank_robustness.csv", index=False)

    if plt is not None:
        for dataset in sorted(avg_df["dataset"].unique()):
            fig, ax = plt.subplots(figsize=(6.8, 4.3))
            for mode in modes:
                g = avg_df[
                    (avg_df["dataset"] == dataset)
                    & (avg_df["normalization_mode"] == mode)
                ].sort_values("horizon_s")
                if not g.empty:
                    ax.plot(g["horizon_s"], g["grid_average_validity"], marker="o", label=mode)
            ax.set_xlabel("Service horizon (s)")
            ax.set_ylabel("Grid-average contract satisfaction")
            ax.set_title(f"Normalization sensitivity: {dataset}")
            ax.legend()
            fig.tight_layout()
            safe = re.sub(r"[^A-Za-z0-9]+", "_", dataset).strip("_").lower()
            fig.savefig(out / f"1C_{safe}_normalization_sensitivity.png", dpi=200)
            plt.close(fig)

    all_pass = (
        set(gate_pivot["dataset"]) >= EXPECTED_E3_DATASETS
        and bool(gate_pivot[gate_pivot["dataset"].isin(EXPECTED_E3_DATASETS)]["passes_all_normalizations"].all())
    )
    return {
        "status": "COMPLETE" if errors_df.empty else "COMPLETE_WITH_DATASET_ERRORS",
        "source_module": _rel(module_path, repo),
        "primary_reproduction_max_abs_difference": max_diff,
        "primary_reproduction_exact_within_1e_9":
            bool(np.isfinite(max_diff) and max_diff <= 1e-9) if frozen_path.exists() else None,
        "all_three_datasets_pass_all_three_normalizations": all_pass,
        "normalization_profile_correlations": corr_df.to_dict("records"),
        "dataset_errors": errors_df.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# 1D: timing sensitivity audit
# ---------------------------------------------------------------------------

TIMING_METRICS = ["ate_m", "heading_mae_deg", "rpe1_m", "rpe5_m", "rpe10_m"]


def _score_timing_csv(path: Path) -> tuple[int, pd.DataFrame | None]:
    try:
        d = _read_csv(path)
    except Exception:
        return (-1, None)
    clean = {_clean(c): c for c in d.columns}
    score = 0
    metric_hits = sum(1 for m in TIMING_METRICS if m in clean or f"{m}_change_pct" in clean or f"delta_{m}_pct" in clean)
    score += metric_hits * 3
    name = path.name.lower()
    if "timing" in name or "jitter" in name or "delay" in name:
        score += 4
    if any(k in clean for k in ["delay_ms", "jitter_ms", "perturbation_ms", "timing_perturbation_ms"]):
        score += 6
    if any("perturb" in k or "condition" in k or "mode" == k for k in clean):
        score += 2
    return score, d


def _standardize_timing_file(path: Path, d: pd.DataFrame) -> pd.DataFrame:
    cols = {_clean(c): c for c in d.columns}
    name = path.name.lower()

    # Determine perturbation level.
    level_col = None
    for k in ["timing_perturbation_ms", "perturbation_ms", "delay_ms", "jitter_ms", "ms"]:
        if k in cols:
            level_col = cols[k]
            break

    typ_default = "jitter" if "jitter" in name else ("delay" if "delay" in name else "")
    type_col = None
    for k in ["perturbation_type", "condition", "mode", "type"]:
        if k in cols:
            type_col = cols[k]
            break

    rows = []
    for _, r in d.iterrows():
        ptype = typ_default
        if type_col is not None:
            txt = str(r[type_col]).lower()
            if "jitter" in txt:
                ptype = "jitter"
            elif "delay" in txt:
                ptype = "delay"

        level = np.nan
        if level_col is not None:
            level = pd.to_numeric(pd.Series([r[level_col]]), errors="coerce").iloc[0]
        elif type_col is not None:
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*ms", str(r[type_col]), re.I)
            if m:
                level = float(m.group(1))

        if not ptype:
            # Separate columns can identify the type.
            if "jitter_ms" in cols and pd.notna(r[cols["jitter_ms"]]):
                ptype = "jitter"
                level = pd.to_numeric(pd.Series([r[cols["jitter_ms"]]]), errors="coerce").iloc[0]
            elif "delay_ms" in cols and pd.notna(r[cols["delay_ms"]]):
                ptype = "delay"
                level = pd.to_numeric(pd.Series([r[cols["delay_ms"]]]), errors="coerce").iloc[0]

        if not ptype or not np.isfinite(level):
            continue

        row = {
            "source_file": str(path),
            "perturbation_type": ptype,
            "perturbation_ms": float(level),
        }
        for metric in TIMING_METRICS:
            abs_col = cols.get(metric)
            pct_candidates = [
                f"{metric}_change_pct",
                f"delta_{metric}_pct",
                f"{metric}_pct_change",
                f"change_{metric}_pct",
            ]
            pct_col = next((cols[k] for k in pct_candidates if k in cols), None)
            row[metric] = pd.to_numeric(pd.Series([r[abs_col]]), errors="coerce").iloc[0] if abs_col else np.nan
            row[f"{metric}_change_pct"] = (
                pd.to_numeric(pd.Series([r[pct_col]]), errors="coerce").iloc[0]
                if pct_col else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _derive_timing_pct(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return d
    out = []
    for ptype, g in d.groupby("perturbation_type"):
        g = g.sort_values("perturbation_ms").copy()
        base_rows = g[np.isclose(g["perturbation_ms"], 0.0)]
        base = base_rows.iloc[0] if len(base_rows) else None
        for _, r in g.iterrows():
            rr = r.to_dict()
            for metric in TIMING_METRICS:
                pct = rr.get(f"{metric}_change_pct", np.nan)
                if not np.isfinite(pd.to_numeric(pd.Series([pct]), errors="coerce").iloc[0]):
                    val = rr.get(metric, np.nan)
                    if base is not None:
                        b = pd.to_numeric(pd.Series([base.get(metric, np.nan)]), errors="coerce").iloc[0]
                        v = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
                        if np.isfinite(b) and abs(b) > 1e-15 and np.isfinite(v):
                            rr[f"{metric}_change_pct"] = 100.0 * (v - b) / b
            out.append(rr)
    return pd.DataFrame(out)


def _scan_timing_sources(repo: Path) -> pd.DataFrame:
    patterns = re.compile(r"jitter|delay|timing[_ -]?sensitivity|perturb", re.I)
    roots = [
        repo / "DigitalTwin",
        repo / "scripts",
        repo,
    ]
    seen = set()
    rows = []
    allowed = {".py", ".json", ".md", ".txt", ".ps1", ".cmd", ".tex"}
    for root in roots:
        if not root.exists():
            continue
        iterator = root.rglob("*") if root != repo else root.glob("*")
        for p in iterator:
            if not p.is_file() or p.suffix.lower() not in allowed:
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            if any(part.lower() in {".git", ".venv", "venv", "node_modules"} for part in p.parts):
                continue
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if not patterns.search(text) and not patterns.search(p.name):
                continue
            lines = text.splitlines()
            match_indices = [i for i, line in enumerate(lines) if patterns.search(line)]
            context_indices = set()
            for mi in match_indices:
                for ci in range(max(0, mi - 3), min(len(lines), mi + 4)):
                    context_indices.add(ci)
            file_rows = 0
            for ci in sorted(context_indices):
                rows.append(
                    {
                        "file": _rel(p, repo),
                        "line": ci + 1,
                        "text": lines[ci].strip()[:700],
                    }
                )
                file_rows += 1
                if file_rows >= 80:
                    break
    return pd.DataFrame(rows).drop_duplicates(["file", "line"]) if rows else pd.DataFrame(columns=["file", "line", "text"])


def _infer_timing_protocol(excerpts: pd.DataFrame) -> dict:
    text = "\n".join(excerpts["text"].astype(str).tolist()) if not excerpts.empty else ""
    low = text.lower()

    if re.search(r"(np\.random\.normal|normal\s*\()", low):
        jitter_distribution = "normal/Gaussian (source code contains normal sampling)"
    elif re.search(r"(np\.random\.uniform|uniform\s*\()", low):
        jitter_distribution = "uniform (source code contains uniform sampling)"
    elif "zero-mean jitter" in low or "zero mean jitter" in low:
        jitter_distribution = "zero-mean stated; distribution shape not resolved"
    else:
        jitter_distribution = "UNRESOLVED"

    seed_matches = re.findall(r"(?:seed|default_rng)\s*(?:=|\()\s*(\d+)", low)
    seed = seed_matches[0] if seed_matches else "UNRESOLVED"

    if "physical" in low and "timestamp" in low and ("perturb" in low or "jitter" in low):
        perturbed_stream = "physical/virtual timing or timestamp path mentioned in source"
    else:
        perturbed_stream = "UNRESOLVED"

    if "trajectory values unchanged" in low or "values unchanged" in low:
        value_boundary = "trajectory/state values explicitly held unchanged"
    else:
        value_boundary = "UNRESOLVED"

    return {
        "jitter_distribution": jitter_distribution,
        "random_seed": seed,
        "perturbed_stream": perturbed_stream,
        "trajectory_value_boundary": value_boundary,
    }


def run_1d(repo: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    timing_dir = repo / "results" / "i2nav_timing_sensitivity"

    inventory_rows = []
    candidates = []
    if timing_dir.exists():
        for p in timing_dir.rglob("*"):
            if p.is_file():
                inventory_rows.append(
                    {
                        "file": _rel(p, repo),
                        "suffix": p.suffix.lower(),
                        "bytes": p.stat().st_size,
                    }
                )
                if p.suffix.lower() == ".csv":
                    score, d = _score_timing_csv(p)
                    if score >= 6 and d is not None:
                        candidates.append((score, p, d))
    pd.DataFrame(inventory_rows).to_csv(out / "1D_timing_input_inventory.csv", index=False)

    standardized_parts = []
    for score, p, d in sorted(candidates, reverse=True, key=lambda x: x[0]):
        x = _standardize_timing_file(p, d)
        if not x.empty:
            standardized_parts.append(x)

    timing = pd.concat(standardized_parts, ignore_index=True) if standardized_parts else pd.DataFrame()
    if not timing.empty:
        timing = _derive_timing_pct(timing)
        # Keep the most information-rich row for duplicate type/level combinations.
        timing["_nonnull"] = timing.notna().sum(axis=1)
        timing = (
            timing.sort_values("_nonnull", ascending=False)
            .drop_duplicates(["perturbation_type", "perturbation_ms"], keep="first")
            .drop(columns="_nonnull")
            .sort_values(["perturbation_type", "perturbation_ms"])
        )
    timing.to_csv(out / "1D_timing_publication_table.csv", index=False)

    excerpts = _scan_timing_sources(repo)
    excerpts.to_csv(out / "1D_timing_source_excerpts.csv", index=False)
    protocol = _infer_timing_protocol(excerpts)

    levels = {}
    if not timing.empty:
        for ptype, g in timing.groupby("perturbation_type"):
            levels[ptype] = sorted(float(x) for x in g["perturbation_ms"].dropna().unique())

    protocol_lines = [
        "# Timing sensitivity publication audit",
        "",
        "## What the audit can verify from the repository",
        "",
        f"- Timing result directory exists: **{timing_dir.exists()}** (`{_rel(timing_dir, repo)}`).",
        f"- Standardized timing rows recovered: **{len(timing)}**.",
        f"- Perturbation levels recovered: `{json.dumps(levels, sort_keys=True)}`.",
        f"- Jitter distribution: **{protocol['jitter_distribution']}**.",
        f"- Random seed: **{protocol['random_seed']}**.",
        f"- Perturbed stream: **{protocol['perturbed_stream']}**.",
        f"- State/trajectory-value boundary: **{protocol['trajectory_value_boundary']}**.",
        "",
        "## Required manuscript boundary",
        "",
        "This experiment is a controlled synchronization/timestamp replay study, not a packet-level",
        "wireless-network experiment, unless the underlying source code explicitly implements a packet",
        "channel. Do not describe an unresolved jitter distribution or stream direction from memory;",
        "fill it from the source/configuration identified in `1D_timing_source_excerpts.csv`.",
        "",
    ]

    unresolved = [
        k for k, v in protocol.items()
        if str(v).startswith("UNRESOLVED") or "not resolved" in str(v)
    ]
    if unresolved:
        protocol_lines += [
            "## Publication blocker(s) still unresolved",
            "",
        ]
        for k in unresolved:
            protocol_lines.append(f"- `{k}`")
        protocol_lines += [
            "",
            "These are protocol-documentation gaps, not failures of the numerical timing result.",
        ]
    else:
        protocol_lines += [
            "## Protocol documentation gate",
            "",
            "**PASS:** the audit recovered the requested timing provenance fields.",
        ]

    _write_md(out / "1D_timing_protocol_audit.md", protocol_lines)

    if plt is not None and not timing.empty:
        for ptype in sorted(timing["perturbation_type"].unique()):
            g = timing[timing["perturbation_type"] == ptype].sort_values("perturbation_ms")
            fig, ax = plt.subplots(figsize=(7.8, 4.6))
            for metric in TIMING_METRICS:
                col = f"{metric}_change_pct"
                if col in g.columns and pd.to_numeric(g[col], errors="coerce").notna().any():
                    ax.plot(
                        g["perturbation_ms"],
                        pd.to_numeric(g[col], errors="coerce"),
                        marker="o",
                        label=metric,
                    )
            ax.axhline(0, linewidth=1)
            ax.set_xlabel("Timing perturbation (ms)")
            ax.set_ylabel("Change from synchronized replay (%)")
            ax.set_title(f"Publication timing sensitivity: {ptype}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"1D_{ptype}_sensitivity_publication.png", dpi=200)
            plt.close(fig)

    timing_status = (
        "COMPLETE"
        if timing_dir.exists() and not timing.empty and not unresolved
        else "NEEDS_PROTOCOL_CLARIFICATION"
    )
    return {
        "status": timing_status,
        "timing_dir_exists": timing_dir.exists(),
        "standardized_rows": len(timing),
        "perturbation_levels": levels,
        "protocol": protocol,
        "unresolved_protocol_fields": unresolved,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> None:
    # Scale definitions.
    class Cfg:
        robust_scale_relative_floor = 0.01
    v = np.arange(101, dtype=float)
    assert _scale_factory("p95_p05")(v, Cfg()) > _scale_factory("p90_p10")(v, Cfg())
    assert _scale_factory("mad")(v, Cfg()) > 0

    # Pair inversion.
    inv, conc, ties = _pairwise_inversions(
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 1.0]),
    )
    assert inv == 3 and conc == 0

    # Timing standardization.
    with tempfile_dir() as td:
        p = td / "jitter_summary.csv"
        pd.DataFrame(
            {
                "jitter_ms": [0, 50],
                "ate_m": [1.0, 1.01],
                "rpe1_m": [0.1, 0.1662],
            }
        ).to_csv(p, index=False)
        d = _read_csv(p)
        s = _derive_timing_pct(_standardize_timing_file(p, d))
        row = s[np.isclose(s["perturbation_ms"], 50)].iloc[0]
        assert abs(float(row["rpe1_m_change_pct"]) - 66.2) < 1e-6


class tempfile_dir:
    def __enter__(self):
        import tempfile
        self._obj = tempfile.TemporaryDirectory()
        return Path(self._obj.name)
    def __exit__(self, exc_type, exc, tb):
        self._obj.cleanup()


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--skip-e3", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        print("stage1_publication_hardening self-test: PASS")
        return 0

    repo = args.repo.resolve()
    out = repo / "results" / "stage1_publication_hardening"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[repo] {repo}")
    print(f"[out ] {out}")
    print("[test] Internal self-test: PASS")

    manifest = {
        "analysis": "stage1_publication_hardening",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": args.bootstrap,
        "claim_boundary": (
            "publication hardening only; no model retraining, no post-hoc threshold selection, "
            "no modification of frozen E1/E2/E3 primary outputs"
        ),
        "experiments": {},
    }

    failures = []

    try:
        e1 = discover_e1(repo)
        print("[1A] Scalar vs standard metrics vs service-contract decision layer")
        manifest["experiments"]["1A"] = run_1a(repo, out, e1)
    except Exception as exc:
        failures.append(("1A", exc, traceback.format_exc()))
        manifest["experiments"]["1A"] = {"status": "FAILED", "error": repr(exc)}

    try:
        e1 = discover_e1(repo)
        print("[1B] Sequence-level bootstrap + leave-one-sequence-out robustness")
        manifest["experiments"]["1B"] = run_1b(repo, out, e1, args.bootstrap)
    except Exception as exc:
        failures.append(("1B", exc, traceback.format_exc()))
        manifest["experiments"]["1B"] = {"status": "FAILED", "error": repr(exc)}

    if args.skip_e3:
        manifest["experiments"]["1C"] = {"status": "SKIPPED_BY_USER"}
    else:
        try:
            print("[1C] E3 normalization sensitivity: p95-p05 vs p90-p10 vs MAD")
            manifest["experiments"]["1C"] = run_1c(repo, out, args.allow_download)
        except Exception as exc:
            failures.append(("1C", exc, traceback.format_exc()))
            manifest["experiments"]["1C"] = {"status": "FAILED", "error": repr(exc)}

    try:
        print("[1D] Timing result + protocol provenance audit")
        manifest["experiments"]["1D"] = run_1d(repo, out)
    except Exception as exc:
        failures.append(("1D", exc, traceback.format_exc()))
        manifest["experiments"]["1D"] = {"status": "FAILED", "error": repr(exc)}

    manifest["hard_failures"] = [name for name, _, _ in failures]
    statuses = {k: v.get("status") for k, v in manifest["experiments"].items()}
    manifest["statuses"] = statuses

    if failures:
        with (out / "STAGE1_FAILURES.txt").open("w", encoding="utf-8") as f:
            for name, exc, tb in failures:
                f.write(f"===== {name}: {repr(exc)} =====\n{tb}\n\n")

    (out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# Stage 1 publication-hardening report",
        "",
        "## Frozen scope",
        "",
        "This stage addresses reviewer-facing methodological gaps without retraining the twin or",
        "changing the frozen E1/E2/E3 primary analyses.",
        "",
        "### 1A - Scalar vs standard metrics vs service contracts",
        "",
        f"Status: **{statuses.get('1A')}**.",
    ]
    a = manifest["experiments"].get("1A", {})
    if a.get("status") == "COMPLETE":
        ev = a.get("evidence", {})
        lines += [
            f"- Local service-matched metrics improve median rank alignment over ATE-only by "
            f"{ev.get('minimum_local_alignment_gain_matched_over_ate', float('nan')):.3f} to "
            f"{ev.get('maximum_local_alignment_gain_matched_over_ate', float('nan')):.3f}.",
            "- Interpretation: conventional metrics remain useful; the contract adds explicit service semantics and a decision layer.",
        ]

    lines += [
        "",
        "### 1B - Sequence-level uncertainty",
        "",
        f"Status: **{statuses.get('1B')}**.",
        "- Resampling unit: physical i2Nav sequence, not frame/window.",
        "- Parking00/parking02 is kept as threshold-surface robustness rather than an IID window bootstrap.",
        "",
        "### 1C - Cross-domain normalization sensitivity",
        "",
        f"Status: **{statuses.get('1C')}**.",
    ]
    c = manifest["experiments"].get("1C", {})
    if c.get("status", "").startswith("COMPLETE"):
        lines += [
            f"- Frozen p95-p05 reproduction max absolute difference: `{c.get('primary_reproduction_max_abs_difference')}`.",
            f"- All MAGNET/FreeTwinEV/SNG structural gates pass under p95-p05, p90-p10, and MAD: "
            f"**{c.get('all_three_datasets_pass_all_three_normalizations')}**.",
        ]

    lines += [
        "",
        "### 1D - Timing protocol hardening",
        "",
        f"Status: **{statuses.get('1D')}**.",
    ]
    d = manifest["experiments"].get("1D", {})
    if d:
        lines += [
            f"- Standardized timing rows: `{d.get('standardized_rows', 0)}`.",
            f"- Recovered perturbation levels: `{json.dumps(d.get('perturbation_levels', {}), sort_keys=True)}`.",
            f"- Unresolved provenance fields: `{d.get('unresolved_protocol_fields', [])}`.",
        ]

    lines += [
        "",
        "## Stage 1 decision",
        "",
    ]
    hard_complete = all(
        statuses.get(k) == "COMPLETE" or (k == "1C" and str(statuses.get(k, "")).startswith("COMPLETE"))
        for k in ["1A", "1B", "1C"]
        if not (k == "1C" and args.skip_e3)
    )
    timing_complete = statuses.get("1D") == "COMPLETE"

    if failures:
        verdict = "INCOMPLETE - one or more computational blocks failed"
    elif hard_complete and timing_complete:
        verdict = "READY FOR STAGE 2 MANUSCRIPT REVISION"
    elif hard_complete:
        verdict = "COMPUTATIONAL RESULTS COMPLETE; TIMING PROTOCOL DOCUMENTATION STILL NEEDS RESOLUTION"
    else:
        verdict = "INCOMPLETE - inspect the per-block outputs"
    lines.append(f"**{verdict}**")
    lines += [
        "",
        "Do not change thresholds or normalization choices in response to these sensitivity results.",
        "If a sensitivity conclusion is weaker than expected, report that honestly in the manuscript.",
    ]
    _write_md(out / "STAGE1_PUBLICATION_HARDENING_REPORT.md", lines)

    print()
    print("=" * 72)
    print("Stage 1 run finished")
    print(f"Results: {out}")
    for k, v in statuses.items():
        print(f"  {k}: {v}")
    print(f"Verdict: {verdict}")
    if failures:
        print("See STAGE1_FAILURES.txt for tracebacks.")
    print("=" * 72)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

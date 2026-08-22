from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _finite(a: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(a), dtype=float)
    return x[np.isfinite(x)]


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _bootstrap_stat(
    x: np.ndarray,
    stat_fn,
    rng: np.random.Generator,
    n_boot: int,
) -> Tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat_fn(x))
    if len(x) == 1:
        return point, point, point
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = x[rng.integers(0, len(x), size=len(x))]
        vals[i] = stat_fn(sample)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return point, float(lo), float(hi)


def _bootstrap_paired(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> Dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return {k: float("nan") for k in ["n", "median_a", "median_b", "median_diff", "diff_ci_low", "diff_ci_high", "median_ratio", "ratio_ci_low", "ratio_ci_high"]}
    diff = b - a
    ratio = b / np.maximum(a, 1e-12)
    boot_diff = np.empty(n_boot)
    boot_ratio = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        boot_diff[i] = np.median(diff[idx])
        boot_ratio[i] = np.median(ratio[idx])
    dlo, dhi = np.quantile(boot_diff, [0.025, 0.975])
    rlo, rhi = np.quantile(boot_ratio, [0.025, 0.975])
    return {
        "n": int(len(a)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "median_diff": float(np.median(diff)),
        "diff_ci_low": float(dlo),
        "diff_ci_high": float(dhi),
        "median_ratio": float(np.median(ratio)),
        "ratio_ci_low": float(rlo),
        "ratio_ci_high": float(rhi),
    }


def _rank_biserial_from_paired(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if len(diff) == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(diff))
    w_pos = float(ranks[diff > 0].sum())
    w_neg = float(ranks[diff < 0].sum())
    denom = w_pos + w_neg
    return (w_pos - w_neg) / denom if denom else float("nan")


def paired_horizon_test(nonoverlap: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    a = pd.to_numeric(nonoverlap["band_0_60_rmse_c"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(nonoverlap["band_301_599_rmse_c"], errors="coerce").to_numpy(float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    base = _bootstrap_paired(a, b, rng, n_boot)
    diff = b - a
    try:
        w = stats.wilcoxon(b, a, alternative="greater", zero_method="wilcox", method="auto")
        stat, p = float(w.statistic), float(w.pvalue)
    except Exception:
        stat, p = float("nan"), float("nan")
    base.update(
        {
            "comparison": "long_301_599s_vs_short_0_60s_RMSE",
            "wilcoxon_statistic": stat,
            "wilcoxon_one_sided_p": p,
            "paired_rank_biserial": _rank_biserial_from_paired(diff),
            "fraction_long_gt_short": float(np.mean(b > a)) if len(a) else float("nan"),
        }
    )
    return pd.DataFrame([base])


def temporally_disjoint(a: pd.Series, b: pd.Series) -> bool:
    return float(a["forecast_end_s"]) < float(b["forecast_start_s"]) or float(b["forecast_end_s"]) < float(a["forecast_start_s"])


def disjoint_diagnostic_pairs(strict: pd.DataFrame, rmse_tol: float) -> pd.DataFrame:
    d = strict.sort_values("forecast_start_s").reset_index(drop=True)
    rows: List[Dict[str, float]] = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            a, b = d.iloc[i], d.iloc[j]
            if not temporally_disjoint(a, b):
                continue
            ra, rb = float(a.rmse_c), float(b.rmse_c)
            mean_rmse = max((ra + rb) / 2.0, 1e-12)
            rel = abs(ra - rb) / mean_rmse
            if rel > rmse_tol:
                continue
            p99a, p99b = float(a.p99_abs_c), float(b.p99_abs_c)
            pa, pb = float(a.persistence_envelope_frac), float(b.persistence_envelope_frac)
            la, lb = float(a.band_301_599_rmse_c), float(b.band_301_599_rmse_c)
            p99_ratio = max(p99a, p99b) / max(min(p99a, p99b), 1e-12)
            long_ratio = max(la, lb) / max(min(la, lb), 1e-12)
            pers_diff = abs(pa - pb)
            separation = max(p99_ratio / 1.5, long_ratio / 1.5, pers_diff / 0.15)
            qualifies = p99_ratio >= 1.5 or long_ratio >= 1.5 or pers_diff >= 0.15
            rows.append(
                {
                    "window_a": int(a.window_id),
                    "window_b": int(b.window_id),
                    "rmse_tolerance_fraction": rmse_tol,
                    "rmse_a_c": ra,
                    "rmse_b_c": rb,
                    "rmse_relative_difference": rel,
                    "p99_a_c": p99a,
                    "p99_b_c": p99b,
                    "p99_ratio": p99_ratio,
                    "persistence_a": pa,
                    "persistence_b": pb,
                    "persistence_difference": pers_diff,
                    "long_horizon_rmse_a_c": la,
                    "long_horizon_rmse_b_c": lb,
                    "long_horizon_ratio": long_ratio,
                    "diagnostic_separation_score": separation,
                    "qualifies_strong_counterexample": bool(qualifies),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["qualifies_strong_counterexample", "diagnostic_separation_score"], ascending=[False, False]).reset_index(drop=True)


def trimmed_horizon_sensitivity(strict: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trim_metric in ["rmse_c", "max_abs_c", "p99_abs_c"]:
        values = pd.to_numeric(strict[trim_metric], errors="coerce")
        for trim_pct in [0, 1, 5, 10, 20]:
            if trim_pct == 0:
                d = strict.copy()
                cutoff = float("inf")
            else:
                cutoff = float(values.quantile(1 - trim_pct / 100.0))
                d = strict[values <= cutoff].copy()
            short = pd.to_numeric(d["band_0_60_rmse_c"], errors="coerce")
            long = pd.to_numeric(d["band_301_599_rmse_c"], errors="coerce")
            sm, lm = float(short.median()), float(long.median())
            rows.append(
                {
                    "trim_metric": trim_metric,
                    "trim_top_percent": trim_pct,
                    "cutoff": cutoff,
                    "n_windows": len(d),
                    "median_short_rmse_c": sm,
                    "median_long_rmse_c": lm,
                    "median_long_short_ratio": lm / max(sm, 1e-12),
                    "median_long_minus_short_c": float((long - short).median()),
                    "fraction_long_gt_short": float((long > short).mean()),
                }
            )
    return pd.DataFrame(rows)


def fixed_threshold_sensitivity(long_df: pd.DataFrame, strict_ids: set[int], nonoverlap_ids: set[int]) -> pd.DataFrame:
    d = long_df[long_df["window_id"].astype(int).isin(strict_ids)].copy()
    rows = []
    for threshold in [1, 2, 5, 10, 20, 50]:
        d["exceed"] = pd.to_numeric(d["abs_error_c"], errors="coerce") > threshold
        per_window = d.groupby("window_id")["exceed"].mean()
        for subset, ids in [("all_strict", strict_ids), ("nonoverlap", nonoverlap_ids)]:
            v = per_window[per_window.index.astype(int).isin(ids)].to_numpy(float)
            rows.append(
                {
                    "threshold_c": threshold,
                    "subset": subset,
                    "n_windows": len(v),
                    "median_persistence_fraction": float(np.median(v)) if len(v) else np.nan,
                    "q25_persistence_fraction": float(np.quantile(v, 0.25)) if len(v) else np.nan,
                    "q75_persistence_fraction": float(np.quantile(v, 0.75)) if len(v) else np.nan,
                    "fraction_windows_with_ge10pct_exceedance": float(np.mean(v >= 0.10)) if len(v) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def envelope_sensitivity(long_df: pd.DataFrame, strict_ids: set[int], nonoverlap_ids: set[int]) -> pd.DataFrame:
    d = long_df[long_df["window_id"].astype(int).isin(strict_ids)].copy()
    d["abs_error_c"] = pd.to_numeric(d["abs_error_c"], errors="coerce")
    early = d[d["horizon_s"] <= 30]
    stats_rows = []
    for sensor, g in early.groupby("sensor"):
        vals = _finite(g["abs_error_c"])
        stats_rows.append((sensor, float(np.median(vals)), 1.4826 * _mad(vals)))
    sensor_stats = {s: (m, rs) for s, m, rs in stats_rows}
    rows = []
    for k in [2.0, 3.0, 4.0, 5.0]:
        for floor in [0.5, 1.0, 2.0]:
            tmp = d[["window_id", "sensor", "abs_error_c"]].copy()
            tmp["threshold"] = tmp["sensor"].map(lambda s: max(sensor_stats[s][0] + k * sensor_stats[s][1], floor))
            tmp["exceed"] = tmp["abs_error_c"] > tmp["threshold"]
            per_window = tmp.groupby("window_id")["exceed"].mean()
            for subset, ids in [("all_strict", strict_ids), ("nonoverlap", nonoverlap_ids)]:
                v = per_window[per_window.index.astype(int).isin(ids)].to_numpy(float)
                rows.append(
                    {
                        "mad_multiplier": k,
                        "floor_c": floor,
                        "subset": subset,
                        "n_windows": len(v),
                        "median_persistence_fraction": float(np.median(v)) if len(v) else np.nan,
                        "q25": float(np.quantile(v, 0.25)) if len(v) else np.nan,
                        "q75": float(np.quantile(v, 0.75)) if len(v) else np.nan,
                        "fraction_windows_ge10pct": float(np.mean(v >= 0.10)) if len(v) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def condition_sensitivity(strict: pd.DataFrame, nonoverlap_ids: set[int]) -> pd.DataFrame:
    rows = []
    for threshold in [0.5, 0.75, 1.0, 1.5]:
        for subset_name, d0 in [("all_strict", strict), ("nonoverlap", strict[strict["window_id"].astype(int).isin(nonoverlap_ids)])]:
            d = d0.copy()
            slope = pd.to_numeric(d["thermal_slope_c_per_min"], errors="coerce")
            d["regime"] = np.where(slope > threshold, "heating", np.where(slope < -threshold, "cooling", "steady"))
            for regime, g in d.groupby("regime"):
                rows.append(
                    {
                        "slope_threshold_c_per_min": threshold,
                        "subset": subset_name,
                        "regime": regime,
                        "n_windows": len(g),
                        "median_rmse_c": float(pd.to_numeric(g["rmse_c"], errors="coerce").median()),
                        "median_long_rmse_c": float(pd.to_numeric(g["band_301_599_rmse_c"], errors="coerce").median()),
                        "median_p99_c": float(pd.to_numeric(g["p99_abs_c"], errors="coerce").median()),
                        "median_persistence": float(pd.to_numeric(g["persistence_envelope_frac"], errors="coerce").median()),
                    }
                )
    return pd.DataFrame(rows)


def condition_nonoverlap_test(strict: pd.DataFrame, nonoverlap_ids: set[int], threshold: float = 0.75) -> pd.DataFrame:
    d = strict[strict["window_id"].astype(int).isin(nonoverlap_ids)].copy()
    slope = pd.to_numeric(d["thermal_slope_c_per_min"], errors="coerce")
    d["regime"] = np.where(slope > threshold, "heating", np.where(slope < -threshold, "cooling", "steady"))
    groups = []
    names = []
    for name, g in d.groupby("regime"):
        vals = _finite(g["rmse_c"])
        if len(vals) >= 2:
            groups.append(vals)
            names.append(name)
    if len(groups) < 2:
        return pd.DataFrame([{"slope_threshold_c_per_min": threshold, "groups": ",".join(names), "kruskal_h": np.nan, "p_value": np.nan, "epsilon_squared": np.nan, "note": "insufficient groups with n>=2"}])
    kw = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = max((float(kw.statistic) - k + 1) / max(n - k, 1), 0.0)
    return pd.DataFrame([{"slope_threshold_c_per_min": threshold, "groups": ",".join(names), "n_total": n, "kruskal_h": float(kw.statistic), "p_value": float(kw.pvalue), "epsilon_squared": eps2, "note": "exploratory; regime labels derive from observed thermal slope"}])


def component_horizon_summary(long_df: pd.DataFrame, strict_ids: set[int]) -> pd.DataFrame:
    d = long_df[long_df["window_id"].astype(int).isin(strict_ids)].copy()
    rows = []
    for sensor, g in d.groupby("sensor"):
        short = g[g["horizon_s"] <= 60].groupby("window_id")["abs_error_c"].apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
        long = g[(g["horizon_s"] >= 301) & (g["horizon_s"] <= 599.5)].groupby("window_id")["abs_error_c"].apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
        both = pd.concat([short.rename("short"), long.rename("long")], axis=1).dropna()
        rows.append(
            {
                "sensor": sensor,
                "n_windows": len(both),
                "median_short_rmse_c": float(both.short.median()) if len(both) else np.nan,
                "median_long_rmse_c": float(both.long.median()) if len(both) else np.nan,
                "median_long_short_ratio": float((both.long / np.maximum(both.short, 1e-12)).median()) if len(both) else np.nan,
                "fraction_long_gt_short": float((both.long > both.short).mean()) if len(both) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("median_long_rmse_c", ascending=False)


def make_hardening_plots(out: Path, paired: pd.DataFrame, trim: pd.DataFrame, fixed: pd.DataFrame, disjoint5: pd.DataFrame) -> None:
    fig = out / "figures"
    fig.mkdir(exist_ok=True, parents=True)

    # Robustness to trimming extreme windows.
    d = trim[(trim["trim_metric"] == "max_abs_c")].copy()
    plt.figure(figsize=(7, 5))
    plt.plot(d["trim_top_percent"], d["median_short_rmse_c"], marker="o", label="0–60 s")
    plt.plot(d["trim_top_percent"], d["median_long_rmse_c"], marker="o", label="301–599 s")
    plt.xlabel("Top windows removed by max error (%)")
    plt.ylabel("Median window RMSE (°C)")
    plt.title("Horizon degradation remains after removing extreme windows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig / "robustness_trim_extremes.png", dpi=200)
    plt.close()

    # Persistence threshold sensitivity.
    d = fixed[fixed["subset"] == "nonoverlap"].copy()
    plt.figure(figsize=(7, 5))
    plt.plot(d["threshold_c"], d["median_persistence_fraction"], marker="o")
    plt.xlabel("Fixed discrepancy threshold (°C)")
    plt.ylabel("Median persistence fraction")
    plt.title("Persistence conclusion across transparent thresholds")
    plt.tight_layout()
    plt.savefig(fig / "persistence_threshold_sensitivity.png", dpi=200)
    plt.close()

    # Strongest temporally disjoint diagnostic pair as a compact metric comparison.
    if not disjoint5.empty:
        q = disjoint5[disjoint5["qualifies_strong_counterexample"]]
        if not q.empty:
            r = q.iloc[0]
            labels = ["RMSE", "p99", "Persistence", "Long RMSE"]
            a = [r.rmse_a_c, r.p99_a_c, r.persistence_a, r.long_horizon_rmse_a_c]
            b = [r.rmse_b_c, r.p99_b_c, r.persistence_b, r.long_horizon_rmse_b_c]
            x = np.arange(len(labels))
            width = 0.35
            plt.figure(figsize=(8, 5))
            plt.bar(x - width/2, a, width, label=f"Window {int(r.window_a)}")
            plt.bar(x + width/2, b, width, label=f"Window {int(r.window_b)}")
            plt.xticks(x, labels)
            plt.ylabel("Raw value (mixed units; persistence is fraction)")
            plt.title("Similar aggregate RMSE, different fidelity diagnostics")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig / "disjoint_similar_rmse_counterexample.png", dpi=200)
            plt.close()


def build_summary(
    out: Path,
    paired: pd.DataFrame,
    trim: pd.DataFrame,
    d5: pd.DataFrame,
    d10: pd.DataFrame,
    cond_test: pd.DataFrame,
    component: pd.DataFrame,
) -> None:
    p = paired.iloc[0]
    base_trim = trim[(trim.trim_metric == "max_abs_c") & (trim.trim_top_percent == 0)].iloc[0]
    trim10 = trim[(trim.trim_metric == "max_abs_c") & (trim.trim_top_percent == 10)].iloc[0]
    n5 = int(d5["qualifies_strong_counterexample"].sum()) if not d5.empty else 0
    n10 = int(d10["qualifies_strong_counterexample"].sum()) if not d10.empty else 0
    top_sensor = component.iloc[0] if not component.empty else None
    ct = cond_test.iloc[0]

    lines = [
        "# MAGNET publication-hardening report",
        "",
        "This report is designed for manuscript decisions. It deliberately avoids the earlier heuristic 0–5 score.",
        "",
        "## A. Independence-aware horizon result",
        f"- Non-overlapping paired windows: **{int(p['n'])}**",
        f"- Median 0–60 s RMSE: **{p['median_a']:.4f} °C**",
        f"- Median 301–599 s RMSE: **{p['median_b']:.4f} °C**",
        f"- Median paired increase: **{p['median_diff']:.4f} °C** (bootstrap 95% CI {p['diff_ci_low']:.4f} to {p['diff_ci_high']:.4f})",
        f"- Median paired long/short ratio: **{p['median_ratio']:.2f}×** (bootstrap 95% CI {p['ratio_ci_low']:.2f}× to {p['ratio_ci_high']:.2f}×)",
        f"- One-sided Wilcoxon p-value for long > short: **{p['wilcoxon_one_sided_p']:.3g}**",
        f"- Paired rank-biserial effect: **{p['paired_rank_biserial']:.3f}**",
        "",
        "## B. Outlier robustness",
        f"- With all strict windows, median long/short RMSE ratio: **{base_trim['median_long_short_ratio']:.2f}×**.",
        f"- After removing the top 10% of windows by maximum absolute error, the ratio is **{trim10['median_long_short_ratio']:.2f}×** across **{int(trim10['n_windows'])}** windows.",
        "- If the second value remains clearly >1, the horizon result is not an artifact of the most numerically unstable forecasts.",
        "",
        "## C. Strong counterexamples using temporally disjoint windows",
        f"- Strong similar-RMSE pairs within **5% RMSE**: **{n5}**.",
        f"- Strong similar-RMSE pairs within **10% RMSE**: **{n10}**.",
        "- These are stronger than the exploratory pair count because the paired forecast intervals do not overlap in time.",
        "",
        "## D. Operating-condition test on non-overlapping windows",
        f"- Kruskal–Wallis p-value: **{ct['p_value']:.3g}**; epsilon-squared: **{ct['epsilon_squared']:.3f}** (when group sizes permit).",
        "- Treat this as secondary/exploratory because the regime labels are derived from thermal slope rather than an externally randomized condition variable.",
        "",
        "## E. Component localization",
    ]
    if top_sensor is not None:
        lines += [
            f"- Largest median long-horizon component RMSE: **{top_sensor['sensor']}**, {top_sensor['median_long_rmse_c']:.3f} °C.",
            f"- Its median long/short ratio is **{top_sensor['median_long_short_ratio']:.2f}×**.",
        ]
    lines += [
        "",
        "## Publication decision",
        "MAGNET belongs in the main paper if three conditions hold after this hardened run: (1) the non-overlap paired horizon effect remains large with a narrow bootstrap interval, (2) the effect remains after trimming extreme windows, and (3) at least one temporally disjoint similar-RMSE counterexample survives. Condition effects and threshold sensitivity are supporting evidence, not gating requirements.",
        "",
        "## Defensible claim",
        "The MAGNET experiment supports **cross-domain transfer** of the fidelity decomposition to an independently developed thermal digital twin. It does not establish universal validity across all digital-twin domains.",
        "",
        "## Muñoz baseline note",
        "Do not reimplement the Muñoz trace-alignment method approximately for MAGNET. Its published method depends on application-specific maximum admissible distance (MAD), usually linked to measurement accuracy. Unless a defensible MAGNET thermocouple accuracy/MAD is established, keep the official Muñoz comparison in the primary robot experiments and use MAGNET specifically as cross-domain transfer evidence.",
    ]
    (out / "PUBLICATION_HARDENING_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reviewer-resistant robustness/statistical hardening for MAGNET TFP results.")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    out = args.results
    required = ["window_metrics.csv", "nonoverlap_windows.csv", "aligned_error_long.csv"]
    missing = [x for x in required if not (out / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing first-stage result files: {missing}. Run run_analysis.py first.")

    windows = pd.read_csv(out / "window_metrics.csv")
    windows["strict_eligible"] = _bool_series(windows["strict_eligible"])
    strict = windows[windows["strict_eligible"]].copy()
    nonoverlap = pd.read_csv(out / "nonoverlap_windows.csv")
    long_df = pd.read_csv(out / "aligned_error_long.csv")
    strict_ids = set(strict["window_id"].astype(int))
    nonoverlap_ids = set(nonoverlap["window_id"].astype(int))

    rng = np.random.default_rng(args.seed)

    paired = paired_horizon_test(nonoverlap, rng, args.bootstrap_samples)
    paired.to_csv(out / "paired_horizon_significance_nonoverlap.csv", index=False)

    d5 = disjoint_diagnostic_pairs(strict, 0.05)
    d10 = disjoint_diagnostic_pairs(strict, 0.10)
    d5.to_csv(out / "diagnostic_pairs_disjoint_5pct.csv", index=False)
    d10.to_csv(out / "diagnostic_pairs_disjoint_10pct.csv", index=False)

    trim = trimmed_horizon_sensitivity(strict)
    trim.to_csv(out / "outlier_trim_sensitivity.csv", index=False)

    fixed = fixed_threshold_sensitivity(long_df, strict_ids, nonoverlap_ids)
    fixed.to_csv(out / "persistence_fixed_threshold_sensitivity.csv", index=False)

    env = envelope_sensitivity(long_df, strict_ids, nonoverlap_ids)
    env.to_csv(out / "persistence_envelope_sensitivity.csv", index=False)

    cond = condition_sensitivity(strict, nonoverlap_ids)
    cond.to_csv(out / "condition_definition_sensitivity.csv", index=False)

    cond_test = condition_nonoverlap_test(strict, nonoverlap_ids, 0.75)
    cond_test.to_csv(out / "condition_nonoverlap_stat_test.csv", index=False)

    comp = component_horizon_summary(long_df, strict_ids)
    comp.to_csv(out / "component_horizon_summary.csv", index=False)

    make_hardening_plots(out, paired, trim, fixed, d5)
    build_summary(out, paired, trim, d5, d10, cond_test, comp)

    manifest = {
        "purpose": "MAGNET cross-domain publication hardening",
        "independence_policy": "Headline inferential horizon statistics use the precomputed greedy non-overlapping subset. Diagnostic counterexample pairs must also have disjoint forecast intervals.",
        "robustness_checks": [
            "trim top 1/5/10/20 percent of windows by RMSE, p99, or maximum absolute error",
            "fixed persistence thresholds from 1 to 50 C",
            "early-envelope MAD multipliers 2 to 5 and floors 0.5 to 2 C",
            "thermal slope regime thresholds 0.5 to 1.5 C/min",
        ],
        "statistical_tests": [
            "paired bootstrap interval for long-short RMSE difference and ratio",
            "one-sided paired Wilcoxon signed-rank test on non-overlapping windows",
            "paired rank-biserial effect size",
            "exploratory Kruskal-Wallis condition comparison on non-overlapping windows",
        ],
        "warning": "Do not treat overlapping forecast windows as independent replicates and do not report the old heuristic evidence score as a scientific metric.",
    }
    (out / "publication_hardening_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("MAGNET publication hardening complete.")
    print(f"Strict windows: {len(strict)} | non-overlap: {len(nonoverlap)}")
    print(f"Disjoint strong counterexamples: 5% RMSE={int(d5['qualifies_strong_counterexample'].sum()) if not d5.empty else 0}; 10% RMSE={int(d10['qualifies_strong_counterexample'].sum()) if not d10.empty else 0}")
    print(f"Summary: {(out / 'PUBLICATION_HARDENING_SUMMARY.md').resolve()}")


if __name__ == "__main__":
    main()

"""Causal, trace-driven geofence qualification on frozen i2Nav V2 outputs.

No model fitting or trajectory alignment occurs here. The 1-Hz physical
reference stream is a *simulated availability schedule* over offline truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/i2nav_v2_full_loso/i2nav_v2_full_loso"
OUT = ROOT / "results/geofence_decision_value"
DELAYS_MS = (0, 25, 50, 100, 200)
JITTER_MS = (0, 10, 25, 50)
JITTER_SEEDS = (0, 1, 2, 3, 4)
TOLERANCES_M = (0.5, 1.0, 2.0, 5.0, 10.0)
AOI_LIMITS_S = (0.6, 1.0, 1.5)
FENCES = (
    ("near", 4.0, 0.0, 3.0),
    ("far", 12.0, 0.0, 6.0),
    ("lateral", 5.0, 5.0, 4.0),
)
METHODS = ("always", "error", "error_fresh", "contract")
DT = 0.1


def discover() -> list[Path]:
    paths = sorted(SOURCE.glob("replicate_*/fold_*/v2_evaluated_trajectory.csv"))
    if len(paths) != 30:
        raise RuntimeError(f"Expected 30 frozen V2 runs, found {len(paths)}")
    return paths


def load(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    cols = ("time_s", "gt_east_m", "gt_north_m", "gt_heading_rad",
            "estimate_east_m", "estimate_north_m")
    if not set(cols).issubset(df):
        raise ValueError(f"Missing columns in {path}")
    a = {c: df[c].to_numpy(float) for c in cols}
    t = a["time_s"]
    if not np.all(np.diff(t) > 0) or abs(np.median(np.diff(t)) - DT) > 0.005:
        raise ValueError(f"Unexpected clock in {path}")
    if any(not np.isfinite(v).all() for v in a.values()):
        raise ValueError(f"Nonfinite trajectory in {path}")
    return a


def fence_membership(x: np.ndarray, y: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def fences(a: dict[str, np.ndarray]) -> list[tuple[str, float, float, float]]:
    x0, y0 = a["gt_east_m"][0], a["gt_north_m"][0]
    th = a["gt_heading_rad"][0]
    c, s = math.cos(th), math.sin(th)
    return [(name, x0 + c * fwd - s * left, y0 + s * fwd + c * left, radius)
            for name, fwd, left, radius in FENCES]


def causal_stream(a: dict[str, np.ndarray], kind: str, value_ms: int,
                  jitter_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return available virtual index and reported stamp for each evaluation tick."""
    t = a["time_s"] - a["time_s"][0]
    if kind == "delay":
        arrival = t + value_ms / 1000
        stamp = t.copy()
    else:
        # Existing timing study corrupts virtual timestamps, not physical truth.
        # Unlike its offline interpolation, a monitor must use arrived samples only.
        arrival = t.copy()
        rng = np.random.default_rng(jitter_seed)
        stamp = t + rng.normal(0, value_ms / 1000, len(t))
    idx = np.searchsorted(arrival, t, side="right") - 1
    return idx, arrival, stamp


def reference_updates(a: dict[str, np.ndarray], arrival: np.ndarray,
                      reported: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulate 1-Hz reference access; compare only time-matched arrived states."""
    t = a["time_s"] - a["time_s"][0]
    schedule = np.arange(0, len(t), 10, dtype=int)
    update = np.zeros(len(t), dtype=bool)
    err = np.full(len(t), np.nan)
    for j in schedule:
        # Reference sample acquired at t[j], delivered 0.2 s later.
        decision = int(np.searchsorted(t, t[j] + 0.2, side="left"))
        if decision >= len(t):
            continue
        # Monitor knows reported time, not hidden true stamp error.
        # Timestamp corruption can make reported stamps nonmonotone. Search a
        # bounded neighborhood of the 10-Hz source index, not a sorted stamp
        # prefix; all candidates must have arrived.
        near = np.arange(max(0, j - 4), min(len(t), j + 5))
        candidates = near[(arrival[near] <= t[decision] + 1e-9) &
                          (reported[near] <= t[j] + 1e-9) &
                          (reported[near] >= t[j] - 0.15)]
        if not len(candidates):
            continue
        match = int(candidates[np.argmax(reported[candidates])])
        update[decision] = True
        err[decision] = math.hypot(a["gt_east_m"][j] - a["estimate_east_m"][match],
                                   a["gt_north_m"][j] - a["estimate_north_m"][match])
    return update, err


def qualification(update: np.ndarray, error: np.ndarray, t: np.ndarray,
                  tolerance: float, aoi_limit: float) -> dict[str, np.ndarray]:
    latest = np.maximum.accumulate(np.where(update, np.arange(len(t)), -1))
    seen = latest >= 0
    score = np.full(len(t), np.nan)
    score[seen] = error[latest[seen]]
    age = np.full(len(t), np.inf)
    # Source time is 0.2 s before a reference update arrives, by construction.
    age[seen] = t[seen] - (t[latest[seen]] - 0.2)
    passed = seen & (score <= tolerance)
    fresh = passed & (age <= aoi_limit + 1e-9)
    return {"always": np.ones(len(t), bool), "error": passed,
            "error_fresh": fresh, "contract": fresh.copy()}


def score(release: np.ndarray, truth: np.ndarray, answer: np.ndarray,
          valid: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    eligible = valid & (weight > 0)
    denom = float(weight[eligible].sum())
    rel = eligible & release
    released = float(weight[rel].sum())
    wrong = rel & (truth != answer)
    return {
        "evaluable_s": denom, "released_s": released,
        "availability": released / denom if denom else np.nan,
        "released_error": float(weight[wrong].sum()) / released if released else np.nan,
        "unnecessary_abstention": float(weight[eligible & ~release & (truth == answer)].sum()) / denom if denom else np.nan,
        "false_inside_s": float(weight[wrong & answer].sum()),
        "false_outside_s": float(weight[wrong & ~answer].sum()),
        "excluded_evaluator_s": float(weight[~valid].sum()),
    }


def evaluate_one(path: Path, kind: str, value_ms: int, jitter_seed: int) -> list[dict]:
    a = load(path)
    t = a["time_s"] - a["time_s"][0]
    idx, arrival, reported = causal_stream(a, kind, value_ms, jitter_seed)
    update, err = reference_updates(a, arrival, reported)
    valid = idx >= 0
    weight = np.minimum(np.diff(t, append=t[-1]), DT)
    weight[-1] = 0
    seq = path.parent.name.split("_", 2)[-1]
    seed = path.parent.parent.name
    rows = []
    for fence_id, cx, cy, radius in fences(a):
        truth = fence_membership(a["gt_east_m"], a["gt_north_m"], cx, cy, radius)
        answer = np.zeros(len(t), bool)
        answer[valid] = fence_membership(a["estimate_east_m"][idx[valid]],
                                         a["estimate_north_m"][idx[valid]], cx, cy, radius)
        crossings = int(np.count_nonzero(np.diff(truth.astype(int))))
        occupancy = float(np.sum(weight[truth]) / np.sum(weight))
        for tol in TOLERANCES_M:
            for age_limit in AOI_LIMITS_S:
                rules = qualification(update, err, t, tol, age_limit)
                # Qualification at tick k governs answers at k+1 onward.
                for method in METHODS:
                    release = np.r_[False, rules[method][:-1]]
                    outcome = score(release, truth, answer, valid, weight)
                    rows.append({"sequence": seq, "replicate": seed, "source": str(path.relative_to(ROOT)),
                                 "timing": kind, "value_ms": value_ms, "jitter_seed": jitter_seed,
                                 "fence": fence_id, "radius_m": radius, "crossings": crossings,
                                 "inside_fraction": occupancy, "tolerance_m": tol,
                                 "aoi_limit_s": age_limit, "method": method,
                                 "monitor_updates": int(update.sum()), **outcome})
    return rows


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["sequence", "timing", "value_ms", "tolerance_m", "aoi_limit_s", "method"]
    # Duration-pool fences and jitter repeats *within* each seed, then average
    # seeds within physical sequence; never count them as new physical sequences.
    df = df.copy()
    df["wrong_s"] = df["released_error"].fillna(0) * df["released_s"]
    df["unnecessary_s"] = df["unnecessary_abstention"] * df["evaluable_s"]
    per_seed = df.groupby(keys + ["replicate"], as_index=False)[
        ["evaluable_s", "released_s", "wrong_s", "unnecessary_s",
         "false_inside_s", "false_outside_s"]].sum()
    per_seed["availability"] = per_seed.released_s / per_seed.evaluable_s
    per_seed["released_error"] = per_seed.wrong_s / per_seed.released_s.replace(0, np.nan)
    per_seed["unnecessary_abstention"] = per_seed.unnecessary_s / per_seed.evaluable_s
    per_seq = per_seed.groupby(keys, as_index=False)[
        ["availability", "released_error", "unnecessary_abstention",
         "false_inside_s", "false_outside_s"]].mean()
    macro = per_seq.groupby(keys[1:], as_index=False).agg(
        availability=("availability", "mean"),
        released_error=("released_error", "mean"),
        unnecessary_abstention=("unnecessary_abstention", "mean"),
        sequences=("sequence", "nunique"))
    return per_seq, macro


def write_outputs(df: pd.DataFrame, manifest: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_run_fence_operating_point.csv", index=False)
    seq, macro = aggregate(df)
    seq.to_csv(OUT / "per_sequence_operating_point.csv", index=False)
    macro.to_csv(OUT / "macro_operating_point.csv", index=False)
    selected = macro[(macro.tolerance_m == 1.0) & (macro.aoi_limit_s == 1.0)]
    selected.to_csv(OUT / "selected_results_table.csv", index=False)
    # Bootstrap physical sequences, not nested runs/fences/timestamps.
    rng = np.random.default_rng(42)
    ci_rows = []
    for timing, value in (("delay", 0), ("delay", 200), ("jitter", 50)):
        g = seq[(seq.timing == timing) & (seq.value_ms == value) &
                (seq.tolerance_m == 1.0) & (seq.aoi_limit_s == 1.0)]
        pivot = g.pivot(index="sequence", columns="method",
                        values=["availability", "released_error"])
        for metric in ("availability", "released_error"):
            for method, baseline in (("error", "always"), ("error_fresh", "error"),
                                     ("contract", "error_fresh")):
                delta = (pivot[(metric, method)] - pivot[(metric, baseline)]).to_numpy(float)
                finite = delta[np.isfinite(delta)]
                if len(finite):
                    draw = rng.choice(finite, size=(5000, len(finite)), replace=True).mean(axis=1)
                    low, high = np.quantile(draw, [0.025, 0.975])
                    mean = float(np.mean(finite))
                else:
                    mean = low = high = np.nan
                ci_rows.append({"timing": timing, "value_ms": value, "metric": metric,
                                "method": method, "baseline": baseline, "n_sequences": len(finite),
                                "mean_difference": mean, "ci95_low": low, "ci95_high": high})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT / "paired_sequence_bootstrap.csv", index=False)
    table_rows = []
    rng = np.random.default_rng(43)
    for timing, value in (("delay", 0), ("delay", 200), ("jitter", 50)):
        g = seq[(seq.timing == timing) & (seq.value_ms == value) &
                (seq.tolerance_m == 1.0) & (seq.aoi_limit_s == 1.0)]
        for method in METHODS:
            h = g[g.method == method]
            row = {"timing": timing, "value_ms": value, "method": method,
                   "n_sequences": h.sequence.nunique()}
            for metric in ("availability", "released_error", "unnecessary_abstention"):
                values = h[metric].dropna().to_numpy(float)
                draw = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
                row[metric] = float(np.mean(values))
                row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = np.quantile(draw, [0.025, 0.975])
            table_rows.append(row)
    publication_table = pd.DataFrame(table_rows)
    publication_table.to_csv(OUT / "publication_results_table.csv", index=False)
    matched_rows = []
    for timing, value in (("delay", 0), ("delay", 200), ("jitter", 50)):
        g = macro[(macro.timing == timing) & (macro.value_ms == value)]
        for left, right in (("error", "error_fresh"), ("error_fresh", "contract")):
            a, b = g[g.method == left], g[g.method == right]
            for _, x in a.iterrows():
                j = (b.availability - x.availability).abs().idxmin()
                y = b.loc[j]
                gap = abs(float(x.availability - y.availability))
                if gap <= 0.01:
                    matched_rows.append({
                        "timing": timing, "value_ms": value, "method_a": left,
                        "method_b": right, "availability_a": x.availability,
                        "availability_b": y.availability, "availability_gap": gap,
                        "released_error_a": x.released_error,
                        "released_error_b": y.released_error,
                        "error_difference_b_minus_a": y.released_error - x.released_error,
                        "tolerance_a_m": x.tolerance_m, "aoi_a_s": x.aoi_limit_s,
                        "tolerance_b_m": y.tolerance_m, "aoi_b_s": y.aoi_limit_s,
                    })
    pd.DataFrame(matched_rows).to_csv(OUT / "matched_availability_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, kind, value in ((axes[0], "delay", 0), (axes[1], "delay", 200)):
        g = macro[(macro.timing == kind) & (macro.value_ms == value)]
        for method, color in (("always", "#333333"), ("error", "#0072B2"),
                              ("error_fresh", "#D55E00")):
            h = g[g.method == method].sort_values("availability")
            ax.scatter(h.availability, h.released_error, s=18, alpha=.5, color=color,
                       label="error + freshness = contract" if method == "error_fresh" else method)
            point = selected[(selected.timing == kind) & (selected.value_ms == value) &
                             (selected.method == method)]
            if len(point):
                ax.scatter(point.availability, point.released_error, s=100, marker="x",
                           linewidths=2, color=color)
        ax.set(title=f"Virtual delivery delay: {value} ms", xlabel="Released-answer availability")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Error among released answers")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "error_availability.png", dpi=180)
    plt.close(fig)
    lines = ["# Geofence decision-value analysis", "",
             "## Input and protocol audit", "",
             "- Source: 30 frozen V2 evaluated trajectories: ten held-out physical",
             "  sequences and three seeds per sequence. Source identifiers and SHA-256",
             "  checksums are recorded in protocol_manifest.json.",
             "- The associated run manifests define each fold's held-out sequence and",
             "  its training/validation sequence assignments; this analysis does not",
             "  refit or select a checkpoint.",
             "- Time is the saved 10-Hz i2Nav clock. Position is local ENU in meters.",
             "  Physical and virtual trajectories already share that frame; no spatial",
             "  alignment or outcome-dependent interpolation is applied.",
             "- The frozen files contain complete reference pose but no per-sample",
             "  reference-quality or missingness flags. Complete truth is used only by",
             "  the evaluator; monitor access follows the simulated schedule below.",
             "- Delivery delay changes when a virtual state becomes available.",
             "  Timestamp jitter changes only the timestamp reported to the monitor;",
             "  hidden true timestamps remain evaluator metadata.",
             "- Out-of-order timestamp reports are handled from arrived samples only.",
             "  Repeated cached evidence does not create a new reference update.", "",
             "## Causal replay boundary", "",
             "This is a simulated reference-availability trace replay, not measured live sensing.",
             "The monitor receives 1-Hz physical observations after 0.2 s; the evaluator retains full offline truth.",
             "The virtual stream is delayed or timestamp-corrupted; reference arrival itself is not perturbed.",
             "Qualification at a tick applies from the next tick. The truth for that answer is unavailable to the monitor.",
             "No GPS/heading gate is imposed on this position-only task. Source quality flags are absent.",
             "Thus the full applicable contract is operationally identical to position error plus freshness.",
             "The timestamp-jitter implementation is causal here, unlike the existing offline interpolation sensitivity study.",
             "The 1.0-m, 1.0-s point is an existing illustrative contract setting, not an application-defined safety requirement.", "",
             "## Selected operating point", "",
             selected.to_markdown(index=False, floatfmt=".3f"), "",
             "Each macro averages within-run fences and jitter seeds, then three seeds within each of ten physical sequences.",
             "No timestamps, fences, or jitter draws are independent experimental units.",
             "No paired advantage is claimed when availability differs materially or when methods coincide.",
             "This analysis assumes an independent reference feed and cannot establish reference-free runtime qualification.",
             "The synthetic initial-pose geofences are task probes, not surveyed application regions.", "",
             "## Task balance and interpretation", "",
             df.drop_duplicates(["sequence", "replicate", "fence"])[
                 ["fence", "crossings", "inside_fraction"]].groupby("fence").agg(
                     crossings_min=("crossings", "min"),
                     crossings_median=("crossings", "median"),
                     inside_fraction_median=("inside_fraction", "median")).to_markdown(floatfmt=".3f"), "",
             "Occupancy is sparse; the lateral fence often has no crossings. It is retained",
             "because exclusion after inspecting outcomes would bias the comparison.",
             "At the selected synchronized point, always-use error is lower than the",
             "qualified rules despite higher availability. Qualification therefore does",
             "not demonstrate prospective geofence protection on these task probes.",
             "The matched error-plus-freshness and full-contract rules have exactly",
             "equal decisions, so their paired difference and confidence interval are zero.",
             "Error-only and freshness-filtered rules have different availability;",
             "the selected-point error rates are not a fair matched-availability win.", "",
             "Across the frozen grid, error-only and error-plus-freshness have",
             "identical points whenever the latter uses the nonbinding 1.5-s AoI",
             "limit. At matched availability, freshness therefore adds no task",
             "benefit in this replay; tighter freshness limits only abstain more.",
             "The full contract is identical to error-plus-freshness throughout.", "",
             "## Sequence-level paired uncertainty", "",
             ci.to_markdown(index=False, floatfmt=".4f"), "",
             "The intervals are percentile bootstrap intervals from 5000 resamples of",
             "the ten physical sequences; they are descriptive, not population guarantees.", "",
             "The compact table with absolute sequence-bootstrap intervals is saved",
             "as publication_results_table.csv; matched-availability operating",
             "points are saved as matched_availability_comparison.csv.", "",
             "## Manuscript-ready text", "",
             "**Methods.** We evaluated a synthetic position-only geofence service on",
             "30 frozen i2Nav V2 held-out trajectories (ten physical sequences, three",
             "seeds each). A simulated 1-Hz independent-reference feed arrived 0.2 s",
             "after acquisition. Four qualification rules used only causal, matched",
             "observations; answers were scored at subsequent 10-Hz ticks against",
             "offline reference positions withheld from the monitor. Three fixed",
             "initial-pose-relative geofences and a frozen tolerance/AoI grid were",
             "evaluated under virtual-stream delivery delay and timestamp corruption.",
             "Seeds, fences, and jitter replicates were aggregated within each sequence.",
             "",
             "**Results.** At the illustrative 1-m/1-s synchronized point, always use",
             "released nearly all answers with 0.3% error, while the full contract",
             "released about 46% with 1.1% error among released answers. The contract",
             "was exactly equivalent to the position-error-plus-freshness rule because",
             "these traces contain no extra reference-quality gates. Sparse fence",
             "occupancy and limited boundary crossings constrain generalization;",
             "this analysis does not establish superior prospective task protection.", "",
             "## Reproduce", "",
             "`python -m DigitalTwin.analysis.geofence_decision_value`", ""]
    (OUT / "decision_value_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    OUT = args.output
    if args.summarize_existing:
        manifest = json.loads((OUT / "protocol_manifest.json").read_text(encoding="utf-8"))
        write_outputs(pd.read_csv(OUT / "per_run_fence_operating_point.csv"), manifest)
        print(OUT / "decision_value_report.md")
        return
    paths = discover()
    manifest = {
        "schema": "i2nav_geofence_decision_value_v1",
        "status": "frozen_before_comparison",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True, check=True).stdout.strip(),
        "frozen_v2_commit": "6540c01f90f3c1074de0d8dae9964a5276fbbc91",
        "source_paths": [str(p.relative_to(ROOT)) for p in paths],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "reference_schedule": "first sample and every tenth 10-Hz sample, arrival 0.2 s later",
        "geofences_initial_heading_frame": FENCES, "boundary": "inside on boundary",
        "delay_ms": DELAYS_MS, "jitter_ms": JITTER_MS, "jitter_seeds": JITTER_SEEDS,
        "tolerances_m": TOLERANCES_M, "aoi_limits_s": AOI_LIMITS_S,
        "selected_point": {"tolerance_m": 1.0, "aoi_limit_s": 1.0},
        "quality_flags": "not present in frozen evaluated trajectories",
        "evaluation_clock_s": DT, "alignment": "none", "coordinate_frame": "local ENU meters",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = []
    conditions = [("delay", d, 0) for d in DELAYS_MS]
    conditions += [("jitter", j, seed) for j in JITTER_MS for seed in ((0,) if j == 0 else JITTER_SEEDS)]
    for path in paths:
        for kind, ms, seed in conditions:
            rows.extend(evaluate_one(path, kind, ms, seed))
    write_outputs(pd.DataFrame(rows), manifest)
    print(OUT / "decision_value_report.md")


if __name__ == "__main__":
    main()

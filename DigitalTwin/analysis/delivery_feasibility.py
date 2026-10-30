"""Offline delivery feasibility bounds and minimum-transmission schedules.

Run: python -m DigitalTwin.analysis.delivery_feasibility
Uses verified outer-test trajectories; source models are never fitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from DigitalTwin.analysis import service_timing_budget_study as base

ROOT = base.ROOT
STRIDES = np.array([1, 2, 5, 10], dtype=int)
GROUPS = [(s["service"],) for s in base.SERVICES] + [
    ("local_1s", "local_5s", "local_10s"),
    tuple(s["service"] for s in base.SERVICES),
]
BANKS = {
    "centered": ("results/doubly_nested_loso_slurm", "results/doubly_nested_loso_evidence", 1),
    "causal": ("results/doubly_nested_causal_loso_slurm",
               "results/doubly_nested_causal_loso_slurm/service_evidence", 0),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def eligible(n, service, prefix=300):
    return np.arange(prefix + int(round(service["horizon_s"] * 10)), n)


def passes(a, service, times, current, previous=None):
    """Vectorized physical checks using exactly the existing SE(2) discrepancy."""
    if service["family"] == "global":
        pos = np.hypot(a["estimate_east_m"][current] - a["gt_east_m"][times],
                       a["estimate_north_m"][current] - a["gt_north_m"][times])
        head = np.abs(np.rad2deg(base.wrap(a["estimate_heading_rad"][current]
                                         - a["gt_heading_rad"][times])))
    else:
        h = int(round(service["horizon_s"] * 10))
        gp = base.relative(a["gt_east_m"], a["gt_north_m"], a["gt_heading_rad"], times - h, times)
        ep = base.relative(a["estimate_east_m"], a["estimate_north_m"],
                           a["estimate_heading_rad"], previous, current)
        pos = np.hypot(ep[0] - gp[0], ep[1] - gp[1])
        head = np.abs(np.rad2deg(base.wrap(ep[2] - gp[2])))
    return (pos <= service["pos_tol_m"]) & (head <= service["heading_tol_deg"])


def replay(a, service, indices, prefix=300):
    n = len(indices)
    use = eligible(n, service, prefix)
    h = int(round(service["horizon_s"] * 10))
    obs = indices[use] >= 0
    if h:
        obs &= indices[use - h] >= 0
    times = use[obs]
    k = indices[times]
    physical = passes(a, service, times, k, indices[times - h] if h else None)
    # Integer ages preserve the inclusive boundary in the primary replay.
    fresh = times - k <= int(round(service["aoi_limit_s"] * 10))
    count = len(times)
    return {"coverage": count / len(use),
            "physical_satisfaction": float(physical.mean()) if count else np.nan,
            "freshness_satisfaction": float(fresh.mean()) if count else np.nan,
            "joint_satisfaction": float((physical & fresh).mean()) if count else np.nan,
            "observable_samples": count, "eligible_samples": len(use)}


def summarize(arrays, indices):
    rows = []
    for service in base.SERVICES:
        values = [replay(a, service, indices) for a in arrays]
        row = {"service": service["service"]}
        for key in values[0]:
            row[key] = float(np.mean([v[key] for v in values]))
        if len({v["observable_samples"] for v in values}) != 1:
            raise AssertionError("Seed-level observability denominators differ")
        for key in ["physical_satisfaction", "freshness_satisfaction", "joint_satisfaction"]:
            total = sum(round(v[key] * v["observable_samples"]) for v in values)
            row[key] = total / sum(v["observable_samples"] for v in values)
        row["qualified"] = int(row["coverage"] >= base.MIN_COVERAGE
                               and row["joint_satisfaction"] >= base.TARGET)
        rows.append(row)
    return rows


def schedule_indices(n, actions, lag, block=100, prefix=300):
    k = np.arange(n)
    idx = k - lag
    tail = k >= prefix
    offset = k[tail] - prefix
    stride = STRIDES[np.asarray(actions)[offset // block]]
    idx[tail] = prefix + (offset // block) * block + ((offset % block) // stride) * stride - lag
    idx[idx < 0] = -1
    return idx


def optimistic_bound(arrays, service, lag, cap=15):
    """Relax schedule consistency but keep a common endpoint choice across seeds.

    The earlier local endpoint may have age up to cap, regardless of whether
    it was fresh. Both endpoint ages are bounded by the declared schedule class.
    Maximizing separately at each time is an upper bound, not a witness policy.
    """
    n = len(arrays[0]["time_s"])
    ts = eligible(n, service)[:, None]
    max_new = min(cap, int(round(service["aoi_limit_s"] * 10)))
    if lag > max_new:
        return 0., np.zeros(len(ts), dtype=np.uint8)
    current = ts - np.arange(lag, max_new + 1)[None, :]
    best = np.zeros(len(ts), dtype=np.uint8)
    h = int(round(service["horizon_s"] * 10))
    old_ages = range(lag, cap + 1) if h else [0]
    for old_age in old_ages:
        previous = ts - h - old_age if h else None
        score = np.zeros(current.shape, dtype=np.uint8)
        valid = current >= 0
        if h:
            valid = valid & (previous >= 0) & (previous <= current)
        for a in arrays:
            score += passes(a, service, ts, np.maximum(current, 0),
                            np.maximum(previous, 0) if h else None) & valid
        best = np.maximum(best, score.max(axis=1))
    return float(best.sum() / (len(arrays) * len(best))), best


def build_factors(arrays, lag, block=100, prefix=300):
    """An edge encodes the rates in two adjacent blocks; horizons <= block.

    All rates deliver at each block boundary, with source time shifted by lag.
    Thus the held state at a receiver time depends only on that block's action.
    Local checks involve at most the current and previous blocks.
    """
    n = len(arrays[0]["time_s"])
    if block < max(int(round(s["horizon_s"] * 10)) for s in base.SERVICES):
        raise ValueError("Block must cover the longest service horizon")
    blocks = math.ceil((n - prefix) / block)
    candidates = [schedule_indices(n, np.full(blocks, a), lag, block, prefix)
                  for a in range(len(STRIDES))]
    edges = [(b, p, c) for b in range(blocks)
             for p in (range(len(STRIDES)) if b else [0]) for c in range(len(STRIDES))]
    lookup = {edge: i for i, edge in enumerate(edges)}
    counts = np.zeros((len(edges), len(base.SERVICES)), dtype=int)
    costs = np.array([math.ceil(min(block, n - prefix - b * block) / STRIDES[c])
                      for b, _, c in edges], dtype=int)
    for si, s in enumerate(base.SERVICES):
        ts = eligible(n, s, prefix)
        h = int(round(s["horizon_s"] * 10))
        bs = (ts - prefix) // block
        for p in range(len(STRIDES)):
            for c in range(len(STRIDES)):
                curr = candidates[c][ts]
                prev = None
                if h:
                    old_ts = ts - h
                    same = (old_ts - prefix) // block == bs
                    prev = np.where(same, candidates[c][old_ts], candidates[p][old_ts])
                fresh = ts - curr <= int(round(s["aoi_limit_s"] * 10))
                success = np.zeros(len(ts), dtype=int)
                for a in arrays:
                    success += passes(a, s, ts, curr, prev) & fresh
                per_block = np.bincount(bs, weights=success, minlength=blocks).astype(int)
                for b in range(blocks):
                    if b or p == 0:
                        counts[lookup[(b, p, c)], si] = per_block[b]
    denominators = np.array([len(eligible(n, s, prefix)) * len(arrays) for s in base.SERVICES])
    return edges, counts, costs, denominators


def solve_schedule(edges, counts, costs, denominators, services, seconds=20.):
    groups = [next(i for i, s in enumerate(base.SERVICES) if s["service"] == name)
              for name in services]
    blocks = max(b for b, _, _ in edges) + 1
    rr, cc, vv = [], [], []
    for e, (b, p, c) in enumerate(edges):
        if b == 0:
            rr.append(0); cc.append(e); vv.append(1.)
        if b < blocks - 1:
            rr.append(1 + b * len(STRIDES) + c); cc.append(e); vv.append(1.)
        if b > 0:
            rr.append(1 + (b - 1) * len(STRIDES) + p); cc.append(e); vv.append(-1.)
    flow_rows = 1 + (blocks - 1) * len(STRIDES)
    for j, si in enumerate(groups):
        for e in np.flatnonzero(counts[:, si]):
            rr.append(flow_rows + j); cc.append(e); vv.append(float(counts[e, si]))
    matrix = coo_matrix((vv, (rr, cc)), shape=(flow_rows + len(groups), len(edges))).tocsc()
    lower = np.r_[1., np.zeros(flow_rows - 1),
                  [math.ceil(base.TARGET * denominators[si] - 1e-9) for si in groups]]
    upper = np.r_[1., np.zeros(flow_rows - 1), np.full(len(groups), np.inf)]
    started = time.monotonic()
    result = milp(costs.astype(float), integrality=np.ones(len(edges)),
                  bounds=Bounds(0., 1.), constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": seconds, "mip_rel_gap": 0.})
    info = {"solver_status": int(result.status), "solver_message": str(result.message),
            "solve_seconds": time.monotonic() - started,
            "objective_lower_bound": float(getattr(result, "mip_dual_bound", np.nan)
                                           if getattr(result, "mip_dual_bound", None) is not None else np.nan),
            "mip_gap": float(getattr(result, "mip_gap", np.nan)
                             if getattr(result, "mip_gap", None) is not None else np.nan)}
    if result.x is None:
        return None, info
    chosen = np.flatnonzero(result.x > .5)
    if len(chosen) != blocks:
        raise AssertionError("Solver witness is not a single complete path")
    actions = np.empty(blocks, dtype=int)
    for e in chosen:
        b, p, c = edges[e]
        if b and actions[b - 1] != p:
            raise AssertionError("Inconsistent adjacent-block history")
        actions[b] = c
    achieved = counts[chosen].sum(axis=0) / denominators
    if any(achieved[si] + 1e-12 < base.TARGET for si in groups):
        raise AssertionError("Solver returned an invalid qualification witness")
    info.update(transmissions=int(costs[chosen].sum()),
                selected_counts=counts[chosen].sum(axis=0).tolist())
    return actions, info


def load_recordings(bank):
    folder, evidence, _ = BANKS[bank]
    manifest_path = ROOT / folder / "merged_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("verified_trajectories") != 300:
        raise ValueError("A complete verified nested bank is required")
    if bank == "causal" and not manifest.get("causal_no_lookahead_preflight", {}).get("passed"):
        raise ValueError("Missing causal feature preflight")
    entries = [e for e in manifest["trajectories"] if e["role"] == "outer_test"]
    if len(entries) != 30 or len({(e["target"], e["seed"]) for e in entries}) != 30:
        raise ValueError("Expected 30 distinct outer-test trajectories")
    rows = {}
    hashes = {str(manifest_path.relative_to(ROOT)): digest(manifest_path)}
    for e in entries:
        if e["outer"] != e["target"]:
            raise ValueError("Incorrect outer-test identity")
        path = ROOT / e["trajectory"]
        hashes[str(path.relative_to(ROOT))] = digest(path)
        if hashes[str(path.relative_to(ROOT))] != e["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
        a = base.load(path)
        if any(not np.isfinite(v).all() for v in a.values()):
            raise ValueError("Nonfinite source signals")
        if not np.allclose(a["time_s"], np.arange(len(a["time_s"])) / 10, atol=1e-6, rtol=0):
            raise ValueError("Experiment requires the audited 10-Hz clock")
        rows.setdefault(e["target"], []).append((e, a))
    for seq, recordings in rows.items():
        if len(recordings) != 3 or {e["seed"] for e, _ in recordings} != {42, 1042, 2042}:
            raise ValueError(f"Incorrect seeds for {seq}")
        first = recordings[0][1]
        for _, a in recordings[1:]:
            for key in ["time_s", "gt_east_m", "gt_north_m", "gt_heading_rad"]:
                if first[key].shape != a[key].shape or not np.allclose(first[key], a[key], atol=1e-8, rtol=0):
                    raise ValueError("Seeds do not share a physical reference and clock")
    return rows, hashes, ROOT / evidence / "per_run_response_surface.csv"


def reproduce(recordings, table_path):
    table = pd.read_csv(table_path)
    worst, checks = 0., 0
    for seq, entries in recordings.items():
        for e, a in entries:
            rows = table[(table.outer_sequence == seq) & (table.sequence == seq)
                         & (table.replicate == f"replicate_{(e['seed'] - 42) // 1000 + 1:02d}_base{e['seed']}")]
            if len(rows) != 80:
                raise AssertionError("Missing outer-test reproduction cells")
            for row in rows.itertuples():
                s = next(s for s in base.SERVICES if s["service"] == row.service)
                idx = base.delivered_indices(a["time_s"], row.rate_hz, row.delay_ms / 1000)
                m = replay(a, s, idx)
                for key in ["coverage", "physical_satisfaction", "freshness_satisfaction", "joint_satisfaction"]:
                    worst = max(worst, abs(m[key] - getattr(row, key)))
                checks += 1
    if worst > 1e-12:
        raise AssertionError(f"Existing response surface not reproduced: {worst}")
    return {"reproduced_run_service_delivery_cells": checks, "maximum_absolute_difference": worst}


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"inputs": {}, "reproduction": {}, "source_code_sha256": digest(__file__),
                  "manuscript_sha256_before": digest(ROOT / "manuscript/main.tex"),
                  "type": "offline reference-informed optimization and upper bounds",
                  "block_seconds": args.block_seconds, "maximum_state_age_seconds": args.max_age_seconds,
                  "delay_ms": args.delays_ms, "rates_hz": (10 / STRIDES).tolist(),
                  "timing": {"centered": "100-ms feature availability plus network delay",
                             "causal": "zero modeled computation latency plus network delay"},
                  "seed_handling": "One common schedule for three seeds; average their satisfaction, never their states",
                  "bound_class": "Timestamp-ordered 10-Hz source states, fixed latency, held age <= declared cap at both local endpoints",
                  "optimizer_class": "Periodic 1/2/5/10-Hz arrivals in fixed blocks, all block starts updated; no extrapolation",
                  "optimizer_uses_reference": True, "solver_time_limit_seconds": args.solver_seconds}
    bounds_rows, baseline_rows, solution_rows, replay_rows = [], [], [], []
    for bank in args.banks:
        recordings, hashes, table_path = load_recordings(bank)
        provenance["inputs"].update(hashes)
        provenance["inputs"][str(table_path.relative_to(ROOT))] = digest(table_path)
        provenance["reproduction"][bank] = reproduce(recordings, table_path)
        for seq, entries in sorted(recordings.items()):
            arrays = [a for _, a in sorted(entries, key=lambda z: z[0]["seed"])]
            n = len(arrays[0]["time_s"])
            block = int(round(args.block_seconds * 10))
            nblocks = math.ceil((n - 300) / block)
            for delay in args.delays_ms:
                lag = BANKS[bank][2] + math.ceil(delay / 100)
                cap = int(round(args.max_age_seconds * 10))
                if cap < lag + int(STRIDES.max()) - 1:
                    raise ValueError("Maximum age does not cover all optimizer actions")
                context = dict(bank=bank, sequence=seq, network_delay_ms=delay,
                               availability_lag_samples=lag, postprefix_seconds=(n - 300) / 10)
                print(f"{bank} {seq} delay={delay}: bounds and schedules", flush=True)
                upper = {}
                for s in base.SERVICES:
                    value, sample_bound = optimistic_bound(arrays, s, lag, cap)
                    upper[s["service"]] = value
                    bounds_rows.append({**context, "service": s["service"], "optimistic_joint_upper_bound": value,
                                        "certified_infeasible": int(value + 1e-12 < base.TARGET)})
                    dest = out / "sample_bounds" / bank / seq
                    dest.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dest / f"{delay}ms_{s['service']}.npz", maximum_passing_seeds=sample_bound)
                edges, counts, costs, denominators = build_factors(arrays, lag, block)
                baselines = []
                for action, stride in enumerate(STRIDES):
                    actions = np.full(nblocks, action)
                    idx = schedule_indices(n, actions, lag, block)
                    tx = sum(math.ceil(min(block, n - 300 - b * block) / stride) for b in range(nblocks))
                    metrics = summarize(arrays, idx)
                    baselines.append((tx, metrics))
                    for m in metrics:
                        if m["joint_satisfaction"] > upper[m["service"]] + 1e-12:
                            raise AssertionError("Upper bound below a feasible baseline")
                        baseline_rows.append({**context, "policy": f"periodic_{10 / stride:g}hz", "transmissions": tx, **m})
                for group in GROUPS:
                    name = "+".join(group)
                    row = {**context, "service_group": name,
                           "best_periodic_transmissions": min((tx for tx, ms in baselines
                               if all(next(m for m in ms if m['service'] == s)['qualified'] for s in group)), default=np.nan)}
                    if any(upper[s] + 1e-12 < base.TARGET for s in group):
                        row["status"] = "certified_infeasible_bounded_age_class"
                    else:
                        actions, info = solve_schedule(edges, counts, costs, denominators, group, args.solver_seconds)
                        row.update(info)
                        if actions is None:
                            row["status"] = "infeasible_block_family" if info["solver_status"] == 2 else "unresolved_solver_limit"
                        else:
                            row["status"] = "optimal_block_family" if info["solver_status"] == 0 else "feasible_solver_limit"
                            idx = schedule_indices(n, actions, lag, block)
                            metrics = summarize(arrays, idx)
                            for si, m in enumerate(metrics):
                                expected = info["selected_counts"][si] / denominators[si]
                                if abs(expected - m["joint_satisfaction"]) > 1e-12:
                                    raise AssertionError("Optimizer factors disagree with direct replay")
                                if m["joint_satisfaction"] > upper[m["service"]] + 1e-12:
                                    raise AssertionError("Witness exceeds upper bound")
                                replay_rows.append({**context, "service_group": name, **m})
                            if not all(next(m for m in metrics if m["service"] == s)["qualified"] for s in group):
                                raise AssertionError("Direct replay rejects claimed feasible schedule")
                            dest = out / "schedules" / bank / seq
                            dest.mkdir(parents=True, exist_ok=True)
                            pd.DataFrame({"receiver_block_start_s": 30 + np.arange(nblocks) * args.block_seconds,
                                          "rate_hz": 10 / STRIDES[actions]}).to_csv(dest / f"{delay}ms_{name}.csv", index=False)
                            row.pop("selected_counts", None)
                            row["effective_rate_hz"] = row["transmissions"] / context["postprefix_seconds"]
                            if np.isfinite(row["best_periodic_transmissions"]):
                                row["transmission_reduction_vs_best_periodic"] = 1 - row["transmissions"] / row["best_periodic_transmissions"]
                    solution_rows.append(row)
                # Save progress after each complete recording/condition.
                for filename, values in [("feasibility_bounds.csv", bounds_rows), ("periodic_baselines.csv", baseline_rows),
                                         ("optimization_results.csv", solution_rows), ("optimized_replay.csv", replay_rows)]:
                    pd.DataFrame(values).to_csv(out / filename, index=False)
    provenance["manuscript_sha256_after"] = digest(ROOT / "manuscript/main.tex")
    if provenance["manuscript_sha256_before"] != provenance["manuscript_sha256_after"]:
        raise AssertionError("Manuscript changed during experiment")
    provenance["status"] = "complete"
    (out / "manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    make_report(out)


def make_report(out):
    b = pd.read_csv(out / "feasibility_bounds.csv")
    o = pd.read_csv(out / "optimization_results.csv")
    lines = ["# Delivery feasibility and minimum communication", "",
             "All source trajectories and references are frozen. One schedule is shared across three model seeds.",
             "Centered states include 100 ms of feature availability delay; causal states use source-time availability.",
             "The exact optimizer chooses one of four periodic rates per 10-s block. Its minima apply to this family.",
             "The broader bound permits any consistent held-state schedule with fixed latency and age at most 1.5 s.",
             "It relaxes consistency between samples; a bound below 0.80 certifies infeasibility within that class.",
             "All optimization uses reference outcomes retrospectively. It is an achievable-limit benchmark, not an online scheduler.", ""]
    for (bank, delay), group in o.groupby(["bank", "network_delay_ms"]):
        bounds = b[(b.bank == bank) & (b.network_delay_ms == delay)]
        lines += [f"## {bank}, {delay:g} ms added network delay", "",
                  f"Single-service upper bounds certify {int(bounds.certified_infeasible.sum())}/40 cases infeasible.", "",
                  "| Required services | Proven optimum | Feasible at time limit | Bound proves impossible | Restricted family infeasible | Unresolved |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, g in group.groupby("service_group", sort=False):
            counts = g.status.value_counts()
            lines.append(f"| {name} | {counts.get('optimal_block_family', 0)} | {counts.get('feasible_solver_limit', 0)} | {counts.get('certified_infeasible_bounded_age_class', 0)} | {counts.get('infeasible_block_family', 0)} | {counts.get('unresolved_solver_limit', 0)} |")
        finite = group[group.transmission_reduction_vs_best_periodic.notna()]
        if len(finite):
            lines += ["", f"Across {len(finite)} groups with both an optimized witness and a qualifying periodic baseline, median transmission reduction is {100 * finite.transmission_reduction_vs_best_periodic.median():.1f}% (range {100 * finite.transmission_reduction_vs_best_periodic.min():.1f}--{100 * finite.transmission_reduction_vs_best_periodic.max():.1f}%)."]
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    plot_results(out, b, o)


def plot_results(out, bounds, optimum):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for row, bank in enumerate(["centered", "causal"]):
        for col, delay in enumerate([0, 200]):
            ax = axes[row, col]
            g = optimum[(optimum.bank == bank) & (optimum.network_delay_ms == delay)]
            labels = [s["service"] for s in base.SERVICES] + ["all local"]
            names = [s["service"] for s in base.SERVICES] + ["local_1s+local_5s+local_10s"]
            for x, name in enumerate(names):
                sub = g[g.service_group == name]
                finite = sub[sub.effective_rate_hz.notna()]
                if len(finite):
                    offsets = np.linspace(-.13, .13, len(finite))
                    ax.scatter(x + offsets, finite.effective_rate_hz, s=20, color="#0072B2", label="Optimized schedule" if x == 0 else None)
                    periodic = finite.best_periodic_transmissions / finite.postprefix_seconds
                    valid = periodic.notna()
                    ax.scatter(x + offsets[valid.to_numpy()], periodic[valid], s=25, marker="x", color="#D55E00", label="Best qualifying fixed rate" if x == 0 else None)
                ax.text(x, 10.7, f"{len(finite)}/10", ha="center", fontsize=8)
            ax.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, 11.3),
                   title=f"{bank.capitalize()} features, {delay} ms network delay")
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y", alpha=.2)
            ax.set_ylabel("Transmissions per second")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Communication needed for service qualification\nEach point is one recording; top counts show feasible witnesses", fontsize=12)
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(out / "minimum_communication.png", dpi=220)
    fig.savefig(out / "minimum_communication.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/delivery_feasibility")
    parser.add_argument("--banks", nargs="+", choices=list(BANKS), default=list(BANKS))
    parser.add_argument("--delays-ms", nargs="+", type=int, default=[0, 200])
    parser.add_argument("--block-seconds", type=float, default=10.)
    parser.add_argument("--max-age-seconds", type=float, default=1.5)
    parser.add_argument("--solver-seconds", type=float, default=20.)
    args = parser.parse_args()
    if args.block_seconds < 10 or args.block_seconds * 10 != round(args.block_seconds * 10):
        parser.error("Blocks must have integer 10-Hz length and cover the 10-s service")
    if any(delay < 0 or delay % 100 for delay in args.delays_ms):
        parser.error("Use nonnegative delays on the 100-ms evaluation grid")
    run(args)


if __name__ == "__main__":
    main()

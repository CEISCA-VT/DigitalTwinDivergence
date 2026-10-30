"""Refine time-limited delivery schedules and summarize physical-recording results."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from DigitalTwin.analysis import delivery_feasibility as study


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=study.Path, default=study.ROOT / "results/delivery_feasibility")
    parser.add_argument("--solver-seconds", type=float, default=60.)
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / "manifest.json").read_text())
    results = pd.read_csv(out / "optimization_results.csv")
    replays = pd.read_csv(out / "optimized_replay.csv")
    bounds = pd.read_csv(out / "feasibility_bounds.csv")
    refinements = []
    limited = results[results.solver_status.eq(1)]
    for bank, group in limited.groupby("bank"):
        recordings, _, _ = study.load_recordings(bank)
        for (seq, delay), cases in group.groupby(["sequence", "network_delay_ms"]):
            arrays = [a for e, a in sorted(recordings[seq], key=lambda z: z[0]["seed"])]
            lag = study.BANKS[bank][2] + math.ceil(delay / 100)
            block = int(round(manifest["block_seconds"] * 10))
            edges, counts, costs, denom = study.build_factors(arrays, lag, block)
            for index, old in cases.iterrows():
                services = tuple(old.service_group.split("+"))
                print(f"Refining {bank} {seq} {delay} ms {old.service_group}", flush=True)
                actions, info = study.solve_schedule(edges, counts, costs, denom, services, args.solver_seconds)
                refinements.append({"bank": bank, "sequence": seq, "delay_ms": int(delay),
                                    "service_group": old.service_group,
                                    "previous_transmissions": float(old.transmissions),
                                    **{k: v for k, v in info.items() if k != "selected_counts"}})
                if actions is None or info["transmissions"] > old.transmissions:
                    continue
                idx = study.schedule_indices(len(arrays[0]["time_s"]), actions, lag, block)
                metrics = study.summarize(arrays, idx)
                bound_rows = bounds[(bounds.bank == bank) & (bounds.sequence == seq)
                                    & (bounds.network_delay_ms == delay)].set_index("service")
                for si, m in enumerate(metrics):
                    if abs(m["joint_satisfaction"] - info["selected_counts"][si] / denom[si]) > 1e-12:
                        raise AssertionError("Refinement factor/replay disagreement")
                    if m["joint_satisfaction"] > bound_rows.loc[m["service"], "optimistic_joint_upper_bound"] + 1e-12:
                        raise AssertionError("Refinement violates feasibility bound")
                    if m["service"] in services and not m["qualified"]:
                        raise AssertionError("Refinement does not meet its contracts")
                for key, value in info.items():
                    if key != "selected_counts":
                        results.loc[index, key] = value
                results.loc[index, "status"] = "optimal_block_family" if info["solver_status"] == 0 else "feasible_solver_limit"
                results.loc[index, "effective_rate_hz"] = info["transmissions"] / old.postprefix_seconds
                if np.isfinite(old.best_periodic_transmissions):
                    results.loc[index, "transmission_reduction_vs_best_periodic"] = 1 - info["transmissions"] / old.best_periodic_transmissions
                mask = (replays.bank == bank) & (replays.sequence == seq) & (replays.network_delay_ms == delay) & (replays.service_group == old.service_group)
                for m in metrics:
                    selected = mask & replays.service.eq(m["service"])
                    for key, value in m.items():
                        replays.loc[selected, key] = value
                destination = out / "schedules" / bank / seq / f"{delay}ms_{old.service_group}.csv"
                pd.DataFrame({"receiver_block_start_s": 30 + np.arange(len(actions)) * manifest["block_seconds"],
                              "rate_hz": 10 / study.STRIDES[actions]}).to_csv(destination, index=False)
                results.to_csv(out / "optimization_results.csv", index=False)
                replays.to_csv(out / "optimized_replay.csv", index=False)
    manifest["refinement"] = {"solver_limit_seconds": args.solver_seconds,
                              "script_sha256": study.digest(__file__), "cases": refinements}
    manifest["checks"] = {"synthetic_unit_tests": "4 passed before full run",
                          "original_cells_reproduced": sum(r["reproduced_run_service_delivery_cells"] for r in manifest["reproduction"].values()),
                          "returned_schedule_metrics_checked_by_direct_replay": True,
                          "manuscript_unchanged": study.digest(study.ROOT / "manuscript/main.tex") == manifest["manuscript_sha256_before"]}
    if not manifest["checks"]["manuscript_unchanged"]:
        raise AssertionError("Manuscript hash changed")
    manifest["output_hashes"] = {name: study.digest(out / name) for name in [
        "feasibility_bounds.csv", "periodic_baselines.csv", "optimization_results.csv", "optimized_replay.csv"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    study.make_report(out)
    summary_rows = []
    rng = np.random.default_rng(20261030)
    for (bank, delay, services), group in results.groupby(["bank", "network_delay_ms", "service_group"]):
        valid = group[group.transmission_reduction_vs_best_periodic.notna()]
        values = valid.transmission_reduction_vs_best_periodic.to_numpy()
        row = dict(bank=bank, network_delay_ms=delay, service_group=services,
                   physical_recordings=len(group), feasible_schedules=int(group.transmissions.notna().sum()),
                   proved_minima=int(group.status.eq("optimal_block_family").sum()),
                   certified_infeasible=int(group.status.eq("certified_infeasible_bounded_age_class").sum()),
                   compared_recordings=len(valid))
        if len(values):
            draws = np.median(values[rng.integers(len(values), size=(5000, len(values)))], axis=1)
            row.update(median_transmission_reduction=float(np.median(values)),
                       median_reduction_ci_low=float(np.quantile(draws, .025)),
                       median_reduction_ci_high=float(np.quantile(draws, .975)),
                       median_effective_rate_hz=float(valid.effective_rate_hz.median()))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "recording_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

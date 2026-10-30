"""Read back the persisted experiments and verify schedules, costs and results."""
import argparse
import json
import platform

import numpy as np
import pandas as pd
import scipy

from DigitalTwin.analysis import delivery_feasibility as study


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=study.Path, default=study.ROOT / "results/delivery_feasibility")
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / "manifest.json").read_text())
    solutions = pd.read_csv(out / "optimization_results.csv")
    results = pd.read_csv(out / "optimized_replay.csv")
    bounds = pd.read_csv(out / "feasibility_bounds.csv")
    greedy = pd.read_csv(out / "greedy_comparison.csv")
    if len(solutions) != 240 or len(bounds) != 160:
        raise AssertionError("Incomplete experiment")
    if manifest["source_code_sha256"] != study.digest(study.__file__):
        raise AssertionError("Executed analysis code has changed")
    if manifest["manuscript_sha256_before"] != study.digest(study.ROOT / "manuscript/main.tex"):
        raise AssertionError("Manuscript hash has changed")
    checked, checked_greedy, checked_samples = 0, 0, 0
    max_metric_difference = 0.
    for bank in solutions.bank.unique():
        recordings, hashes, _ = study.load_recordings(bank)
        if any(manifest["inputs"].get(path) != sha for path, sha in hashes.items()):
            raise AssertionError("Recorded input hash mismatch")
        for r in solutions[solutions.bank.eq(bank)].itertuples():
            if not np.isfinite(r.transmissions):
                continue
            arrays = [a for e, a in sorted(recordings[r.sequence], key=lambda z: z[0]["seed"])]
            n = len(arrays[0]["time_s"])
            schedule = pd.read_csv(out / "schedules" / bank / r.sequence /
                                   f"{r.network_delay_ms}ms_{r.service_group}.csv")
            if not np.allclose(schedule.receiver_block_start_s, 30 + np.arange(len(schedule)) * 10):
                raise AssertionError("Unexpected block starts")
            actions = np.array([(10 / study.STRIDES).tolist().index(rate) for rate in schedule.rate_hz])
            lag = int(r.availability_lag_samples)
            idx = study.schedule_indices(n, actions, lag)
            if not np.all(np.diff(idx[300:]) >= 0) or not np.all(idx[300:] <= np.arange(300, n) - lag):
                raise AssertionError("Serialized schedule violates availability or timestamp order")
            if np.max(np.arange(300, n) - idx[300:]) > round(manifest["maximum_state_age_seconds"] * 10):
                raise AssertionError("Schedule exceeds the declared age cap")
            transmissions = int(np.count_nonzero(idx[300:] != idx[299:-1]))
            if transmissions != int(r.transmissions):
                raise AssertionError("Serialized schedule cost mismatch")
            if r.status == "optimal_block_family" and abs(r.transmissions - r.objective_lower_bound) > 1e-6:
                raise AssertionError("Unclosed optimizer gap labeled optimal")
            stored = results[(results.bank == bank) & (results.sequence == r.sequence)
                             & (results.network_delay_ms == r.network_delay_ms)
                             & (results.service_group == r.service_group)].set_index("service")
            for m in study.summarize(arrays, idx):
                for key in ["coverage", "physical_satisfaction", "freshness_satisfaction", "joint_satisfaction"]:
                    max_metric_difference = max(max_metric_difference, abs(m[key] - stored.loc[m["service"], key]))
                if m["service"] in r.service_group.split("+") and not m["qualified"]:
                    raise AssertionError("Serialized optimized schedule fails a required service")
            checked += 1
            grow = greedy[(greedy.bank == bank) & (greedy.sequence == r.sequence)
                          & (greedy.network_delay_ms == r.network_delay_ms)
                          & (greedy.service_group == r.service_group)]
            if len(grow) == 1:
                schedule = pd.read_csv(out / "greedy_schedules" / bank / r.sequence /
                                       f"{r.network_delay_ms}ms_{r.service_group}.csv")
                actions = np.array([(10 / study.STRIDES).tolist().index(rate) for rate in schedule.rate_hz])
                idx = study.schedule_indices(n, actions, lag)
                tx = int(np.count_nonzero(idx[300:] != idx[299:-1]))
                if tx != int(grow.greedy_transmissions.iloc[0]):
                    raise AssertionError("Serialized greedy cost mismatch")
                for m in study.summarize(arrays, idx):
                    if m["service"] in r.service_group.split("+") and not m["qualified"]:
                        raise AssertionError("Serialized greedy schedule fails a required service")
                checked_greedy += 1
        for r in bounds[bounds.bank.eq(bank)].itertuples():
            samples = np.load(out / "sample_bounds" / bank / r.sequence /
                              f"{r.network_delay_ms}ms_{r.service}.npz")["maximum_passing_seeds"]
            if not np.isin(samples, [0, 1, 2, 3]).all():
                raise AssertionError("Invalid seed counts in feasibility bound")
            if abs(samples.mean() / 3 - r.optimistic_joint_upper_bound) > 1e-12:
                raise AssertionError("Serialized sample bound disagrees with reported mean")
            checked_samples += 1
    if max_metric_difference > 1e-12:
        raise AssertionError("Serialized replay metrics disagree")
    code = ["DigitalTwin/analysis/delivery_feasibility.py", "DigitalTwin/analysis/service_timing_budget_study.py",
            "scripts/refine_delivery_feasibility.py", "scripts/compare_delivery_greedy.py",
            "scripts/plot_delivery_feasibility.py", "scripts/audit_delivery_feasibility.py",
            "tests/test_delivery_feasibility.py", "docs/delivery_feasibility_protocol.md"]
    audit = {"status": "PASS", "python": platform.python_version(), "numpy": np.__version__,
             "pandas": pd.__version__, "scipy": scipy.__version__, "original_cells_reproduced": 4800,
             "optimized_schedules_read_back": checked, "greedy_schedules_read_back": checked_greedy,
             "sample_bounds_read_back": checked_samples, "maximum_replay_metric_difference": max_metric_difference,
             "proved_minima_in_block_family": int(solutions.status.eq("optimal_block_family").sum()),
             "remaining_optimality_gaps": solutions.loc[solutions.status.eq("feasible_solver_limit"), "mip_gap"].tolist(),
             "manuscript_unchanged": True,
             "code_hashes": {path: study.digest(study.ROOT / path) for path in code},
             "output_hashes": {str(path.relative_to(out)): study.digest(path) for path in out.rglob("*")
                               if path.is_file() and path.name != "verification.json"}}
    (out / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k not in ["code_hashes", "output_hashes"]}, indent=2))


if __name__ == "__main__":
    main()

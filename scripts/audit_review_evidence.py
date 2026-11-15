"""Audit saved manuscript evidence without refitting predictors or source models."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from DigitalTwin.analysis import delivery_feasibility as planning

ROOT = Path(__file__).resolve().parents[1]
BANKS = {
    "centered": ROOT / "results/doubly_nested_loso_evidence",
    "past_only": ROOT / "results/doubly_nested_causal_loso_slurm/service_evidence",
}
KEY = ["sequence", "service", "rate_hz", "delay_ms"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_planner():
    from tests.test_delivery_feasibility import recordings

    checks = []
    for blocks, lag in [(3, 0), (4, 1), (5, 2), (6, 3)]:
        n = 300 + (blocks - 1) * 100 + 37
        arrays = recordings(n)
        edges, counts, costs, denominators = planning.build_factors(arrays, lag)
        lookup = {edge: i for i, edge in enumerate(edges)}
        upper = {s["service"]: planning.optimistic_bound(arrays, s, lag)[0]
                 for s in planning.base.SERVICES}
        minima = {tuple(group): np.inf for group in planning.GROUPS}
        for actions in itertools.product(range(4), repeat=blocks):
            idx = planning.schedule_indices(n, actions, lag)
            chosen = [lookup[(b, actions[b - 1] if b else 0, actions[b])]
                      for b in range(blocks)]
            direct_cost = int(np.count_nonzero(idx[300:] != idx[299:-1]))
            if direct_cost != int(costs[chosen].sum()):
                raise AssertionError("Partial-block or boundary cost mismatch")
            metrics = planning.summarize(arrays, idx)
            for i, row in enumerate(metrics):
                expected = counts[chosen, i].sum() / denominators[i]
                np.testing.assert_allclose(row["joint_satisfaction"], expected,
                                           atol=1e-12, rtol=0)
                if row["joint_satisfaction"] > upper[row["service"]] + 1e-12:
                    raise AssertionError("Replay exceeds optimistic bound")
            passed = {m["service"] for m in metrics if m["qualified"]}
            for group in minima:
                if set(group).issubset(passed):
                    minima[group] = min(minima[group], direct_cost)
        for group, minimum in minima.items():
            actions, info = planning.solve_schedule(edges, counts, costs,
                                                     denominators, group, 20)
            if np.isfinite(minimum):
                if actions is None or info["solver_status"] != 0:
                    raise AssertionError("Exhaustively feasible case was not solved")
                if info["transmissions"] != minimum:
                    raise AssertionError("MILP differs from exhaustive minimum")
            elif actions is not None or info["solver_status"] != 2:
                raise AssertionError("Exhaustive infeasibility disagrees with MILP")
        checks.append(dict(blocks=blocks, lag_samples=lag, final_block_samples=37,
                           schedules_checked=4 ** blocks, groups_checked=len(minima)))
        print(f"Planner: {blocks} blocks, lag {lag}, {4 ** blocks} schedules PASS",
              flush=True)
    return checks


def query_phase_audit(output, inputs):
    rows = []
    for label, bank in [("centered", "centered"), ("past_only", "causal")]:
        recordings, hashes, _ = planning.load_recordings(bank)
        inputs.update(hashes)
        for sequence, entries in sorted(recordings.items()):
            for phase_ms in [0, 25, 50, 75]:
                for condition, rate, delay_ms in [("degraded", 2, 200),
                                                   ("practical", 5, 50),
                                                   ("ideal", 10, 0)]:
                    for service in planning.base.SERVICES:
                        h = int(round(service["horizon_s"] * 10))
                        scores = []
                        for _, a in entries:
                            clock = np.rint(a["time_s"] * 1e9).astype(np.int64)
                            query = np.arange(300 + h, len(clock)) * 100_000_000
                            query = query + phase_ms * 1_000_000
                            query = query[query <= clock[-1]]
                            source = np.arange(0, len(clock), 10 // rate)
                            arrivals = clock[source] + delay_ms * 1_000_000
                            current = source[np.searchsorted(arrivals, query, side="right") - 1]
                            gt_x = np.interp(query, clock, a["gt_east_m"])
                            gt_y = np.interp(query, clock, a["gt_north_m"])
                            gt_heading = np.interp(query, clock, np.unwrap(a["gt_heading_rad"]))
                            if h:
                                previous_query = query - h * 100_000_000
                                previous = source[np.searchsorted(arrivals, previous_query,
                                                                  side="right") - 1]
                                old_heading = np.interp(previous_query, clock,
                                                        np.unwrap(a["gt_heading_rad"]))
                                dx = gt_x - np.interp(previous_query, clock, a["gt_east_m"])
                                dy = gt_y - np.interp(previous_query, clock, a["gt_north_m"])
                                gx = np.cos(old_heading) * dx + np.sin(old_heading) * dy
                                gy = -np.sin(old_heading) * dx + np.cos(old_heading) * dy
                                ex, ey, eh = planning.base.relative(
                                    a["estimate_east_m"], a["estimate_north_m"],
                                    a["estimate_heading_rad"], previous, current)
                                position = np.hypot(ex - gx, ey - gy)
                                heading = np.abs(np.rad2deg(planning.base.wrap(
                                    eh - planning.base.wrap(gt_heading - old_heading))))
                            else:
                                position = np.hypot(a["estimate_east_m"][current] - gt_x,
                                                    a["estimate_north_m"][current] - gt_y)
                                heading = np.abs(np.rad2deg(planning.base.wrap(
                                    a["estimate_heading_rad"][current] - gt_heading)))
                            physical = ((position <= service["pos_tol_m"])
                                        & (heading <= service["heading_tol_deg"]))
                            age_limit = round(service["aoi_limit_s"] * 1e9)
                            fresh = query - clock[current] <= age_limit
                            scores.append((physical.mean(), fresh.mean(),
                                           (physical & fresh).mean()))
                        physical, fresh, joint = np.mean(scores, axis=0)
                        # Exact event integration over the same continuous scored span.
                        start = (300 + h) * 100_000_000
                        end = clock[-1]
                        left = np.maximum(arrivals, start)
                        right = np.minimum(np.r_[arrivals[1:], end], end)
                        fresh_right = np.minimum(right, clock[source] + age_limit)
                        continuous_freshness = np.maximum(fresh_right - left, 0).sum() / (end - start)
                        rows.append(dict(bank=label, sequence=sequence,
                                         phase_ms=phase_ms, condition=condition,
                                         service=service["service"], rate_hz=rate,
                                         delay_ms=delay_ms, physical_satisfaction=physical,
                                         freshness_satisfaction=fresh, joint_satisfaction=joint,
                                         time_weighted_freshness=continuous_freshness,
                                         qualified=int(joint >= .8)))
        print(f"Query phases: {label} PASS", flush=True)
    data = pd.DataFrame(rows)
    summaries = []
    for (bank, phase), group in data.groupby(["bank", "phase_ms"]):
        cases = group.pivot(index=["sequence", "service"], columns="condition", values="qualified")
        classification = np.where(cases.degraded.eq(1), "already",
                                  np.where(cases.ideal.eq(1), "remediable", "persistent"))
        reference = pd.read_csv(BANKS[bank] / "sequence_service_case_ledger.csv")
        reference = reference.set_index(["sequence", "service"]).reindex(cases.index)
        names = {"already_qualified_degraded": "already",
                 "delivery_remediable": "remediable", "persistent_under_ideal": "persistent"}
        if phase == 0:
            for condition in ["degraded", "practical", "ideal"]:
                measured = group[group.condition.eq(condition)].set_index(["sequence", "service"])
                np.testing.assert_allclose(
                    measured.joint_satisfaction.sort_index(),
                    reference[f"{condition}_joint_satisfaction"].sort_index(), atol=1e-12, rtol=0)
        summaries.append(dict(bank=bank, phase_ms=phase,
                              remediable=int((classification == "remediable").sum()),
                              persistent=int((classification == "persistent").sum()),
                              already=int((classification == "already").sum()),
                              changed_from_aligned=int(np.count_nonzero(
                                  classification != reference.classification.map(names))),
                              practical_restores=int(((classification == "remediable")
                                                      & cases.practical.eq(1)).sum())))
    data.to_csv(output / "query_phase_cases.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "query_phase_summary.csv", index=False)
    return summaries


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/reviewer_audit_20261106")
    parser.add_argument("--validate-planner", action="store_true")
    parser.add_argument("--query-phase", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs, patterns, seed_rows, screening, failures, exposure = {}, [], [], [], [], []
    summary = {}
    for bank, evidence in BANKS.items():
        folder = "causal" if bank == "past_only" else "centered"
        structured = ROOT / "results/structured_qualification_baselines" / folder
        paths = [evidence / "per_run_response_surface.csv",
                 evidence / "per_sequence_response_surface.csv",
                 evidence / "sequence_service_case_ledger.csv",
                 structured / "predictions.csv",
                 structured / "per_sequence_operating_points.csv"]
        for path in paths:
            inputs[str(path.relative_to(ROOT))] = digest(path)
        runs, seq, ledger, predictions, operating = map(pd.read_csv, paths)
        runs = runs[runs.outer_sequence.eq(runs.sequence)].copy()
        seq = seq[seq.outer_sequence.eq(seq.sequence)].copy()
        if len(runs) != 2400 or len(seq) != 800 or len(ledger) != 40:
            raise AssertionError("Incomplete outer-test evidence")
        if runs.groupby(KEY).size().ne(3).any():
            raise AssertionError("Expected three seeds per cell")
        seeds = runs.groupby(KEY).agg(
            joint_mean=("joint_satisfaction", "mean"),
            joint_min=("joint_satisfaction", "min"),
            joint_max=("joint_satisfaction", "max"),
            coverage_min=("coverage", "min"),
            all_seeds_qualified=("qualified", "min"),
        ).reset_index().merge(seq[KEY + ["qualified", "joint_satisfaction"]],
                              on=KEY, validate="one_to_one")
        np.testing.assert_allclose(seeds.joint_mean, seeds.joint_satisfaction,
                                   atol=1e-12, rtol=0)
        seeds["bank"] = bank
        seeds["decision_changed"] = seeds.qualified.ne(seeds.all_seeds_qualified)
        seed_rows.append(seeds)
        pattern = (ledger.degraded_qualified.astype(str)
                   + ledger.practical_qualified.astype(str)
                   + ledger.ideal_qualified.astype(str))
        counts = pattern.value_counts()
        for value in map("".join, itertools.product("01", repeat=3)):
            patterns.append(dict(bank=bank, degraded_practical_ideal=value,
                                 cases=int(counts.get(value, 0))))
        selected = operating[np.isclose(operating.target_acceptance, .25)
                             & operating.scope.eq("pooled")]
        outer = predictions[predictions.prediction_role.eq("outer_test")].merge(
            selected[["heldout_sequence", "method", "threshold", "accepted_count", "false_count"]],
            left_on=["outer_sequence", "method"],
            right_on=["heldout_sequence", "method"], validate="many_to_one")
        outer["accepted"] = outer.score.ge(outer.threshold)
        outer["false_qualified"] = outer.accepted & outer.actual_qualified.eq(0)
        observed = outer.groupby(["outer_sequence", "method"])[
            ["accepted", "false_qualified"]].sum()
        expected = selected.set_index(["heldout_sequence", "method"]).sort_index()
        np.testing.assert_array_equal(observed.accepted, expected.accepted_count)
        np.testing.assert_array_equal(observed.false_qualified, expected.false_count)
        outer["bank"] = bank
        screening.append(outer)
        failed = outer[outer.false_qualified].merge(
            seq[KEY + ["joint_satisfaction"]], on=KEY, validate="many_to_one")
        failed["joint_shortfall"] = .8 - failed.joint_satisfaction
        failures.append(failed)
        for sequence, group in runs.groupby("sequence"):
            row = group[group.service.eq("global")].iloc[0]
            exposure.append(dict(bank=bank, sequence=sequence,
                                 duration_s=(int(row.eligible_samples) + 299) / 10,
                                 global_scored_samples=int(row.eligible_samples),
                                 minimum_observability=float(group.coverage.min())))
        ledger_seeds = seeds[(seeds.rate_hz.eq(2) & seeds.delay_ms.eq(200))
                            | (seeds.rate_hz.eq(5) & seeds.delay_ms.eq(50))
                            | (seeds.rate_hz.eq(10) & seeds.delay_ms.eq(0))]
        summary[bank] = dict(grid_seed_rule_changes=int(seeds.decision_changed.sum()),
                             ledger_point_seed_rule_changes=int(ledger_seeds.decision_changed.sum()),
                             ledger_pairs_with_seed_rule_changes=int(
                                 ledger_seeds[ledger_seeds.decision_changed][["sequence", "service"]]
                                 .drop_duplicates().shape[0]),
                             practical_pass_ideal_fail=int(((ledger.practical_qualified == 1)
                                                           & (ledger.ideal_qualified == 0)).sum()),
                             degraded_pass_ideal_fail=int(((ledger.degraded_qualified == 1)
                                                          & (ledger.ideal_qualified == 0)).sum()))
    plotted = pd.concat(screening, ignore_index=True)
    per_service = plotted.groupby(["bank", "method", "service"]).agg(
        accepted=("accepted", "sum"), false_qualified=("false_qualified", "sum"),
        evaluated=("accepted", "size")).reset_index()
    per_recording = plotted.groupby(["bank", "method", "sequence"]).agg(
        accepted=("accepted", "sum"), false_qualified=("false_qualified", "sum"),
        evaluated=("accepted", "size")).reset_index()
    for name, table in [("per_service", per_service), ("per_recording", per_recording),
                        ("ledger_pass_patterns", pd.DataFrame(patterns)),
                        ("seed_sensitivity", pd.concat(seed_rows, ignore_index=True)),
                        ("accepted_failures", pd.concat(failures, ignore_index=True)),
                        ("sequence_exposure", pd.DataFrame(exposure))]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    checks = validate_planner() if args.validate_planner else []
    phases = query_phase_audit(args.output, inputs) if args.query_phase else []
    audit = dict(inputs=inputs, summary=summary, planner_checks=checks,
                 query_phase_checks=phases,
                 source_sha256=digest(Path(__file__)),
                 verification_code_hashes={name: digest(ROOT / name) for name in [
                     "DigitalTwin/analysis/delivery_feasibility.py",
                     "DigitalTwin/analysis/service_timing_budget_study.py",
                     "tests/test_delivery_feasibility.py"]},
                 interpretation="Saved frozen-cutoff outcomes; no refitting or trajectory changes")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    close = ["service_empirical", "service_specific_logistic", "service_specific_tree"]
    print(per_service[per_service.method.isin(close)].to_string(index=False))
    print(per_recording[per_recording.method.eq("service_empirical")].to_string(index=False))
    print(pd.DataFrame(patterns).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

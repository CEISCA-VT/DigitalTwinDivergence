"""Compare exact block scheduling with reference-informed greedy rate reduction."""
import argparse
import math

import numpy as np
import pandas as pd

from DigitalTwin.analysis import delivery_feasibility as study


def greedy(edges, counts, costs, denom, group, start):
    blocks = max(b for b, _, _ in edges) + 1
    lookup = {edge: e for e, edge in enumerate(edges)}
    selected = [i for i, s in enumerate(study.base.SERVICES) if s["service"] in group]
    required = np.ceil(study.base.TARGET * denom[selected] - 1e-9).astype(int)
    actions = np.full(blocks, start, dtype=int)
    path = [lookup[(b, start if b else 0, start)] for b in range(blocks)]
    totals = counts[path].sum(axis=0)
    if np.any(totals[selected] < required):
        return None
    while True:
        candidates = []
        for b in np.flatnonzero(actions < len(study.STRIDES) - 1):
            new_action = actions[b] + 1
            old_edges = [lookup[(b, actions[b - 1] if b else 0, actions[b])]]
            new_edges = [lookup[(b, actions[b - 1] if b else 0, new_action)]]
            if b + 1 < blocks:
                old_edges.append(lookup[(b + 1, actions[b], actions[b + 1])])
                new_edges.append(lookup[(b + 1, new_action, actions[b + 1])])
            delta = counts[new_edges].sum(axis=0) - counts[old_edges].sum(axis=0)
            if np.any((totals + delta)[selected] < required):
                continue
            saved = costs[old_edges].sum() - costs[new_edges].sum()
            if saved <= 0:
                continue
            loss = np.maximum(-delta[selected] / denom[selected], 0).sum()
            candidates.append((loss / saved, -saved, int(b), delta))
        if not candidates:
            break
        _, _, b, delta = min(candidates, key=lambda item: item[:3])
        actions[b] += 1
        totals += delta
    path = [lookup[(b, actions[b - 1] if b else 0, actions[b])] for b in range(blocks)]
    return actions, int(costs[path].sum()), totals


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=study.Path, default=study.ROOT / "results/delivery_feasibility")
    args = parser.parse_args()
    out = args.output
    solutions = pd.read_csv(out / "optimization_results.csv")
    rows = []
    for bank in solutions.bank.unique():
        recordings, _, _ = study.load_recordings(bank)
        for (seq, delay), groups in solutions[solutions.bank.eq(bank)].groupby(["sequence", "network_delay_ms"]):
            arrays = [a for e, a in sorted(recordings[seq], key=lambda z: z[0]["seed"])]
            lag = study.BANKS[bank][2] + math.ceil(delay / 100)
            edges, counts, costs, denom = study.build_factors(arrays, lag)
            for r in groups.itertuples():
                if not np.isfinite(r.transmissions):
                    continue
                group = r.service_group.split("+")
                starts = [greedy(edges, counts, costs, denom, group, a) for a in range(4)]
                starts = [s for s in starts if s is not None]
                if not starts:
                    continue
                actions, tx, totals = min(starts, key=lambda s: s[1])
                idx = study.schedule_indices(len(arrays[0]["time_s"]), actions, lag)
                for si, m in enumerate(study.summarize(arrays, idx)):
                    if abs(m["joint_satisfaction"] - totals[si] / denom[si]) > 1e-12:
                        raise AssertionError("Greedy factor/replay disagreement")
                    if m["service"] in group and not m["qualified"]:
                        raise AssertionError("Greedy schedule fails its required contract")
                if tx < r.objective_lower_bound - 1e-6:
                    raise AssertionError("Greedy solution contradicts optimizer lower bound")
                dest = out / "greedy_schedules" / bank / seq
                dest.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"receiver_block_start_s": 30 + np.arange(len(actions)) * 10,
                              "rate_hz": 10 / study.STRIDES[actions]}).to_csv(dest / f"{delay}ms_{r.service_group}.csv", index=False)
                rows.append(dict(bank=bank, sequence=seq, network_delay_ms=delay,
                                 service_group=r.service_group, greedy_transmissions=tx,
                                 optimizer_transmissions=r.transmissions,
                                 optimizer_lower_bound=r.objective_lower_bound,
                                 optimizer_proved_optimal=r.status == "optimal_block_family",
                                 best_periodic_transmissions=r.best_periodic_transmissions,
                                 reduction_vs_greedy=1 - r.transmissions / tx))
            pd.DataFrame(rows).to_csv(out / "greedy_comparison.csv", index=False)
            print(f"Greedy replay checked: {bank} {seq} {delay} ms", flush=True)
    table = pd.DataFrame(rows)
    summary = table.groupby(["bank", "network_delay_ms", "service_group"], as_index=False).agg(
        recordings=("sequence", "size"), median_reduction_vs_greedy=("reduction_vs_greedy", "median"),
        max_reduction_vs_greedy=("reduction_vs_greedy", "max"))
    summary.to_csv(out / "greedy_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

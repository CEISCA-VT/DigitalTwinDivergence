"""Exact sequence-ranking audit for the E1 local/global inversion counts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "e1_e2_service_contract_publication" / "E1_i2nav" / "e1_pairwise_service_ordering_summary.csv"
OUTPUT = ROOT / "results" / "e1_rank_inference"


def mahonian_counts(n: int) -> list[int]:
    counts = [1]
    for size in range(2, n + 1):
        updated = [0] * (len(counts) + size - 1)
        for inversions, count in enumerate(counts):
            for added in range(size):
                updated[inversions + added] += count
        counts = updated
    return counts


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    counts = mahonian_counts(10)
    total = math.factorial(10)
    maximum = 45
    rows = []
    for source in source_rows:
        observed = int(source["inversions"])
        lower = sum(counts[: observed + 1]) / total
        upper = sum(counts[maximum - observed :]) / total
        rows.append({"horizon_s": source["local_horizon_s"], "physical_sequences": 10,
                     "pairwise_orderings": maximum, "observed_inversions": observed,
                     "kendall_tau_from_inversions": 1 - 2 * observed / maximum,
                     "exact_random_ranking_two_sided_p": min(1.0, 2 * min(lower, upper)),
                     "independent_unit": "physical sequence",
                     "interpretation": "descriptive existence/extent; pairwise orderings are dependent"})
    with (OUTPUT / "exact_rank_inference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {"schema": "e1_exact_rank_inference_v1", "null": "two independent random rankings of ten sequences",
                "method": "exact Mahonian inversion-count distribution", "independent_units": 10,
                "warning": "The 45 pairwise comparisons share sequences and are not independent replicates.",
                "block_bootstrap_scope": "Within-sequence block resampling may quantify temporal estimator uncertainty but cannot increase the physical-sequence sample size."}
    (OUTPUT / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# E1 Exact Sequence-Ranking Audit", "", "The independent experimental unit remains the physical sequence (n=10). The 45 sequence pairs are dependent.", "",
             "| Horizon | Inversions | Kendall tau | Exact random-ranking p |", "|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['horizon_s']} s | {row['observed_inversions']}/45 | {row['kendall_tau_from_inversions']:.3f} | {row['exact_random_ranking_two_sided_p']:.4f} |")
    lines += ["", "These tests do not support a preassigned p<0.001 claim. E1 is an existence and threshold-robust demonstration of service-dependent ordering, not a population prevalence estimate.",
              "Block bootstrap must not be used to relabel hundreds of windows as independent physical experiments."]
    (OUTPUT / "rank_inference_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()

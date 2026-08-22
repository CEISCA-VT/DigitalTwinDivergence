#!/usr/bin/env python3
"""Assemble generated pre-GPS analyses into one review report."""
from pathlib import Path
import pandas as pd

ROOT=Path("results/publication_hardening")
ROOT.mkdir(parents=True,exist_ok=True)

def table(name, cols=None, n=20):
    p=ROOT/name
    if not p.exists(): return f"_Missing: `{p}`_\n"
    d=pd.read_csv(p)
    if cols: d=d[[c for c in cols if c in d.columns]]
    return d.head(n).to_markdown(index=False)

parts=["# Pre-GPS publication hardening report","",
"## 1. Sequence-level V1 vs V2 statistics","",
table("publication_sequence_statistics.csv",
      ["metric","n_sequences","v1_median","v2_median",
       "mean_difference_ci_low","mean_difference_ci_high",
       "wilcoxon_two_sided_p","matched_pairs_rank_biserial_v2_minus_v1","v2_improved_sequences"]),
"",
"## 2. TFP vs Muñoz uncertainty (representative MAD values)","",
table("munoz_tfp_bootstrap_correlations_midMAD.csv",
      ["method","domain","mad","metric_a","metric_b","n_sequences","spearman","bootstrap_ci_low","bootstrap_ci_high"]),
"",
"## 3. Controlled yaw-bias replay","",
table("yaw_bias_macro_by_magnitude.csv"),
"",
"## 4. Claim audit","",
table("claim_audit.csv",["status","claim","expected_value","note"]),
"",
"## 5. Interpretation guardrails","",
"- Use physical sequence as the statistical unit.",
"- Treat TFP and Muñoz as complementary evaluators, not universally ordered.",
"- Do not claim cross-domain validation from a domain-general mathematical formulation alone.",
"- Treat yaw-bias injection as controlled replay sensitivity, not attack detection.",
"- Do not publish yaw-bias results unless `yaw_bias_replay_status.txt` says PASS.",
"- Keep operational fidelity (unaligned) separate from official aligned benchmark accuracy.",
"- Resolve the 1.635 m versus exact-evo 1.251 m V2 APE aggregation/provenance discrepancy before final freeze.",
]
(ROOT/"PRE_GPS_HARDENING_REPORT.md").write_text("\n".join(parts),encoding="utf-8")
print(ROOT/"PRE_GPS_HARDENING_REPORT.md")

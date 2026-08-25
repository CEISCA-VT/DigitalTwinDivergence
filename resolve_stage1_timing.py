#!/usr/bin/env python3
"""
Resolve Stage-1 timing protocol documentation from the already-completed
i2Nav timing experiment.

Run from repository root:
    python resolve_stage1_timing.py

This does NOT rerun the timing experiment. It:
1. verifies the protocol from DigitalTwin/analysis/i2nav_timing_sensitivity.py,
2. reads results/i2nav_timing_sensitivity/timing_sensitivity_paired_statistics.csv,
3. regenerates the Stage-1 timing publication tables,
4. writes a resolved protocol audit,
5. updates the Stage-1 manifest/report only after all verification gates pass.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REQUIRED_STATS_COLS = [
    "perturbation",
    "value_ms",
    "metric",
    "n_sequences",
    "mean_paired_difference",
    "paired_ci95_low",
    "paired_ci95_high",
    "relative_change_pct_vs_0ms",
]

HEADLINE_METRICS = [
    "ate_m",
    "heading_mae_deg",
    "rpe1_m",
    "rpe5_m",
    "rpe10_m",
]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def source_fact(pattern: str, text: str, label: str) -> None:
    require(re.search(pattern, text, flags=re.I | re.S) is not None,
            f"Protocol verification failed: {label}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    args = ap.parse_args()

    repo = args.repo.resolve()
    src = repo / "DigitalTwin" / "analysis" / "i2nav_timing_sensitivity.py"
    stats = repo / "results" / "i2nav_timing_sensitivity" / "timing_sensitivity_paired_statistics.csv"
    stage1 = repo / "results" / "stage1_publication_hardening"

    require(src.exists(), f"Missing timing source: {src}")
    require(stats.exists(), f"Missing paired statistics: {stats}")
    stage1.mkdir(parents=True, exist_ok=True)

    text = src.read_text(encoding="utf-8", errors="replace")

    # Protocol gates: verify facts from code, not memory.
    source_fact(r'--delay-ms".*?default="0,25,50,100,200"', text,
                "fixed delay levels 0,25,50,100,200 ms")
    source_fact(r'--clock-offset-ms".*?default="-100,-50,-25,0,25,50,100"', text,
                "clock offset levels")
    source_fact(r'--jitter-ms".*?default="0,10,25,50"', text,
                "jitter sigma levels 0,10,25,50 ms")
    source_fact(r'--jitter-seeds".*?default="0,1,2,3,4"', text,
                "jitter user seeds 0..4")
    source_fact(r'--bootstrap".*?default=20000', text,
                "paired bootstrap count 20000")
    source_fact(r'--bootstrap-seed".*?default=42', text,
                "paired bootstrap seed 42")
    source_fact(r'def\s+jitter_variant.*?default_rng\(seed\).*?normal\(0\.0,\s*sigma_s',
                text, "zero-mean Gaussian jitter")
    source_fact(r'file_term\s*=.*?sum\(path\.as_posix\(\)\.encode\("utf-8"\)\).*?% 100000',
                text, "stable per-file seed term")
    source_fact(r'eff_seed\s*=\s*int\(js\s*\+\s*100003\s*\*\s*file_term\)',
                text, "effective per-file jitter seed")
    source_fact(r'def\s+fixed_delay_variant.*?arrival\s*=\s*t\s*\+\s*delay_s.*?causal_hold',
                text, "fixed-delay causal zero-order hold")
    source_fact(r'def\s+jitter_variant.*?stamped\s*=\s*t\s*\+.*?normal',
                text, "timestamp jitter applied to virtual-state timestamps")
    source_fact(r'not a test of how delayed raw imu/odometry inputs alter',
                text.lower(), "raw-input claim boundary")
    source_fact(
    r'run_group\s*=\s*\[\s*"file"\s*,\s*"sequence"\s*,\s*"seed"\s*,'
    r'\s*"perturbation"\s*,\s*"value_ms"\s*\].*?'
    r'groupby\(\s*\[\s*"sequence"\s*,\s*"perturbation"\s*,\s*"value_ms"',
    text,
    "statistical hierarchy"
)

    d = pd.read_csv(stats)
    missing = [c for c in REQUIRED_STATS_COLS if c not in d.columns]
    require(not missing, f"Paired statistics missing columns: {missing}")

    d["value_ms"] = pd.to_numeric(d["value_ms"], errors="coerce")
    d["n_sequences"] = pd.to_numeric(d["n_sequences"], errors="coerce")
    for c in [
        "mean_paired_difference",
        "paired_ci95_low",
        "paired_ci95_high",
        "relative_change_pct_vs_0ms",
    ]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    require(d["perturbation"].isin(
        ["fixed_delay", "clock_offset", "timestamp_jitter"]).any(),
        "No expected perturbation rows found.")

    # Verify all headline nonzero conditions are sequence-level n=10.
    nonzero = d[
        d["perturbation"].isin(["fixed_delay", "timestamp_jitter"])
        & (d["value_ms"] > 0)
        & d["metric"].isin(HEADLINE_METRICS)
    ]
    require(len(nonzero) > 0, "No headline timing rows found.")
    require((nonzero["n_sequences"] == 10).all(),
            "Expected n_sequences=10 for all headline timing comparisons.")

    # Full long-form publication table.
    pub = d[
        d["metric"].isin(HEADLINE_METRICS)
    ].copy()
    pub = pub.sort_values(["perturbation", "value_ms", "metric"])
    pub.to_csv(stage1 / "1D_timing_publication_table.csv", index=False)

    # Compact headline table: the conditions emphasized in the manuscript.
    headline_conditions = [
        ("timestamp_jitter", 50.0),
        ("fixed_delay", 100.0),
        ("fixed_delay", 200.0),
    ]
    hparts = []
    for kind, value in headline_conditions:
        g = d[
            (d["perturbation"] == kind)
            & (d["value_ms"] == value)
            & d["metric"].isin(HEADLINE_METRICS)
        ].copy()
        require(len(g) == len(HEADLINE_METRICS),
                f"Missing headline rows for {kind} {value} ms")
        hparts.append(g)
    headline = pd.concat(hparts, ignore_index=True)
    headline.to_csv(stage1 / "1D_timing_headline_table.csv", index=False)

    # Wide table convenient for manuscript entry.
    wide = headline.pivot(
        index=["perturbation", "value_ms"],
        columns="metric",
        values="relative_change_pct_vs_0ms",
    ).reset_index()
    wide.to_csv(stage1 / "1D_timing_headline_relative_changes_wide.csv", index=False)

    audit_lines = [
        "# Timing sensitivity publication audit - RESOLVED",
        "",
        "## Protocol verified from source code",
        "",
        "- Analysis type: controlled physical-virtual synchronization/timestamp replay on saved frozen V2 trajectories.",
        "- Fixed update delay: 0, 25, 50, 100, 200 ms.",
        "- Fixed delay mechanism: virtual-state delivery is delayed and evaluated with causal zero-order hold.",
        "- Clock/timestamp offset: -100, -50, -25, 0, 25, 50, 100 ms.",
        "- Timestamp jitter sigma: 0, 10, 25, 50 ms.",
        "- Jitter distribution: zero-mean Gaussian, N(0, sigma^2).",
        "- Jitter user seeds for nonzero sigma: 0, 1, 2, 3, 4.",
        "- Per-file effective jitter seed: js + 100003 * (sum(path UTF-8 bytes) mod 100000).",
        "- Paired-confidence bootstrap: 20,000 resamples, seed 42.",
        "- Statistical unit: physical sequence. Jitter replicates are averaged within each original run; original run/checkpoint results are averaged within sequence before paired inference.",
        "- Number of physical sequences in the paired headline statistics: 10.",
        "",
        "## Exact claim boundary",
        "",
        "The experiment perturbs the timing relationship between the saved physical trajectory and the already-generated frozen Twin V2 virtual-state stream. It does not delay raw IMU/odometry inputs and does not re-run or alter the neural correction model. Fixed delay therefore characterizes virtual-state delivery/synchronization sensitivity, while timestamp jitter characterizes sensitivity to stochastic virtual-state timestamp displacement.",
        "",
        "The underlying saved trajectory streams are frozen. Interpolation/causal hold changes which already-generated virtual state is associated with a physical query time; it does not regenerate the twin trajectory.",
        "",
        "## Publication status",
        "",
        "**PASS / COMPLETE. No timing-protocol provenance fields remain unresolved.**",
        "",
        "Use `1D_timing_publication_table.csv` for the complete long-form statistics and",
        "`1D_timing_headline_table.csv` for the three headline perturbation conditions.",
        "",
    ]
    (stage1 / "1D_timing_protocol_audit.md").write_text(
        "\n".join(audit_lines), encoding="utf-8"
    )

    protocol = {
        "analysis_type": "controlled synchronization/timestamp replay",
        "fixed_delay_ms": [0, 25, 50, 100, 200],
        "clock_offset_ms": [-100, -50, -25, 0, 25, 50, 100],
        "jitter_sigma_ms": [0, 10, 25, 50],
        "jitter_distribution": "Gaussian N(0, sigma^2)",
        "jitter_user_seeds_nonzero": [0, 1, 2, 3, 4],
        "effective_jitter_seed":
            "js + 100003 * (sum(path.as_posix().encode('utf-8')) % 100000)",
        "paired_bootstrap_replicates": 20000,
        "paired_bootstrap_seed": 42,
        "n_physical_sequences": 10,
        "resampling_unit": "physical sequence",
        "raw_imu_odometry_delayed": False,
        "twin_retrained": False,
    }
    (stage1 / "1D_timing_protocol_resolved.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    # Update Stage-1 manifest if present.
    manifest_path = stage1 / "analysis_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        exps = manifest.setdefault("experiments", {})
        exps["1D"] = {
            "status": "COMPLETE",
            "timing_dir_exists": True,
            "standardized_rows": int(len(pub)),
            "perturbation_levels": {
                "fixed_delay": [0, 25, 50, 100, 200],
                "clock_offset": [-100, -50, -25, 0, 25, 50, 100],
                "timestamp_jitter": [0, 10, 25, 50],
            },
            "protocol": protocol,
            "unresolved_protocol_fields": [],
        }
        manifest["statuses"] = {
            k: v.get("status") for k, v in exps.items()
        }
        manifest["hard_failures"] = [
            x for x in manifest.get("hard_failures", []) if x != "1D"
        ]
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    # Update main Stage-1 report verdict without deleting prior block detail.
    report_path = stage1 / "STAGE1_PUBLICATION_HARDENING_REPORT.md"
    if report_path.exists():
        old = report_path.read_text(encoding="utf-8", errors="replace")
        old = old.replace(
            "Status: **NEEDS_PROTOCOL_CLARIFICATION**.",
            "Status: **COMPLETE**."
        )
        old = re.sub(
            r"- Unresolved provenance fields: `[^`]*`\.",
            "- Unresolved provenance fields: `[]`.",
            old
        )
        old = old.replace(
            "**COMPUTATIONAL RESULTS COMPLETE; TIMING PROTOCOL DOCUMENTATION STILL NEEDS RESOLUTION**",
            "**READY FOR STAGE 2 MANUSCRIPT REVISION**"
        )
        report_path.write_text(old, encoding="utf-8")

    print("=" * 72)
    print("Stage 1 timing resolution: PASS")
    print(f"Source: {src}")
    print(f"Paired statistics: {stats}")
    print(f"Output: {stage1}")
    print(f"Publication rows: {len(pub)}")
    print("Jitter: zero-mean Gaussian; sigma = 0,10,25,50 ms")
    print("Jitter user seeds: 0,1,2,3,4; deterministic per-file effective seed")
    print("Paired bootstrap: 20,000 resamples; seed 42; physical sequence is unit")
    print("Stage 1 verdict: READY FOR STAGE 2 MANUSCRIPT REVISION")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate that frozen V1 and V2 trajectories use identical physical timelines/GT.

This is a provenance/protocol check, not a model-performance comparison.
It pairs files by (sequence, replicate/seed), canonicalizes known column aliases,
and verifies row count, timestamps, and ground-truth pose arrays.

Exit status is non-zero if any pair is missing or any checked array differs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from DigitalTwin.baselines.common import canonicalize_columns, sequence_id, seed_id

CHECK_COLS = ("time_s", "gt_east_m", "gt_north_m", "gt_heading_rad")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v1-root", default="results/i2nav_v1_frozen/canonical_predictions")
    p.add_argument("--v1-glob", default="**/*trajectory.csv")
    p.add_argument("--v2-root", default="results/i2nav_v2_full_loso/i2nav_v2_full_loso")
    p.add_argument("--v2-glob", default="**/v2_evaluated_trajectory.csv")
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--output", default="results/i2nav_fidelity_baselines/v1_v2_protocol_validation.csv")
    return p.parse_args()


def collect(root: Path, pattern: str, label: str):
    files = sorted(root.glob(pattern), key=lambda p: str(p).lower()) if root.exists() else []
    if not files:
        raise FileNotFoundError(f"No {label} files found under {root} with glob {pattern!r}")
    out = {}
    for p in files:
        key = (sequence_id(p), seed_id(p))
        if key in out:
            raise RuntimeError(f"Duplicate {label} key {key}: {out[key]} and {p}")
        out[key] = p
    return out


def load_checked(path: Path):
    d = canonicalize_columns(pd.read_csv(path))
    missing = [c for c in CHECK_COLS if c not in d.columns]
    if missing:
        raise ValueError(f"{path}: missing required protocol columns {missing}")
    return d


def main():
    a = parse_args()
    v1 = collect(Path(a.v1_root), a.v1_glob, "V1")
    v2 = collect(Path(a.v2_root), a.v2_glob, "V2")
    keys = sorted(set(v1) | set(v2))
    rows = []

    for key in keys:
        s, seed = key
        p1, p2 = v1.get(key), v2.get(key)
        row = {
            "sequence": s, "seed": seed,
            "v1_path": str(p1) if p1 else "",
            "v2_path": str(p2) if p2 else "",
            "pair_present": p1 is not None and p2 is not None,
        }
        if p1 is None or p2 is None:
            row.update({"rows_v1": np.nan, "rows_v2": np.nan, "row_count_identical": False,
                        **{f"{c}_identical": False for c in CHECK_COLS}, "all_identical": False})
            rows.append(row)
            continue

        A, B = load_checked(p1), load_checked(p2)
        same_len = len(A) == len(B)
        row["rows_v1"] = len(A); row["rows_v2"] = len(B); row["row_count_identical"] = same_len
        all_ok = same_len
        for c in CHECK_COLS:
            ok = False
            max_abs = np.nan
            if same_len:
                aa = pd.to_numeric(A[c], errors="coerce").to_numpy(float)
                bb = pd.to_numeric(B[c], errors="coerce").to_numpy(float)
                ok = bool(np.allclose(aa, bb, equal_nan=True, rtol=0.0, atol=a.atol))
                finite = np.isfinite(aa) & np.isfinite(bb)
                if finite.any():
                    max_abs = float(np.max(np.abs(aa[finite] - bb[finite])))
            row[f"{c}_identical"] = ok
            row[f"{c}_max_abs_diff"] = max_abs
            all_ok = all_ok and ok
        row["all_identical"] = bool(all_ok)
        rows.append(row)

    out = pd.DataFrame(rows)
    op = Path(a.output); op.parent.mkdir(parents=True, exist_ok=True); out.to_csv(op, index=False)
    print(f"V1 files: {len(v1)} | V2 files: {len(v2)} | paired keys: {sum(out['pair_present'])}/{len(out)}")
    print(out[["sequence","seed","rows_v1","rows_v2","all_identical"]].to_string(index=False))
    passed = len(v1) == 30 and len(v2) == 30 and len(out) == 30 and bool(out["all_identical"].all())
    print(f"\nProtocol equivalence: {'PASS' if passed else 'FAIL'}")
    print(f"Wrote {op}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

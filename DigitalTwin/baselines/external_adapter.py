#!/usr/bin/env python3
"""Normalize an externally generated baseline trajectory into the suite schema.

Use this for methods that should not be reimplemented casually (e.g. WING or
an official YNet/LWOI implementation). It does not run the external algorithm;
it only performs timestamp alignment and schema normalization so TFP, Muñoz,
and Bergs evaluators can consume its output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .common import canonicalize_columns, load_raw_sequence, wrap_angle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="External method CSV")
    p.add_argument("--reference", required=True, help="Reference i2Nav raw/GT trajectory CSV for the same sequence")
    p.add_argument("--output", required=True)
    p.add_argument("--method-name", required=True)
    p.add_argument("--column-map-json", default="{}", help='Map canonical names to external names, e.g. {"time_s":"stamp","estimate_east_m":"x"}')
    p.add_argument("--heading-degrees", action="store_true")
    p.add_argument("--time-offset-s", type=float, default=0.0)
    p.add_argument("--max-time-error-s", type=float, default=0.15)
    return p.parse_args()


def main():
    a = parse_args()
    ext = pd.read_csv(a.input)
    mapping = json.loads(a.column_map_json)
    # mapping is canonical -> source; pandas rename needs source -> canonical
    ext = ext.rename(columns={src: canonical for canonical, src in mapping.items()})
    ext = canonicalize_columns(ext)
    req = ["time_s", "estimate_east_m", "estimate_north_m", "estimate_heading_rad"]
    miss = [c for c in req if c not in ext.columns]
    if miss:
        raise SystemExit(f"External CSV missing {miss}. Supply --column-map-json. Columns={list(ext.columns)}")
    for c in req:
        ext[c] = pd.to_numeric(ext[c], errors="coerce")
    ext = ext.dropna(subset=req).sort_values("time_s")
    ext["time_s"] += a.time_offset_s
    if a.heading_degrees:
        ext["estimate_heading_rad"] = np.deg2rad(ext["estimate_heading_rad"].to_numpy(float))
    ext["estimate_heading_rad"] = wrap_angle(ext["estimate_heading_rad"].to_numpy(float))

    ref = load_raw_sequence(a.reference, merge_prediction_trace=True)
    left = ref.sort_values("time_s").copy()
    right = ext[req].sort_values("time_s")
    merged = pd.merge_asof(left, right, on="time_s", direction="nearest", tolerance=a.max_time_error_s)
    merged = merged.dropna(subset=["estimate_east_m", "estimate_north_m", "estimate_heading_rad"]).reset_index(drop=True)
    if len(merged) < 3:
        raise SystemExit("Fewer than 3 aligned external samples; inspect timestamps/offset/tolerance")
    merged["method"] = a.method_name
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in [
        "time_s", "gt_east_m", "gt_north_m", "gt_heading_rad",
        "estimate_east_m", "estimate_north_m", "estimate_heading_rad",
        "odo_speed_mps", "imu_yaw_rate_radps", "wheel_yaw_radps",
        "wheel_imu_yaw_disagreement_radps", "method",
    ] if c in merged.columns]
    merged[keep].to_csv(out, index=False)
    print(f"Wrote {out} ({len(merged)} aligned rows)")


if __name__ == "__main__":
    main()

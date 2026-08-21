#!/usr/bin/env python3
"""Find candidate frozen trajectory CSVs under results/ without choosing among ambiguous sets."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from DigitalTwin.baselines.common import canonicalize_columns, REQUIRED_POSE, sequence_id, seed_id


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="results"); p.add_argument("--output",default="results/i2nav_fidelity_baselines/trajectory_candidates.csv"); return p.parse_args()


def infer_method(path:Path)->str:
    s=str(path).lower()
    if "fixed" in s and "phys" in s: return "Fixed Physics candidate"
    if "v1" in s: return "Twin V1 candidate"
    if "v2" in s: return "Twin V2 candidate"
    if "ekf" in s: return "EKF candidate"
    return "unclassified"


def main():
    a=parse_args(); root=Path(a.root); rows=[]
    for p in root.rglob("*.csv"):
        if "trajectory" not in p.name.lower(): continue
        try:
            h=pd.read_csv(p,nrows=2); h=canonicalize_columns(h)
            if not all(c in h.columns for c in REQUIRED_POSE): continue
        except Exception: continue
        rows.append({"candidate_method":infer_method(p),"sequence":sequence_id(p),"seed":seed_id(p),"path":str(p),"parent":str(p.parent)})
    d=pd.DataFrame(rows)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); d.to_csv(out,index=False)
    if len(d):
        print(d.groupby("candidate_method").agg(files=("path","size"),sequences=("sequence","nunique"),seeds=("seed","nunique")).to_string())
        print("\nFixed-Physics candidates:")
        print(d[d.candidate_method=="Fixed Physics candidate"].to_string(index=False))
    else: print("No canonical evaluated trajectory CSVs found")
    print(f"Wrote {out}")

if __name__=="__main__": main()

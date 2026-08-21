#!/usr/bin/env python3
"""Synthetic invariance/sanity checks for the Muñoz-style adaptation.

This does NOT replace validation against the authors' official artifact, but it
catches implementation regressions before running the expensive i2Nav matrix.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .munoz_trace_alignment_multimethod import score_position, score_heading


def make_trace(offset_xy=(0.0,0.0), heading_offset=0.0, duplicate_prefix=0):
    t=np.arange(0,60,1.0)
    gx=0.5*t; gy=2*np.sin(t/15); gh=np.arctan2(np.gradient(gy),np.gradient(gx))
    ex=gx+offset_xy[0]; ey=gy+offset_xy[1]; eh=gh+heading_offset
    if duplicate_prefix>0:
        ex=np.r_[np.repeat(ex[0],duplicate_prefix),ex][:len(ex)]
        ey=np.r_[np.repeat(ey[0],duplicate_prefix),ey][:len(ey)]
        eh=np.r_[np.repeat(eh[0],duplicate_prefix),eh][:len(eh)]
    dt=np.ones_like(t)
    gs=np.hypot(np.diff(gx,prepend=gx[0]),np.diff(gy,prepend=gy[0]))/dt
    es=np.hypot(np.diff(ex,prepend=ex[0]),np.diff(ey,prepend=ey[0]))/dt
    return {"t":t,"gx":gx,"gy":gy,"gh":gh,"ex":ex,"ey":ey,"eh":eh,"gspeed":gs,"espeed":es,"duration_s":59.0,"effective_hz":1.0}


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="results/i2nav_fidelity_baselines/validation/munoz_synthetic_checks.csv"); return p.parse_args()


def main():
    a=parse_args(); rows=[]
    cases=[
        ("identical",make_trace(),True),
        ("large_position_offset",make_trace(offset_xy=(0.0,5.0)),False),
        ("large_heading_offset",make_trace(heading_offset=np.deg2rad(60)),False),
        ("temporal_prefix_delay",make_trace(duplicate_prefix=4),None),
    ]
    for name,tr,_ in cases:
        p=score_position(tr,0.5,-1.0,-0.1,1.0,0.05)
        h=score_heading(tr,5.0,-1.0,-0.1,1.0,0.05)
        rows.append({"case":name,"position_pct_matched":p["pct_matched"],"position_ed_m":p["ed_mean_m"],"heading_pct_matched":h["pct_matched"],"heading_ed_deg":h["ed_mean_deg"]})
    d=pd.DataFrame(rows)
    ident=d[d.case=="identical"].iloc[0]
    posoff=d[d.case=="large_position_offset"].iloc[0]
    headoff=d[d.case=="large_heading_offset"].iloc[0]
    checks=[
        ("identical_position_100",abs(ident.position_pct_matched-100)<1e-6),
        ("identical_position_zero_ed",abs(ident.position_ed_m)<1e-9),
        ("identical_heading_100",abs(ident.heading_pct_matched-100)<1e-6),
        ("identical_heading_zero_ed",abs(ident.heading_ed_deg)<1e-9),
        ("large_position_offset_low_match",posoff.position_pct_matched<10),
        ("large_heading_offset_low_match",headoff.heading_pct_matched<10),
    ]
    d["notes"]=""
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); d.to_csv(out,index=False)
    for name,ok in checks: print(("PASS" if ok else "FAIL"),name)
    if not all(ok for _,ok in checks): raise SystemExit(2)
    print(d.to_string(index=False)); print(f"Wrote {out}")

if __name__=="__main__": main()

#!/usr/bin/env python3
"""Create a reproducibility snapshot for the paper/supplement."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
import pandas as pd

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def git(cmd):
    try:
        return subprocess.check_output(["git"]+cmd,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return "unavailable"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v1-root",default="results/i2nav_v1_frozen/canonical_predictions")
    ap.add_argument("--v2-root",default="results/i2nav_v2_full_loso/i2nav_v2_full_loso")
    ap.add_argument("--output-root",default="results/publication_hardening")
    a=ap.parse_args()
    out=Path(a.output_root);out.mkdir(parents=True,exist_ok=True)
    v1=sorted(Path(a.v1_root).rglob("*.csv")) if Path(a.v1_root).exists() else []
    v2=sorted(Path(a.v2_root).rglob("v2_evaluated_trajectory.csv")) if Path(a.v2_root).exists() else []
    sample=[]
    for p in (v1[:3]+v2[:3]):
        sample.append({"path":str(p),"sha256":sha256(p),"bytes":p.stat().st_size})
    info={
      "generated_utc":pd.Timestamp.utcnow().isoformat(),
      "python":sys.version.replace("\n"," "),
      "platform":platform.platform(),
      "git_commit":git(["rev-parse","HEAD"]),
      "git_status_porcelain":git(["status","--porcelain"]),
      "v1_csv_count":len(v1),"v2_evaluated_trajectory_count":len(v2),
      "primary_statistical_unit":"physical sequence",
      "known_i2nav_sequences":["building00","building01","building02","parking00","parking01","parking02","playground00","street00","street01","street02"],
      "known_training_seed_bases":[42,1042,2042],
      "operational_alignment":"common initialized physical-virtual frame; no post-hoc alignment",
      "official_benchmark_alignment":"reported separately; audit exact evo/source aggregation before final freeze",
      "sample_file_hashes":sample,
    }
    (out/"reproducibility_snapshot.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    md=["# Reproducibility snapshot","",
        f"- Git commit: `{info['git_commit']}`",
        f"- V1 CSV count: {len(v1)}",
        f"- Frozen V2 evaluated trajectories: {len(v2)}",
        "- Primary statistical unit: physical sequence.",
        "- Operational fidelity uses the common initialized physical--virtual frame without post-hoc trajectory alignment.",
        "- Official benchmark alignment is reported separately.",
        "- The V1/V2 protocol-equivalence validator should be retained with the release.",
        "",
        "## Environment","",
        f"- Python: `{info['python']}`",
        f"- Platform: `{info['platform']}`",
        "",
        "## Release checklist","",
        "- Record interpolation and missing-data rules.",
        "- Record quantile definition and bootstrap seed/count.",
        "- Record Muñoz MAD/gap/LCAW settings.",
        "- Record timing perturbation procedure.",
        "- Release per-sequence and per-seed trajectories/results where permitted.",
        "- Keep GPS/RTK evaluation-only if used as an independent reference."
    ]
    (out/"reproducibility_snapshot.md").write_text("\n".join(md),encoding="utf-8")
    print(json.dumps(info,indent=2))

if __name__=="__main__":
    main()

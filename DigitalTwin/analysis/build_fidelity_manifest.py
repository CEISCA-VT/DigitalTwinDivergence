#!/usr/bin/env python3
"""Build one trajectory manifest spanning official and generated baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from DigitalTwin.baselines.common import sequence_id, seed_id, safe_relative


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="baseline_suite_config.json")
    p.add_argument("--output", default="results/i2nav_fidelity_baselines/trajectory_manifest.csv")
    p.add_argument("--repo-root", default=".")
    return p.parse_args()


def main():
    a = parse_args(); repo = Path(a.repo_root).resolve()
    cfg_path = Path(a.config); cfg_path = cfg_path if cfg_path.is_absolute() else repo / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    rows = []

    gen_manifest = cfg.get("generated_baseline_manifest")
    if gen_manifest:
        gp = Path(gen_manifest); gp = gp if gp.is_absolute() else repo / gp
        if gp.exists():
            g = pd.read_csv(gp)
            g = g[g.get("status", "ok") == "ok"] if "status" in g.columns else g
            for _, r in g.iterrows():
                p = Path(str(r["trajectory"])); p = p if p.is_absolute() else repo / p
                if not p.exists():
                    raise FileNotFoundError(f"Generated baseline manifest references missing file: {p}")
                rows.append({
                    "method": str(r["method"]), "sequence": str(r["sequence"]), "seed": str(r["seed"]),
                    "trajectory": safe_relative(p, repo), "source": "generated_baseline",
                    "provenance": str(r.get("adaptation_level", "")),
                })
        elif cfg.get("generated_baseline_manifest_required", False):
            raise FileNotFoundError(gp)
        else:
            print(f"NOTE: generated baseline manifest not found; skipping: {gp}")

    for spec in cfg.get("official_methods", []):
        if not bool(spec.get("enabled", True)):
            print(f"NOTE: official method disabled in config: {spec.get('name', '<unnamed>')}")
            continue
        name = spec["name"]
        root = Path(spec["root"]); root = root if root.is_absolute() else repo / root
        pattern = spec.get("glob", "**/*evaluated_trajectory.csv")
        files = sorted(root.glob(pattern), key=lambda p: str(p).lower()) if root.exists() else []
        required = bool(spec.get("required", False))
        expected = spec.get("expected_files")
        if not files:
            msg = f"No files for official method {name!r}: {root / pattern}"
            if required:
                raise FileNotFoundError(msg)
            print("NOTE:", msg, "-- skipped")
            continue
        if expected is not None and len(files) != int(expected):
            raise RuntimeError(f"{name}: expected {expected} files but found {len(files)} under {root}")
        for p in files:
            rows.append({
                "method": name,
                "sequence": sequence_id(p),
                "seed": seed_id(p),
                "trajectory": safe_relative(p, repo),
                "source": spec.get("source", "official_frozen"),
                "provenance": spec.get("provenance", ""),
            })

    m = pd.DataFrame(rows)
    if not len(m):
        raise SystemExit("No trajectory files were added to the manifest")
    m = m.drop_duplicates(subset=["method", "sequence", "seed", "trajectory"]).sort_values(["method", "sequence", "seed"]).reset_index(drop=True)
    out = Path(a.output); out = out if out.is_absolute() else repo / out; out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    counts = m.groupby("method").agg(files=("trajectory", "size"), sequences=("sequence", "nunique"), seeds=("seed", "nunique")).reset_index()
    counts.to_csv(out.with_name("trajectory_manifest_counts.csv"), index=False)
    print(counts.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Collect publication-relevant figures from the DigitalTwinDivergence repository.

Run from repository root:
    python collect_manuscript_rewrite_figures.py

Optional:
    python collect_manuscript_rewrite_figures.py --refresh
    python collect_manuscript_rewrite_figures.py --strict
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable

CORE = [
    {
        "name": "e1_parking_full_grid_inversion.png",
        "patterns": ["e1_parking_full_grid_inversion.png"],
        "experiment": "E1 i2Nav",
        "role": "CORE",
        "note": "Threshold-robust parking00 vs parking02 local/global service inversion.",
        "required": True,
    },
    {
        "name": "e1_metric_service_rank_alignment.png",
        "patterns": ["e1_metric_service_rank_alignment.png"],
        "experiment": "E1 i2Nav",
        "role": "CORE",
        "note": "ATE/RPE rank alignment with different service-validity claims.",
        "required": True,
    },
    {
        "name": "E1_E2_cross_platform_position_contracts.png",
        "patterns": [
            "E1_E2_cross_platform_position_contracts.png",
            "e1_e2_cross_platform_position_contracts.png",
        ],
        "experiment": "E1/E2",
        "role": "CORE",
        "note": "Unchanged service-contract evaluation across i2Nav and TerraSentia.",
        "required": True,
    },
    {
        "name": "cross_domain_horizon_profile.png",
        "patterns": ["cross_domain_horizon_profile.png"],
        "experiment": "E3 cross-domain",
        "role": "CORE",
        "note": "MAGNET, FreeTwinEV, and TU Wien SNG horizon-dependent contract validity.",
        "required": True,
    },
    {
        "name": "cross_domain_contract_heatmap.png",
        "patterns": ["cross_domain_contract_heatmap.png"],
        "experiment": "E3 cross-domain",
        "role": "CORE",
        "note": "Cross-domain quantity x horizon x tolerance contract surface.",
        "required": True,
    },
    {
        "name": "condition_dependent_fidelity.png",
        "patterns": ["condition_dependent_fidelity.png"],
        "experiment": "i2Nav condition analysis",
        "role": "SUPPORTING",
        "note": "Condition-dependent fidelity characterization; not an online-monitor claim.",
        "required": False,
    },
    {
        "name": "persistent_yaw_mechanism.png",
        "patterns": ["persistent_yaw_mechanism.png"],
        "experiment": "i2Nav mechanism",
        "role": "SUPPORTING",
        "note": "Persistent yaw-disagreement mechanism behind accumulated divergence.",
        "required": False,
    },
    {
        "name": "coverage_vs_sharpness.png",
        "patterns": ["coverage_vs_sharpness.png", "*coverage*sharpness*.png"],
        "experiment": "Conditioned envelope analysis",
        "role": "SUPPORTING",
        "note": "Coverage/sharpness diagnostic; do not claim universal conditioned-envelope improvement.",
        "required": False,
    },
    {
        "name": "parking02_tfp_global_tail.png",
        "patterns": ["parking02_tfp_global_tail.png", "*parking02*global*tail*.png"],
        "experiment": "i2Nav parking02",
        "role": "SUPPORTING",
        "note": "Illustrative local-good/global-bad physical-virtual divergence case.",
        "required": False,
    },
    {
        "name": "magnet_horizon_component_counterexample.png",
        "patterns": ["magnet_horizon_component_counterexample.png", "*magnet*horizon*counterexample*.png"],
        "experiment": "MAGNET",
        "role": "SUPPORTING",
        "note": "Non-robot horizon-sensitive fidelity evidence.",
        "required": False,
    },
    {
        "name": "ugv01_asset_instantiation.png",
        "patterns": ["ugv01_asset_instantiation.png"],
        "experiment": "UGV01",
        "role": "SUPPORTING",
        "note": "Physical asset-specific twin instantiation.",
        "required": False,
    },
    {
        "name": "ugv01_condition_fidelity_profile.png",
        "patterns": ["ugv01_condition_fidelity_profile.png"],
        "experiment": "UGV01",
        "role": "SUPPORTING",
        "note": "UGV01 condition-dependent fidelity profile.",
        "required": False,
    },
    {
        "name": "ugv01_local_vs_global_fidelity.png",
        "patterns": ["ugv01_local_vs_global_fidelity.png"],
        "experiment": "UGV01",
        "role": "SUPPORTING",
        "note": "UGV01 local-versus-global fidelity illustration.",
        "required": False,
    },
    {
        "name": "representative_divergence_trace.png",
        "patterns": ["representative_divergence_trace.png"],
        "experiment": "Physical-virtual trace",
        "role": "SUPPORTING",
        "note": "Representative synchronized physical-virtual divergence trace.",
        "required": False,
    },
]

SUPPLEMENT_PATTERNS = [
    "*v1*v2*.png",
    "*V1*V2*.png",
    "*service*surface*.png",
    "*contract*surface*.png",
    "*freetwinev*.png",
    "*sng*.png",
    "*tuwien*.png",
    "*magnet*.png",
    "*terrasentia*.png",
    "*aifarms*.png",
    "*timing*.png",
    "*jitter*.png",
    "*delay*.png",
    "*condition*.png",
    "*envelope*.png",
    "*parking00*.png",
    "*parking01*.png",
    "*parking02*.png",
]

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
}


def should_exclude(path: Path, repo: Path, out_root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(repo.resolve())
    except ValueError:
        return True

    if set(rel.parts) & EXCLUDED_DIR_NAMES:
        return True

    try:
        path.resolve().relative_to(out_root.resolve())
        return True
    except ValueError:
        return False


def find_matches(repo: Path, out_root: Path, patterns: Iterable[str]) -> list[Path]:
    found = {}
    for pattern in patterns:
        for p in repo.rglob(pattern):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
                continue
            if should_exclude(p, repo, out_root):
                continue
            found[p.resolve()] = None
    return list(found.keys())


def newest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def copy_file(src: Path, dst: Path, refresh: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not refresh:
        return "kept_existing"

    if src.resolve() == dst.resolve():
        return "already_in_place"

    shutil.copy2(src, dst)
    return "copied"


def unique_destination(directory: Path, filename: str) -> Path:
    dst = directory / filename
    if not dst.exists():
        return dst

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 2
    while True:
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def relpath(path: Path | None, repo: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_root = repo / "figures"
    supp_dir = out_root / "supplement"
    out_root.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    claimed_sources = set()
    missing_required = []

    print(f"[repo] {repo}")
    print(f"[out ] {out_root}")
    print()
    print("[1/2] Collecting curated manuscript figures...")

    for item in CORE:
        matches = find_matches(repo, out_root, item["patterns"])
        src = newest(matches)

        if src is None:
            status = "MISSING_REQUIRED" if item["required"] else "not_found"
            if item["required"]:
                missing_required.append(item["name"])
            print(f"  [{status}] {item['name']}")
            manifest.append({
                "destination": item["name"],
                "experiment": item["experiment"],
                "role": item["role"],
                "required": str(item["required"]),
                "status": status,
                "source": "",
                "note": item["note"],
            })
            continue

        claimed_sources.add(src.resolve())
        dst = out_root / item["name"]
        status = copy_file(src, dst, args.refresh)
        print(f"  [{status}] {item['name']} <- {relpath(src, repo)}")
        manifest.append({
            "destination": relpath(dst, repo),
            "experiment": item["experiment"],
            "role": item["role"],
            "required": str(item["required"]),
            "status": status,
            "source": relpath(src, repo),
            "note": item["note"],
        })

    print()
    print("[2/2] Collecting secondary/context figures...")

    supplement_sources = {}
    for pattern in SUPPLEMENT_PATTERNS:
        for p in find_matches(repo, out_root, [pattern]):
            rp = p.resolve()
            if rp in claimed_sources:
                continue
            supplement_sources[rp] = None

    supplement_count = 0
    for src in sorted(supplement_sources.keys(), key=lambda p: str(p).lower()):
        dst = unique_destination(supp_dir, src.name)
        status = copy_file(src, dst, args.refresh)
        supplement_count += 1
        manifest.append({
            "destination": relpath(dst, repo),
            "experiment": "secondary/context",
            "role": "SUPPLEMENT",
            "required": "False",
            "status": status,
            "source": relpath(src, repo),
            "note": "Automatically collected context/diagnostic figure; inspect before manuscript use.",
        })

    csv_path = out_root / "FIGURE_REWRITE_MANIFEST.csv"
    fields = ["destination", "experiment", "role", "required", "status", "source", "note"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    md_path = out_root / "FIGURE_REWRITE_MANIFEST.md"
    core_rows = [r for r in manifest if r["role"] != "SUPPLEMENT"]

    lines = [
        "# Manuscript rewrite figure manifest",
        "",
        "Generated by `collect_manuscript_rewrite_figures.py`.",
        "",
        "## Recommended main-paper starting set",
        "",
        "1. E1 full-grid parking00/parking02 inversion.",
        "2. E1 metric-to-service rank-alignment comparison.",
        "3. E1/E2 i2Nav-to-TerraSentia contract portability.",
        "4. E3 MAGNET/FreeTwinEV/SNG horizon profile.",
        "5. E3 cross-domain contract heatmap.",
        "6. Condition-dependent fidelity and persistent-yaw mechanism if space permits.",
        "",
        "## Curated figures",
        "",
        "| Destination | Experiment | Role | Status | Source |",
        "|---|---|---|---|---|",
    ]
    for r in core_rows:
        lines.append(
            f"| `{r['destination']}` | {r['experiment']} | {r['role']} | "
            f"{r['status']} | `{r['source']}` |"
        )

    lines += [
        "",
        "## Claim-boundary reminders",
        "",
        "- E1 supports service-relative fidelity; it does not claim ATE or RPE are invalid.",
        "- E2 supports unchanged contract-structure portability to TerraSentia; it is not a frozen-V2 superiority claim.",
        "- E3 supports cross-domain portability of the contract representation; normalized tolerances are not universal safety limits.",
        "- Old V1/V2 internal figures should not be mixed with the later official Fixed-Physics-vs-V2 headline comparison.",
        "",
        f"Supplementary/context figures collected: **{supplement_count}**.",
        "",
    ]

    if missing_required:
        lines += ["## Missing required figures", ""]
        for name in missing_required:
            lines.append(f"- `{name}`")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("============================================================")
    print("Figure collection complete")
    print(f"Curated output : {out_root}")
    print(f"Supplement     : {supp_dir}")
    print(f"Manifest CSV   : {csv_path}")
    print(f"Manifest MD    : {md_path}")
    print(f"Supplement figs: {supplement_count}")

    if missing_required:
        print()
        print("Missing required E1/E2/E3 figures:")
        for name in missing_required:
            print(f"  - {name}")

    print("============================================================")

    if args.strict and missing_required:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

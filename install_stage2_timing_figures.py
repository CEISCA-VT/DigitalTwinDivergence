#!/usr/bin/env python3
"""
Install the Stage-2 timing figures into the manuscript figures directory.

Run from the DigitalTwinDivergence repository root:

    python .\install_stage2_timing_figures.py

What it does:
1. Finds the frozen timing plots already produced by the timing study:
       results/i2nav_timing_sensitivity/jitter_sensitivity.png
       results/i2nav_timing_sensitivity/delay_sensitivity.png
2. Copies them to the canonical Stage-2 manuscript names:
       figures/timing_jitter_stage2.png
       figures/timing_delay_stage2.png
3. Scans root-level .tex files for any additional missing timing-jitter or
   timing-delay figure paths and fills those aliases too.
4. Verifies that the timing figure references now exist.

No external Python packages are required.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def newest(paths):
    paths = [p for p in paths if p.exists() and p.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def find_source(repo: Path, kind: str) -> Path:
    filename = f"{kind}_sensitivity.png"

    # Preferred frozen timing-study location.
    preferred = repo / "results" / "i2nav_timing_sensitivity" / filename
    if preferred.exists():
        return preferred

    # Common fallback if the figure was copied to repository root.
    root_copy = repo / filename
    if root_copy.exists():
        return root_copy

    # Last-resort recursive search. Exclude manuscript figures so that we
    # do not accidentally use a previous alias as the scientific source.
    candidates = []
    for p in repo.rglob(filename):
        try:
            rel = p.relative_to(repo)
        except ValueError:
            continue

        parts_lower = {x.lower() for x in rel.parts}
        if "figures" in parts_lower:
            continue
        if ".git" in parts_lower or "venv" in parts_lower or ".venv" in parts_lower:
            continue
        candidates.append(p)

    src = newest(candidates)
    if src is None:
        raise FileNotFoundError(
            f"Could not find {filename}.\n"
            f"Expected: {preferred}\n"
            "Re-run the frozen i2Nav timing analysis if that source figure is absent."
        )
    return src


def copy_figure(src: Path, dst: Path, repo: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Avoid copying a file onto itself.
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)

    print(f"  [ok] {dst.relative_to(repo)}")
    print(f"       <- {src.relative_to(repo)}")


def extract_graphic_paths(tex_text: str):
    # Handles both \safegraphic{...} and \includegraphics[...]{...}.
    pattern = re.compile(
        r"\\(?:safegraphic|includegraphics)"
        r"(?:\s*\[[^\]]*\])?"
        r"\s*\{([^{}]+)\}"
    )
    return [m.group(1).strip() for m in pattern.finditer(tex_text)]


def resolve_tex_graphic(repo: Path, tex_file: Path, ref: str) -> Path:
    p = Path(ref.replace("\\", "/"))

    # LaTeX path is normally relative to the .tex file.
    direct = tex_file.parent / p
    if direct.exists():
        return direct

    # In this project manuscripts are generally compiled from repo root.
    return repo / p


def main() -> int:
    repo = Path.cwd().resolve()
    figures = repo / "figures"

    print(f"[repo] {repo}")
    print("[1/3] Locating frozen timing-study figures...")

    jitter_src = find_source(repo, "jitter")
    delay_src = find_source(repo, "delay")

    print("[2/3] Installing canonical Stage-2 figure names...")

    canonical = {
        "jitter": figures / "timing_jitter_stage2.png",
        "delay": figures / "timing_delay_stage2.png",
    }

    copy_figure(jitter_src, canonical["jitter"], repo)
    copy_figure(delay_src, canonical["delay"], repo)

    print("[3/3] Checking .tex timing references and filling aliases...")

    # Prefer the known Stage-2 manuscript names, but scan every root-level
    # manuscript too so the script still works if main.tex was renamed.
    tex_files = sorted(repo.glob("*.tex"))

    if not tex_files:
        print("  [warn] No root-level .tex files found; canonical figures were still installed.")
    else:
        for tex in tex_files:
            text = tex.read_text(encoding="utf-8", errors="replace")
            refs = extract_graphic_paths(text)

            for ref in refs:
                low = ref.lower().replace("\\", "/")
                if "timing" not in low:
                    continue

                target = resolve_tex_graphic(repo, tex, ref)

                if target.exists():
                    continue

                if "jitter" in low:
                    copy_figure(jitter_src, target, repo)
                elif "delay" in low:
                    copy_figure(delay_src, target, repo)

    # Final verification for the exact paths used by the Stage-2 rewrite.
    required = [
        figures / "timing_jitter_stage2.png",
        figures / "timing_delay_stage2.png",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n[error] Required timing figures are still missing:")
        for p in missing:
            print(" ", p)
        return 2

    print("\nTiming figures installed successfully.")
    print("Required manuscript files now present:")
    for p in required:
        print(" ", p.relative_to(repo))

    print("\nYou can now recompile the manuscript.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

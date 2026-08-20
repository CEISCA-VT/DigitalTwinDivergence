"""Audit the paper-facing source, figures, and frozen result inputs.

This is a read-only check. It does not retrain models or regenerate results.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Make direct execution (`python scripts/audit_paper_package.py`) resolve the
# repository package without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DigitalTwin.paper.paths import (
    FIGURES_ROOT,
    PAPER_SOURCE,
    REQUIRED_PAPER_ARTIFACTS,
    REQUIRED_PAPER_FIGURES,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict[str, object]:
    missing_inputs = [str(path) for path in REQUIRED_PAPER_ARTIFACTS if not path.is_file()]
    missing_figures = [name for name in REQUIRED_PAPER_FIGURES if not (FIGURES_ROOT / name).is_file()]
    return {
        "schema": "digital_twin_fidelity_paper_package_audit_v1",
        "paper_source": str(PAPER_SOURCE),
        "paper_source_exists": PAPER_SOURCE.is_file(),
        "missing_result_inputs": missing_inputs,
        "missing_figures": missing_figures,
        "status": "pass" if PAPER_SOURCE.is_file() and not missing_inputs and not missing_figures else "fail",
        "checksums": {
            "paper_source": sha256(PAPER_SOURCE) if PAPER_SOURCE.is_file() else None,
            "figures": {
                name: sha256(FIGURES_ROOT / name)
                for name in REQUIRED_PAPER_FIGURES
                if (FIGURES_ROOT / name).is_file()
            },
        },
    }


def main() -> None:
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()

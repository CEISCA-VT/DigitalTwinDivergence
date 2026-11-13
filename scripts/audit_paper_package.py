"""Audit the paper-facing source, figures, and frozen result inputs.

This is a read-only check. It does not retrain models or regenerate results.
"""

from __future__ import annotations

import hashlib
import argparse
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


def audit(require_evidence: bool = False) -> dict[str, object]:
    missing_evidence = [
        str(path.relative_to(PAPER_SOURCE.parents[1]))
        for path in REQUIRED_PAPER_ARTIFACTS
        if not path.is_file()
    ]
    missing_figures = [name for name in REQUIRED_PAPER_FIGURES if not (FIGURES_ROOT / name).is_file()]
    source_ok = PAPER_SOURCE.is_file()
    figures_ok = not missing_figures
    evidence_ok = not missing_evidence
    return {
        "schema": "digital_twin_fidelity_paper_package_audit_v1",
        "paper_source": str(PAPER_SOURCE),
        "paper_source_exists": source_ok,
        "evidence_status": "complete" if evidence_ok else "not bundled or incomplete",
        "missing_result_evidence": missing_evidence,
        "evidence_required_for_pass": require_evidence,
        "missing_figures": missing_figures,
        "status": "pass" if source_ok and figures_ok and (evidence_ok or not require_evidence) else "fail",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-evidence",
        action="store_true",
        help="fail unless all locally generated evidence files are present",
    )
    args = parser.parse_args()
    result = audit(require_evidence=args.require_evidence)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

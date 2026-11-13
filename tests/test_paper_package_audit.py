import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import audit_paper_package


class PaperPackageAuditTests(unittest.TestCase):
    def test_evidence_is_optional_unless_explicitly_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manuscript = root / "manuscript"
            figures = manuscript / "figures"
            manuscript.mkdir()
            figures.mkdir()
            source = manuscript / "main.tex"
            source.write_text("\\documentclass{article}\n", encoding="utf-8")
            (figures / "paper.png").write_bytes(b"figure")
            missing_evidence = root / "results" / "evidence.json"

            with patch.multiple(
                audit_paper_package,
                PAPER_SOURCE=source,
                FIGURES_ROOT=figures,
                REQUIRED_PAPER_FIGURES=("paper.png",),
                REQUIRED_PAPER_ARTIFACTS=(missing_evidence,),
            ):
                self.assertEqual(audit_paper_package.audit()["status"], "pass")
                self.assertEqual(
                    audit_paper_package.audit()["evidence_status"],
                    "not bundled or incomplete",
                )
                self.assertEqual(
                    audit_paper_package.audit(require_evidence=True)["status"],
                    "fail",
                )


if __name__ == "__main__":
    unittest.main()

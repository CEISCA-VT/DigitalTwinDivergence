"""Canonical paths for the frozen digital-twin fidelity paper package."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
FIGURES_ROOT = REPOSITORY_ROOT / "figures"

PAPER_SOURCE = REPOSITORY_ROOT / "DigitalTwin_Fidelity_Research_Draft.tex"
FROZEN_LOSO_ROOT = RESULTS_ROOT / "i2nav_v2_full_loso"
POST_LOSO_ROOT = RESULTS_ROOT / "i2nav_v2_post_loso_analysis"
UGV01_ROOT = RESULTS_ROOT / "ugv01_asset_instantiation"
OFFICIAL_BENCHMARK_ROOT = RESULTS_ROOT / "i2nav_official_benchmark"
SENSING_ROOT = RESULTS_ROOT / "i2nav_sensing_fidelity"
FINAL_AUDIT_ROOT = RESULTS_ROOT / "final_audit"


REQUIRED_PAPER_ARTIFACTS = (
    POST_LOSO_ROOT / "all_sequence_mechanism" / "mechanism_summary.md",
    POST_LOSO_ROOT / "condition_fidelity" / "condition_fidelity_summary.md",
    POST_LOSO_ROOT / "benign_fidelity_characterization" / "benign_fidelity_framework_summary.md",
    POST_LOSO_ROOT / "loso_envelope_validation" / "loso_benign_envelope_validation_summary.md",
    UGV01_ROOT / "ugv01_asset_instantiation_summary.md",
    OFFICIAL_BENCHMARK_ROOT / "official_macro_summary.csv",
    SENSING_ROOT / "sensing_fidelity_summary.md",
    FINAL_AUDIT_ROOT / "FINAL_RESULT_FREEZE_READINESS.md",
)


REQUIRED_PAPER_FIGURES = (
    "local_vs_global_fidelity.png",
    "persistent_yaw_mechanism.png",
    "condition_dependent_fidelity.png",
    "benign_envelope_by_condition.png",
    "loso_envelope_coverage.png",
    "ugv01_asset_instantiation.png",
    "ugv01_condition_fidelity_profile.png",
    "ugv01_rpe_vs_global_fidelity.png",
    "official_benchmark_results.png",
    "sensing_fidelity_tradeoff.png",
)

"""Canonical paths for the frozen digital-twin fidelity paper package."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
FIGURES_ROOT = REPOSITORY_ROOT / "manuscript" / "figures"

PAPER_SOURCE = REPOSITORY_ROOT / "manuscript" / "main.tex"
FROZEN_LOSO_ROOT = RESULTS_ROOT / "i2nav_v2_full_loso"
POST_LOSO_ROOT = RESULTS_ROOT / "i2nav_frozen_v2_fidelity_analysis"
UGV01_ROOT = RESULTS_ROOT / "ugv01_physical_instantiation"
OFFICIAL_BENCHMARK_ROOT = RESULTS_ROOT / "i2nav_official_benchmark"
SENSING_ROOT = RESULTS_ROOT / "sensing_fidelity_comparison"
FINAL_AUDIT_ROOT = RESULTS_ROOT / "result_freeze_audit"
LIVE_AUDIT_ROOT = RESULTS_ROOT / "ugv01_live_contract_dataset_audit"
LIVE_REPLAY_ROOT = RESULTS_ROOT / "ugv01_live_contract_trace_replay"


REQUIRED_PAPER_ARTIFACTS = (
    POST_LOSO_ROOT / "all_sequence_mechanism" / "mechanism_summary.md",
    POST_LOSO_ROOT / "condition_fidelity" / "condition_fidelity_summary.md",
    POST_LOSO_ROOT / "benign_fidelity_characterization" / "benign_fidelity_framework_summary.md",
    POST_LOSO_ROOT / "loso_envelope_validation" / "loso_benign_envelope_validation_summary.md",
    UGV01_ROOT / "ugv01_asset_instantiation_summary.md",
    OFFICIAL_BENCHMARK_ROOT / "official_macro_summary.csv",
    SENSING_ROOT / "sensing_fidelity_summary.md",
    FINAL_AUDIT_ROOT / "FINAL_RESULT_FREEZE_READINESS.md",
    LIVE_AUDIT_ROOT / "dataset_audit_report.md",
    LIVE_REPLAY_ROOT / "trace_replay_report.md",
)


REQUIRED_PAPER_FIGURES = (
    "e1_parking_full_grid_inversion.png",
    "e1_metric_service_rank_alignment.png",
    "E1_E2_cross_platform_position_contracts.png",
    "cross_domain_horizon_profile.png",
    "cross_domain_contract_heatmap.png",
    "persistent_yaw_mechanism.png",
    "condition_dependent_fidelity.png",
    "timing_jitter_stage2.png",
    "timing_delay_stage2.png",
    "ugv01_aug29_validation_summary.png",
    "ugv01_aug29_trajectory_fidelity.png",
    "ugv01_aug29_fidelity_diagnostics.png",
    "trace_replay_policy_comparison.png",
    "transport_capacity_sensitivity.png",
)

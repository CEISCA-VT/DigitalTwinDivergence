"""Canonical paths for the current networked-IoT fidelity manuscript."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
FIGURES_ROOT = REPOSITORY_ROOT / "manuscript" / "figures"
PAPER_SOURCE = REPOSITORY_ROOT / "manuscript" / "main.tex"

REQUIRED_PAPER_ARTIFACTS = (
    RESULTS_ROOT / "doubly_nested_loso_slurm" / "merged_manifest.json",
    RESULTS_ROOT / "doubly_nested_loso_slurm" / "full_audit.json",
    RESULTS_ROOT / "doubly_nested_loso_evidence" / "evidence_manifest.json",
    RESULTS_ROOT / "doubly_nested_loso_evidence" / "sequence_service_case_ledger.csv",
    RESULTS_ROOT / "doubly_nested_causal_loso_slurm" / "merged_manifest.json",
    RESULTS_ROOT / "doubly_nested_causal_loso_slurm" / "full_audit.json",
    RESULTS_ROOT / "doubly_nested_causal_loso_slurm" / "service_evidence" / "evidence_manifest.json",
    RESULTS_ROOT / "buffered_centered_heldout_100ms" / "experiment_manifest.json",
    RESULTS_ROOT / "buffered_centered_heldout_100ms" / "report.md",
    RESULTS_ROOT / "delivery_feasibility" / "manifest.json",
    RESULTS_ROOT / "delivery_feasibility" / "verification.json",
    RESULTS_ROOT / "delivery_feasibility" / "assessment.md",
    RESULTS_ROOT / "netem_delivery" / "evidence" / "netem_delivery_manifest.json",
    RESULTS_ROOT / "netem_delivery" / "evidence" / "netem_delivery_case_ledger.csv",
    RESULTS_ROOT / "ugv01_position_ledger" / "ugv01_position_ledger_manifest.json",
    RESULTS_ROOT / "cross_domain_contract_generalization" / "analysis_manifest.json",
)

REQUIRED_PAPER_FIGURES = (
    "framework_receiver_contract_overview.png",
    "i2nav_local_global_fidelity.png",
    "i2nav_building00_worked_example.pdf",
    "i2nav_service_delivery_qualification_grid.pdf",
    "delivery_planning_summary.png",
    "contract_component_comparison.pdf",
    "qualification_risk_sweep.pdf",
    "qualification_timing_25_50.pdf",
    "physical_tolerance.png",
    "satisfaction_requirement.png",
    "freshness_limit.png",
    "i2nav_freshness_three_rates.pdf",
    "ugv01_paired_case_column.png",
    "magnet_physical_discrepancy_pass_fraction.pdf",
)

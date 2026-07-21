import json
from pathlib import Path

from DigitalTwin.analysis.covariance_poisoning import (
    build_paired_effects,
    poisoning_conclusion,
)
from DigitalTwin.analysis.real_data_study import ATTACK_START_FRACTIONS, DRIFT_RATES_MPS


def _scenario(mode: str, q_ratio: float, state_error: float) -> dict[str, object]:
    return {
        "run_id": "low_smooth_trial-1",
        "speed": "low",
        "surface": "smooth",
        "trial": 1,
        "source_csv": "source.csv",
        "attack": "drift",
        "direction": "cross",
        "rate_mps": 0.05,
        "attack_start_fraction": 0.5,
        "attack_start_s": 10.0,
        "evaluation_horizon_s": 20.0,
        "transport": "baseline",
        "detector_variant": mode,
        "attack_window_q_trace_ratio": q_ratio,
        "attack_window_s_trace_ratio": 1.0,
        "attack_window_nis_ratio": 1.0,
        "max_undetected_state_deviation_m": state_error,
        "harmful_but_stealthy": int(state_error > 5.0),
        "run_detected": 0,
    }


def test_paired_effects_match_identical_attack_scenarios():
    rows = [
        _scenario("naive_adaptive", 1.2, 6.0),
        _scenario("fixed", 1.0, 4.0),
        _scenario("frozen_clean", 1.0, 5.0),
        _scenario("gps_independent", 1.0, 4.5),
        _scenario("evidence_gated", 1.1, 5.5),
    ]
    effects = build_paired_effects(rows)
    assert len(effects) == 4
    frozen = next(row for row in effects if row["comparison"].endswith("frozen_clean"))
    assert abs(float(frozen["q_trace_ratio_delta"]) - 0.2) < 1e-12
    assert float(frozen["max_undetected_state_deviation_delta_m"]) == 1.0


def test_primary_conclusion_separates_mechanism_from_operational_effect():
    metrics = {
        "q_trace_ratio_delta": (0.1, 0.05, 0.15),
        "nis_ratio_delta": (-0.1, -0.2, -0.05),
        "max_undetected_state_deviation_delta_m": (0.0, -0.1, 0.1),
        "harmful_but_stealthy_probability_delta": (0.0, -0.1, 0.1),
        "detection_probability_delta": (0.0, -0.1, 0.1),
    }
    rows = [
        {
            "comparison": "naive_adaptive_vs_frozen_clean",
            "scope": "standard_drift_pooled",
            "metric": metric,
            "mean_paired_effect": values[0],
            "cluster_bootstrap_ci95_low": values[1],
            "cluster_bootstrap_ci95_high": values[2],
        }
        for metric, values in metrics.items()
    ]
    conclusion = poisoning_conclusion(rows)
    assert conclusion["covariance_inflation_supported"] is True
    assert conclusion["nis_suppression_supported"] is True
    assert conclusion["covariance_poisoning_mechanism_supported"] is True
    assert conclusion["operational_attacker_advantage_supported"] is False


def test_covariance_analysis_config_matches_campaign_constants():
    payload = json.loads(
        Path("DigitalTwin/configs/covariance_poisoning_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["attack_start_fractions"] == list(ATTACK_START_FRACTIONS)
    assert payload["standard_drift_rates_mps"] == list(DRIFT_RATES_MPS)
    assert payload["primary_comparison"] == "naive_adaptive_vs_frozen_clean"

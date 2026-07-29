import math

import numpy as np

from DigitalTwin.detector import (
    chi_square_threshold,
    detectability_loss_factor,
    directional_detectability_bound,
    envelope_region,
    instantaneous_stealth_bound,
    nis_score_decomposition,
    structural_detectability_bound,
)


def test_chi_square_threshold_for_two_dimensional_gps():
    assert math.isclose(chi_square_threshold(2, 0.05), 5.991464547107982)


def test_structural_bound_uses_largest_innovation_eigenvalue():
    S = np.diag([1.0, 4.0])
    assert math.isclose(structural_detectability_bound(S, 9.0), 6.0)
    assert math.isclose(instantaneous_stealth_bound(S, 4.0), 4.0)


def test_directional_bound_and_detectability_loss_follow_information_form():
    reference = np.diag([1.0, 4.0])
    attacked = np.diag([4.0, 4.0])
    direction = np.array([1.0, 0.0])

    assert math.isclose(
        directional_detectability_bound(reference, 9.0, direction), 3.0
    )
    assert math.isclose(
        directional_detectability_bound(attacked, 9.0, direction), 6.0
    )
    assert math.isclose(
        detectability_loss_factor(attacked, reference, direction), 2.0
    )


def test_nis_decomposition_is_exact_and_separates_credit_from_growth():
    reference_S = np.eye(2)
    attacked_S = 2.0 * np.eye(2)
    reference_innovation = np.array([1.0, 0.0])
    attacked_innovation = np.array([2.0, 0.0])

    result = nis_score_decomposition(
        attacked_innovation,
        attacked_S,
        reference_innovation,
        reference_S,
    )

    assert math.isclose(result.attacked_nis, 2.0)
    assert math.isclose(result.reference_nis, 1.0)
    assert math.isclose(result.counterfactual_attacked_nis, 4.0)
    assert math.isclose(result.normalization_credit, 2.0)
    assert math.isclose(result.reference_metric_innovation_change, 3.0)
    assert math.isclose(result.nis_delta, 1.0)
    assert result.covariance_order_holds
    assert not result.suppression_condition_holds


def test_envelope_regions_match_proposal_thresholds():
    assert envelope_region(0.95) == "safe"
    assert envelope_region(0.75) == "warning"
    assert envelope_region(0.20) == "blind"

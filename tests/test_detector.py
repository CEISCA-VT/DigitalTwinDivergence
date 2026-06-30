import math

import numpy as np

from DigitalTwin.detector import (
    chi_square_threshold,
    envelope_region,
    instantaneous_stealth_bound,
    structural_detectability_bound,
)


def test_chi_square_threshold_for_two_dimensional_gps():
    assert math.isclose(chi_square_threshold(2, 0.05), 5.991464547107982)


def test_structural_bound_uses_largest_innovation_eigenvalue():
    S = np.diag([1.0, 4.0])
    assert math.isclose(structural_detectability_bound(S, 9.0), 6.0)
    assert math.isclose(instantaneous_stealth_bound(S, 4.0), 4.0)


def test_envelope_regions_match_proposal_thresholds():
    assert envelope_region(0.95) == "safe"
    assert envelope_region(0.75) == "warning"
    assert envelope_region(0.20) == "blind"

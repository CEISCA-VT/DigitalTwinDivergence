import numpy as np

from DigitalTwin.analysis.ugv01_position_remediability_ledger import delivered_indices, interval_has_large_gap


def test_delivered_indices_respect_rate_and_delay() -> None:
    t = np.arange(0.0, 2.01, 0.1)
    delivered = delivered_indices(t, rate_hz=2.0, delay_s=0.2)
    assert delivered[1] == -1
    assert delivered[2] == 0
    assert delivered[7] == 5


def test_interval_gap_detection() -> None:
    t = np.array([0.0, 0.5, 1.0, 2.0, 2.5])
    result = interval_has_large_gap(t, np.array([0, 2, 3]), np.array([1, 3, 4]), 0.75)
    np.testing.assert_array_equal(result, [False, True, False])

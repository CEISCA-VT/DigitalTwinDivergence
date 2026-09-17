import pytest

from DigitalTwin.analysis.magnet_timing_provenance import (
    require_causal_forecast_times,
    timestamp_semantics,
)


def test_target_time_is_not_promoted_to_issue_time():
    columns = ["Time (s)", "Heat Pipe TC-01"]
    assert timestamp_semantics(columns)["issue_time"] is None
    with pytest.raises(ValueError, match="issue times"):
        require_causal_forecast_times(columns)


def test_issue_time_must_be_independent_column():
    columns = ["Time (s)", "issue_time_s", "Heat Pipe TC-01"]
    semantics = require_causal_forecast_times(columns)
    assert semantics["target_time"] != semantics["issue_time"]


def test_arrival_is_not_fabricated_when_absent():
    assert timestamp_semantics(["Time (s)", "issue_time_s"])["arrival_time"] is None

"""Timing helpers for edge-side telemetry logging."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True, slots=True)
class ClockOffsetEstimate:
    remote_time_s: float
    edge_time_s: float
    offset_s: float
    calibrated: bool
    samples: int


class SessionClockCalibrator:
    """Estimate edge_time_s - remote_time_s for one logging session.

    The UGV01 firmware reports `millis()`, which is monotonic but not wall time.
    For bench logs we map that clock onto the edge machine's monotonic clock
    using the midpoint between request send and response receive.
    """

    def __init__(self, *, window_size: int = 25, min_samples: int = 5) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self.min_samples = min_samples
        self.offsets: deque[float] = deque(maxlen=window_size)

    def observe(self, remote_time_s: float, edge_time_s: float) -> ClockOffsetEstimate:
        offset_s = float(edge_time_s) - float(remote_time_s)
        self.offsets.append(offset_s)
        estimate_s = median(self.offsets)
        return ClockOffsetEstimate(
            remote_time_s=float(remote_time_s),
            edge_time_s=float(edge_time_s),
            offset_s=estimate_s,
            calibrated=len(self.offsets) >= self.min_samples,
            samples=len(self.offsets),
        )

    def edge_time_from_remote(self, remote_time_s: float) -> float | None:
        if not self.offsets:
            return None
        return float(remote_time_s) + median(self.offsets)

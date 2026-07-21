"""Operational alarm policy layered on top of per-update detector scores."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AlarmConfig:
    window_size: int = 5
    required_exceedances: int = 3
    initialization_gps_samples: int = 5
    motion_speed_threshold_mps: float = 0.02
    motion_yaw_rate_threshold_radps: float = 0.10
    motion_consecutive_updates: int = 2

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 1 <= self.required_exceedances <= self.window_size:
            raise ValueError("required_exceedances must be in [1, window_size]")
        if self.initialization_gps_samples <= 0:
            raise ValueError("initialization_gps_samples must be positive")
        if self.motion_consecutive_updates <= 0:
            raise ValueError("motion_consecutive_updates must be positive")


class PersistentAlarm:
    """Raise an alarm after k of the latest n NIS scores exceed threshold."""

    def __init__(self, threshold: float, config: AlarmConfig | None = None) -> None:
        self.threshold = float(threshold)
        self.config = config or AlarmConfig()
        self.exceedances: deque[bool] = deque(maxlen=self.config.window_size)

    def observe(self, score: float, *, enabled: bool) -> bool:
        if not enabled:
            self.exceedances.clear()
            return False
        self.exceedances.append(float(score) > self.threshold)
        return (
            len(self.exceedances) == self.config.window_size
            and sum(self.exceedances) >= self.config.required_exceedances
        )


def motion_start_index(controls: np.ndarray, config: AlarmConfig | None = None) -> int:
    """Return the first index completing a sustained tracked-drive motion onset."""

    policy = config or AlarmConfig()
    consecutive = 0
    for index, (velocity, yaw_rate) in enumerate(np.asarray(controls, dtype=float)):
        moving = (
            abs(float(velocity)) >= policy.motion_speed_threshold_mps
            or abs(float(yaw_rate)) >= policy.motion_yaw_rate_threshold_radps
        )
        consecutive = consecutive + 1 if moving else 0
        if consecutive >= policy.motion_consecutive_updates:
            return index - policy.motion_consecutive_updates + 1
    return 0


def robust_initial_state(
    gps_xy: np.ndarray,
    start_index: int,
    config: AlarmConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate initial position and covariance from recent pre-mission GPS."""

    policy = config or AlarmConfig()
    first = max(0, start_index - policy.initialization_gps_samples + 1)
    points = np.asarray(gps_xy[first : start_index + 1], dtype=float)
    center = np.median(points, axis=0)
    variances = np.maximum(np.var(points, axis=0), 0.25)
    state = np.array([center[0], center[1], 0.0], dtype=float)
    covariance = np.diag([variances[0], variances[1], 0.25])
    return state, covariance


def operational_run_statistic(
    scores: np.ndarray,
    enabled: np.ndarray,
    config: AlarmConfig | None = None,
) -> float:
    """Maximum k-of-n threshold statistic observed during one benign run.

    For a 3-of-5 alarm, each full window contributes its third-largest score.
    Locking above the maximum benign value prevents an alarm on calibration
    runs while retaining substantially more sensitivity than the raw run max.
    """

    policy = config or AlarmConfig()
    active_scores = np.asarray(scores, dtype=float)[np.asarray(enabled, dtype=bool)]
    if len(active_scores) < policy.window_size:
        return float(active_scores.max()) if len(active_scores) else 0.0
    statistics: list[float] = []
    rank = policy.required_exceedances
    for index in range(policy.window_size - 1, len(active_scores)):
        window = np.sort(active_scores[index - policy.window_size + 1 : index + 1])[::-1]
        statistics.append(float(window[rank - 1]))
    return max(statistics, default=0.0)

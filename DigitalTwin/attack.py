"""Attack injection module for synthetic and replayed telemetry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True, slots=True)
class AttackConfig:
    kind: str = "none"
    start_s: float = 0.0
    end_s: float | None = None
    epsilon_x_m: float = 0.0
    epsilon_y_m: float = 0.0
    drift_x_mps: float = 0.0
    drift_y_mps: float = 0.0
    replay_delay_s: float = 5.0
    noise_std_mps: float = 0.03


class AttackInjector:
    def __init__(self, config: AttackConfig, dt_s: float, seed: int = 11) -> None:
        self.config = config
        self.dt_s = dt_s
        self.rng = random.Random(seed)
        self.replay_buffer: deque[np.ndarray] = deque(maxlen=max(1, int(config.replay_delay_s / dt_s)))
        self.freeze_value: np.ndarray | None = None

    def active(self, t_s: float) -> bool:
        if self.config.kind == "none":
            return False
        if t_s < self.config.start_s:
            return False
        return self.config.end_s is None or t_s <= self.config.end_s

    def apply(self, t_s: float, measurement_xy: np.ndarray) -> tuple[np.ndarray, str]:
        measurement_xy = np.asarray(measurement_xy, dtype=float)
        self.replay_buffer.append(measurement_xy.copy())
        if not self.active(t_s):
            return measurement_xy, "none"

        kind = self.config.kind.lower()
        if kind == "step":
            return measurement_xy + np.array([self.config.epsilon_x_m, self.config.epsilon_y_m]), "step"
        if kind == "freeze":
            if self.freeze_value is None:
                self.freeze_value = measurement_xy.copy()
            return self.freeze_value.copy(), "freeze"
        if kind == "replay":
            if len(self.replay_buffer) == self.replay_buffer.maxlen:
                return self.replay_buffer[0].copy(), "replay"
            return measurement_xy, "replay-warmup"
        if kind in {"drift", "random_drift"}:
            age = max(0.0, t_s - self.config.start_s)
            deterministic = np.array([self.config.drift_x_mps, self.config.drift_y_mps]) * age
            random_walk = np.array(
                [
                    self.rng.gauss(0.0, self.config.noise_std_mps),
                    self.rng.gauss(0.0, self.config.noise_std_mps),
                ]
            ) * age
            return measurement_xy + deterministic + random_walk, "random_drift"
        raise ValueError(f"unknown attack kind {self.config.kind!r}")

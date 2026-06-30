"""Latency and buffered-delivery simulation for telemetry packets."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import random
from typing import Any


@dataclass(order=True, slots=True)
class DelayedItem:
    release_s: float
    order: int
    generated_s: float = field(compare=False)
    item: Any = field(compare=False)


@dataclass(frozen=True, slots=True)
class DeliveredItem:
    delivery_s: float
    generated_s: float
    item: Any


class LatencyQueue:
    """Queue packets with latency, jitter, and optional buffered release.

    Constant latency alone shifts all packets equally and does not change
    inter-arrival spacing.  Buffered release creates the burst/gap behavior that
    the proposal's Delta t_k feature is meant to capture.
    """

    def __init__(
        self,
        latency_ms: float,
        *,
        jitter_ms: float | None = None,
        buffered: bool = True,
        seed: int = 0,
    ) -> None:
        self.latency_s = max(latency_ms, 0.0) / 1000.0
        if jitter_ms is None:
            jitter_ms = 1.0 if latency_ms <= 10.0 else 0.20 * latency_ms
        self.jitter_s = max(jitter_ms, 0.0) / 1000.0
        self.buffered = buffered
        self.rng = random.Random(seed)
        self.counter = 0
        self.last_release_s = 0.0
        self.heap: list[DelayedItem] = []

    def push(self, t_s: float, item: Any) -> None:
        release_s = t_s + self.latency_s
        if self.buffered and self.latency_s > 0.0:
            release_s = math.ceil(release_s / self.latency_s) * self.latency_s
        if self.jitter_s > 0.0:
            release_s += max(0.0, self.rng.gauss(0.0, self.jitter_s))
        release_s = max(release_s, self.last_release_s)
        self.last_release_s = release_s
        heapq.heappush(self.heap, DelayedItem(release_s, self.counter, t_s, item))
        self.counter += 1

    def pop_ready(self, t_s: float) -> list[DeliveredItem]:
        ready: list[DeliveredItem] = []
        while self.heap and self.heap[0].release_s <= t_s:
            delayed = heapq.heappop(self.heap)
            ready.append(DeliveredItem(delayed.release_s, delayed.generated_s, delayed.item))
        return ready

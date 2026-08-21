#!/usr/bin/env python3
"""Recomputed planar Fixed-Physics baseline from wheel speed + IMU yaw rate.

This is a *recomputed sanity baseline*, not a replacement for the project's
frozen official Fixed Physics trajectories. For publication, prefer the frozen
official trajectories when they are available and use this implementation only
as a reproducible raw-input reference/check.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd

from .common import integrate_planar, standardized_output

METHOD_NAME = "Fixed Physics (recomputed)"


def run_fixed_physics(test_df: pd.DataFrame, speed_scale: float = 1.0, speed_bias: float = 0.0) -> pd.DataFrame:
    t = test_df["time_s"].to_numpy(float)
    v = speed_scale * test_df["odo_speed_mps"].to_numpy(float) + speed_bias
    w = test_df["imu_yaw_rate_radps"].to_numpy(float)
    init = (
        float(test_df["gt_east_m"].iloc[0]),
        float(test_df["gt_north_m"].iloc[0]),
        float(test_df["gt_heading_rad"].iloc[0]),
    )
    x, y, h = integrate_planar(t, v, w, init)
    return standardized_output(
        test_df, x, y, h, METHOD_NAME,
        corrected_v=v, corrected_omega=w,
        extra={"speed_scale": speed_scale, "speed_bias_mps": speed_bias},
    )

"""Segment a flight clip into hover / climb / cruise / coast and turn the useful
segments into observations.

Operates on velocity time-series (horizontal + vertical), which `track.py` is
responsible for recovering from pixels + scale. The segmentation logic here is
pure and testable on synthetic series; only the pixel->velocity step is blocked
on real footage + a tracker.

Mass comes ONLY from coast/cruise drag (spec §14): climb acceleration is
mass-independent and is used solely as a T/W consistency check, never for mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from augur.fusion.observations import Observation
from augur.physics import drag
from augur.types import FlightState


@dataclass
class Segment:
    state: FlightState
    start_idx: int
    end_idx: int
    mean_speed_m_s: float
    mean_accel_m_s2: float  # signed along horizontal motion


def segment(times: np.ndarray, horizontal_v: np.ndarray, vertical_v: np.ndarray,
            powered: np.ndarray | None = None, speed_eps: float = 0.5,
            accel_eps: float = 0.4, min_len: int = 3) -> list[Segment]:
    """Classify each frame, then merge consecutive same-state runs into segments.

    `powered` (optional bool per frame) distinguishes coast (power off + decel)
    from a powered slow-down. Without it, a sustained deceleration while moving is
    treated as a coast candidate.
    """
    times = np.asarray(times, float)
    hv = np.asarray(horizontal_v, float)
    vv = np.asarray(vertical_v, float)
    n = hv.size
    if not (times.size == vv.size == n) or n < 2:
        raise ValueError("times, horizontal_v, vertical_v must be equal length >= 2")

    dt = np.gradient(times)
    accel = np.gradient(hv) / np.where(dt == 0, np.nan, dt)
    accel = np.nan_to_num(accel)

    labels = []
    for i in range(n):
        is_powered = True if powered is None else bool(powered[i])
        if abs(vv[i]) > speed_eps and abs(vv[i]) > abs(hv[i]):
            labels.append(FlightState.climb)
        elif hv[i] < speed_eps and abs(vv[i]) < speed_eps:
            labels.append(FlightState.hover)
        elif accel[i] < -accel_eps and (not is_powered or powered is None):
            labels.append(FlightState.coast)
        elif abs(accel[i]) <= accel_eps and hv[i] > speed_eps:
            labels.append(FlightState.cruise)
        else:
            labels.append(FlightState.mixed)

    return _merge_runs(labels, times, hv, accel, min_len)


def _merge_runs(labels, times, hv, accel, min_len) -> list[Segment]:
    segments: list[Segment] = []
    i = 0
    n = len(labels)
    while i < n:
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        if (j - i + 1) >= min_len:
            sl = slice(i, j + 1)
            segments.append(Segment(
                state=labels[i], start_idx=i, end_idx=j,
                mean_speed_m_s=float(np.mean(hv[sl])),
                mean_accel_m_s2=float(np.mean(accel[sl])),
            ))
        i = j + 1
    return segments


def observations_from_segments(segments: list[Segment], frontal_area_m2: float,
                               rng: np.random.Generator | None = None) -> list[Observation]:
    """Build fusion observations from coast (mass) and cruise (thrust) segments."""
    obs: list[Observation] = []
    for seg in segments:
        if seg.state == FlightState.coast and seg.mean_accel_m_s2 < 0 and seg.mean_speed_m_s > 0:
            mass_dist = drag.coast_down_mass(abs(seg.mean_accel_m_s2), seg.mean_speed_m_s,
                                             frontal_area_m2, rng=rng)
            lo, hi = mass_dist.interval(0.9)
            obs.append(Observation(variable="mass_kg", value=mass_dist.median,
                                   sigma=max((hi - lo) / 3.29, 1e-3), source="video",
                                   note="coast-down"))
    return obs

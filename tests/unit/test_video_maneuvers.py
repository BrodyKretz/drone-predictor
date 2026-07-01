"""Maneuver segmentation on synthetic velocity series + track stub guard."""

import numpy as np
import pytest

from augur.types import FlightState
from augur.video import maneuvers, track


def test_segments_cruise_then_coast():
    t = np.linspace(0, 10, 100)
    # 0-5s steady cruise at 12 m/s, 5-10s power-off coast decelerating.
    hv = np.where(t < 5, 12.0, np.clip(12.0 - 3.0 * (t - 5), 0, None))
    vv = np.zeros_like(t)
    powered = t < 5
    segs = maneuvers.segment(t, hv, vv, powered=powered)
    states = {s.state for s in segs}
    assert FlightState.cruise in states
    assert FlightState.coast in states


def test_segments_climb():
    t = np.linspace(0, 5, 60)
    hv = np.zeros_like(t)
    vv = np.full_like(t, 3.0)  # steady vertical climb
    segs = maneuvers.segment(t, hv, vv)
    assert any(s.state == FlightState.climb for s in segs)


def test_segments_hover():
    t = np.linspace(0, 5, 60)
    hv = np.zeros_like(t)
    vv = np.zeros_like(t)
    segs = maneuvers.segment(t, hv, vv)
    assert any(s.state == FlightState.hover for s in segs)


def test_coast_produces_mass_observation():
    t = np.linspace(0, 6, 80)
    hv = np.clip(15.0 - 2.5 * t, 0, None)  # pure coast-down
    vv = np.zeros_like(t)
    powered = np.zeros_like(t, dtype=bool)
    segs = maneuvers.segment(t, hv, vv, powered=powered)
    obs = maneuvers.observations_from_segments(segs, frontal_area_m2=0.06,
                                               rng=np.random.default_rng(0))
    mass_obs = [o for o in obs if o.variable == "mass_kg"]
    assert mass_obs and mass_obs[0].value > 0 and mass_obs[0].source == "video"


def test_segment_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        maneuvers.segment(np.arange(5), np.arange(4), np.arange(5))


def test_track_requires_scale():
    # Tracking is wired now; without a metric scale it fails fast (see test_video_track).
    with pytest.raises(ValueError):
        track.track_drone("clip.mp4")

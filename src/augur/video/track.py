"""Detect + track the drone across video frames -> velocity, acceleration, tilt.

STATUS: blocked on (a) real flight footage with synchronized ground truth
(Betaflight blackbox / DJI log) and (b) a tracker dependency (opencv + norfair,
the `vision` extra), neither of which can be exercised without data. The pixel->
velocity recovery is therefore a stub.

The downstream consumer — maneuvers.segment / observations_from_segments — is
fully implemented and tested on synthetic velocity series, so once this returns
real (times, horizontal_v, vertical_v, tilt) arrays the video path is complete.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    times_s: np.ndarray
    horizontal_v_m_s: np.ndarray
    vertical_v_m_s: np.ndarray
    tilt_rad: np.ndarray


def track_drone(video_path: str, m_per_pixel: float | None = None) -> Track:
    """Recover the drone's velocity/tilt time-series from a clip. NOT YET WIRED."""
    raise NotImplementedError(
        "Video tracking is not wired yet. Install the 'vision' extra (opencv + "
        "norfair), detect/track the drone, convert pixel motion to metric velocity "
        "using scale metadata, and return a Track. Feed it to "
        "video.maneuvers.segment() (already implemented + tested)."
    )

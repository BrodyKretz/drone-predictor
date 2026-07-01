"""Detect + track the drone across video frames -> velocity, tilt time-series.

Split into pieces by what they depend on:

- `track_from_positions` / `_centroid_detector` / `track_positions_from_frames`
  are pure numpy — the actual kinematics and blob detection — and are unit-tested
  on synthetic pixel trajectories and frames.
- `track_drone` adds video *decode* via opencv (the `vision` extra). That layer
  is the only part that needs real footage to validate end-to-end; the logic it
  delegates to is already tested.

The output Track feeds `video.maneuvers.segment` (implemented + tested), which is
where coast-down mass and cruise thrust observations come from. Scale
(`m_per_pixel`) must be supplied — it comes from an image scale reference; without
it pixel motion can't be turned into metric velocity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from augur.config import load_priors

Position = tuple[float, float]  # (px, py) in pixels, y increasing downward


@dataclass
class Track:
    times_s: np.ndarray
    horizontal_v_m_s: np.ndarray
    vertical_v_m_s: np.ndarray
    tilt_rad: np.ndarray


def _centroid_detector(frame: np.ndarray, threshold: float = 127.0) -> Position | None:
    """Center of mass of the bright pixels in a frame (pure numpy).

    A deliberately simple detector for a light object on a darker background. Real
    footage will want a proper detector (background subtraction / a model); this
    is injectable via `detector` so it can be swapped without touching the
    kinematics."""
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    mask = gray >= threshold
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def track_from_positions(times: np.ndarray, px: np.ndarray, py: np.ndarray,
                         m_per_pixel: float) -> Track:
    """Convert a pixel-position time-series into metric velocities + tilt.

    Horizontal velocity is d(px)/dt scaled to metres; vertical velocity flips
    sign because image y grows downward (up is positive). Tilt is the physical
    hover proxy tilt ≈ atan(a_horizontal / g): in accelerated flight the thrust
    vector tilts to supply horizontal accel against gravity."""
    if m_per_pixel is None or m_per_pixel <= 0:
        raise ValueError("m_per_pixel must be a positive scale (metres per pixel)")

    times = np.asarray(times, float)
    px = np.asarray(px, float) * m_per_pixel
    py = np.asarray(py, float) * m_per_pixel
    if not (times.size == px.size == py.size) or times.size < 2:
        raise ValueError("times, px, py must be equal length >= 2")

    dt = np.gradient(times)
    dt = np.where(dt == 0, np.nan, dt)
    hv = np.nan_to_num(np.gradient(px) / dt)
    vv = np.nan_to_num(-np.gradient(py) / dt)

    g = load_priors().gravity
    a_h = np.nan_to_num(np.gradient(hv) / dt)
    tilt = np.arctan2(np.abs(a_h), g)
    return Track(times_s=times, horizontal_v_m_s=hv, vertical_v_m_s=vv, tilt_rad=tilt)


def track_positions_from_frames(frames: Iterable[np.ndarray], fps: float, m_per_pixel: float,
                                detector: Callable[[np.ndarray], Position | None] | None = None) -> Track:
    """Run a detector over frames, collect positions, and build a Track.

    Frames where the detector finds nothing are skipped (drone out of view /
    occluded), keeping the timestamps aligned to detected frames."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    detector = detector or _centroid_detector

    times, pxs, pys = [], [], []
    for idx, frame in enumerate(frames):
        pos = detector(frame)
        if pos is not None:
            times.append(idx / fps)
            pxs.append(pos[0])
            pys.append(pos[1])

    if len(times) < 2:
        raise ValueError("drone detected in fewer than 2 frames; cannot form a track")
    return track_from_positions(np.array(times), np.array(pxs), np.array(pys), m_per_pixel)


def track_drone(video_path: str, m_per_pixel: float | None = None,
                detector: Callable[[np.ndarray], Position | None] | None = None) -> Track:
    """Decode a clip with opencv, detect the drone per frame, return a Track.

    Requires the `vision` extra (opencv). Validated end-to-end only against real
    footage; the detection + kinematics it delegates to are covered by unit tests
    on synthetic frames."""
    if m_per_pixel is None:
        raise ValueError("m_per_pixel is required — recover scale from an image reference first")
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("video tracking needs the 'vision' extra: pip install -e '.[vision]'") from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _frames():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    return track_positions_from_frames(_frames(), fps=fps, m_per_pixel=m_per_pixel,
                                       detector=detector)

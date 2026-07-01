"""Video tracking kinematics + detection on synthetic data. The opencv decode
path is exercised only when the vision extra is installed."""

import numpy as np
import pytest

from augur.video import maneuvers, track


def test_track_from_positions_constant_horizontal_velocity():
    times = np.arange(20) * 0.1                 # 10 fps
    px = 100.0 + 10.0 * np.arange(20)           # 10 px/frame
    py = np.full(20, 50.0)                       # no vertical motion
    tr = track.track_from_positions(times, px, py, m_per_pixel=0.01)
    # 10 px/frame * 0.01 m/px / 0.1 s = 1.0 m/s horizontal, ~0 vertical.
    assert np.allclose(tr.horizontal_v_m_s, 1.0, atol=1e-6)
    assert np.allclose(tr.vertical_v_m_s, 0.0, atol=1e-6)


def test_track_from_positions_vertical_sign():
    times = np.arange(10) * 0.1
    px = np.full(10, 100.0)
    py = 200.0 - 5.0 * np.arange(10)   # pixel y decreasing => moving UP
    tr = track.track_from_positions(times, px, py, m_per_pixel=0.02)
    assert np.all(tr.vertical_v_m_s > 0)   # up is positive


def test_track_from_positions_requires_scale():
    times = np.arange(5) * 0.1
    with pytest.raises(ValueError):
        track.track_from_positions(times, np.arange(5), np.arange(5), m_per_pixel=None)


def test_track_from_positions_requires_length():
    with pytest.raises(ValueError):
        track.track_from_positions(np.array([0.0]), np.array([1.0]), np.array([1.0]), 0.01)


def test_centroid_detector_finds_blob():
    frame = np.zeros((100, 100), dtype=float)
    frame[40:50, 60:70] = 255.0     # bright square centered at (~64.5, 44.5)
    px, py = track._centroid_detector(frame)
    assert px == pytest.approx(64.5, abs=0.5)
    assert py == pytest.approx(44.5, abs=0.5)


def test_centroid_detector_returns_none_on_blank():
    assert track._centroid_detector(np.zeros((10, 10))) is None


def _moving_blob_frames(n=30, x0=20.0, vx=3.0, size=8):
    """A white square drifting right on a black background."""
    frames = []
    for i in range(n):
        f = np.zeros((120, 200), dtype=float)
        x = int(x0 + vx * i)
        f[56:56 + size, x:x + size] = 255.0
        frames.append(f)
    return frames


def test_track_positions_from_frames_recovers_motion():
    frames = _moving_blob_frames(vx=3.0)
    tr = track.track_positions_from_frames(frames, fps=10.0, m_per_pixel=0.05)
    # 3 px/frame * 0.05 m/px * 10 fps = 1.5 m/s.
    assert np.median(tr.horizontal_v_m_s) == pytest.approx(1.5, rel=0.1)


def test_track_positions_skips_undetected_frames():
    frames = _moving_blob_frames(n=5)
    frames.append(np.zeros((120, 200)))  # a blank frame is skipped, not an error
    tr = track.track_positions_from_frames(frames, fps=10.0, m_per_pixel=0.05)
    assert tr.times_s.size == 5


def test_track_feeds_maneuver_segmentation():
    """A recovered Track flows into segmentation without adaptation."""
    frames = _moving_blob_frames(n=40, vx=2.0)
    tr = track.track_positions_from_frames(frames, fps=20.0, m_per_pixel=0.05)
    segs = maneuvers.segment(tr.times_s, tr.horizontal_v_m_s, tr.vertical_v_m_s)
    assert segs  # produced at least one segment


def test_track_drone_requires_scale():
    with pytest.raises(ValueError):
        track.track_drone("clip.mp4", m_per_pixel=None)


# --- opencv decode path: only when the vision extra is present --------------


def test_track_drone_end_to_end(tmp_path):
    cv2 = pytest.importorskip("cv2", reason="vision extra not installed")
    path = tmp_path / "blob.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (200, 120))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this opencv build")
    for frame in _moving_blob_frames(n=25, vx=3.0):
        writer.write(np.repeat(frame[:, :, None], 3, axis=2).astype(np.uint8))
    writer.release()

    tr = track.track_drone(str(path), m_per_pixel=0.05)
    assert np.median(tr.horizontal_v_m_s) == pytest.approx(1.5, rel=0.25)

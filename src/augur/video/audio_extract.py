"""Demux a video's audio track so the acoustic pipeline can run on footage.

The counter-UAS case this serves: the camera never catches the drone (footage
of a wall, the sky, cover) but the microphone did. The extracted track feeds
the same audio -> RPM path as a plain WAV.

Uses the ffmpeg/ffprobe binaries via subprocess — no new Python dependency.
A missing binary is an environment problem the user can fix, so it raises;
a video that simply has no audio stream is a data property, so it returns
None and the caller degrades honestly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise RuntimeError(
            f"{binary} is required to extract audio from video — install ffmpeg "
            "(macOS: brew install ffmpeg) or pass the audio as --audio directly"
        )
    return path


def has_audio_stream(video_path: str | Path) -> bool:
    """True if ffprobe reports at least one audio stream in the container."""
    ffprobe = _require("ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def extract_audio(video_path: str | Path, out_wav: str | Path) -> Path | None:
    """Extract the audio track to a mono 16-bit WAV; None if there is none."""
    _require("ffmpeg")
    if not has_audio_stream(video_path):
        return None

    out = Path(out_wav)
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
         "-vn", "-acodec", "pcm_s16le", "-ac", "1", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {video_path}: {result.stderr.strip()}"
        )
    return out

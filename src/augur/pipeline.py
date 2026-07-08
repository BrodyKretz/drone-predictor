"""Glue: turn raw inputs (audio file, verbal JSON, video) into a FusionInputs
bundle and run the fusion. A video contributes its audio track (demuxed via
ffmpeg) when no explicit audio is given — footage that never shows the drone
still yields the full acoustic analysis. Visual tracking wiring is stubbed
until real footage validates it."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from augur.audio import hover, rpm, spectral
from augur.fusion.posterior import FusionInputs, fuse
from augur.types import FlightState, PredictionReport, VerbalSpec
from augur.video import audio_extract


def audio_to_inputs(audio_path: str | Path, num_blades_hint: int | None = None,
                    rng: np.random.Generator | None = None) -> tuple:
    """Load audio, estimate RPM + flight state. Returns (rpm_estimate, state)."""
    samples, sr = spectral.load_wav(audio_path)
    state = hover.classify_state(samples, sr)
    freqs, spec_db = spectral.average_spectrum(samples, sr)
    est = rpm.estimate_rpm(freqs, spec_db, num_blades=num_blades_hint or 2, rng=rng)
    return est, state


def load_verbal(verbal_path: str | Path) -> VerbalSpec:
    with open(verbal_path) as f:
        return VerbalSpec(**json.load(f))


def predict(audio: str | Path | None = None, verbal: str | Path | None = None,
            image: list[str] | None = None, video: str | Path | None = None,
            n: int = 8000, seed: int = 0) -> PredictionReport:
    """Top-level prediction over any subset of inputs."""
    verbal_spec = load_verbal(verbal) if verbal else None
    blade_hint = verbal_spec.blade_count if verbal_spec else None

    rpm_est, state = (None, FlightState.unknown)
    extra_notes: list[str] = []
    if audio:
        rpm_est, state = audio_to_inputs(audio, num_blades_hint=blade_hint,
                                         rng=np.random.default_rng(seed))
    elif video:
        with tempfile.TemporaryDirectory() as tmp:
            wav = audio_extract.extract_audio(video, Path(tmp) / "video_audio.wav")
            if wav is not None:
                rpm_est, state = audio_to_inputs(wav, num_blades_hint=blade_hint,
                                                 rng=np.random.default_rng(seed))
                extra_notes.append(
                    "Acoustic analysis ran on the video's audio track — usable "
                    "even when the drone is never in frame.")
            else:
                extra_notes.append(
                    "Video has no audio track: no acoustic analysis. Visual "
                    "tracking is not yet wired, so the video contributed nothing.")

    # Visual tracking not yet wired — it will append video Observations here.
    inputs = FusionInputs(
        verbal=verbal_spec,
        rpm_estimate=rpm_est,
        flight_state=state,
        image_observations=[],
        video_observations=[],
    )
    report = fuse(inputs, n=n, seed=seed)
    report.notes.extend(extra_notes)
    return report

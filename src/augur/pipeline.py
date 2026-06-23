"""Glue: turn raw inputs (audio file, verbal JSON) into a FusionInputs bundle
and run the fusion. Image/video wiring is stubbed until those modules land."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from augur.audio import hover, rpm, spectral
from augur.fusion.posterior import FusionInputs, fuse
from augur.types import FlightState, PredictionReport, VerbalSpec


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
    if audio:
        rpm_est, state = audio_to_inputs(audio, num_blades_hint=blade_hint,
                                         rng=np.random.default_rng(seed))

    # Image/video modules not yet wired — they will append Observations here.
    inputs = FusionInputs(
        verbal=verbal_spec,
        rpm_estimate=rpm_est,
        flight_state=state,
        image_observations=[],
        video_observations=[],
    )
    return fuse(inputs, n=n, seed=seed)

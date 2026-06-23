"""End-to-end: simulate a drone -> write WAV -> run the full pipeline -> check
recovered properties land near truth (round-trip against the simulator oracle)."""

import json

import numpy as np
import pytest
from scipy.io import wavfile

from augur.physics.simulator import DroneSpec, FlightCondition, compute_truth, synth_waveform
from augur.pipeline import predict
from augur.report import render


def _spec():
    return DroneSpec(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0,
                     blade_count=2, c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8,
                     mass_kg=0.55, battery_wh=55.5, drone_class="racing")


def _hover_rpm(spec):
    """RPM at which sum(T) = m*g for this spec — a genuine hover point."""
    d_m = spec.prop_diameter_inch * 0.0254
    n_rev_s = np.sqrt(spec.mass_kg * 9.80665 /
                      (spec.num_motors * spec.c_t * 1.225 * d_m**4))
    return n_rev_s * 60.0


@pytest.fixture
def golden_clip(tmp_path):
    spec = _spec()
    cond = FlightCondition(rpm=_hover_rpm(spec), state="hover")
    sig, sr = synth_waveform(spec, cond, duration_s=2.0, snr_db=30.0,
                             rng=np.random.default_rng(0))
    wav = tmp_path / "drone.wav"
    pcm = np.int16(sig / np.max(np.abs(sig)) * 32767)
    wavfile.write(wav, sr, pcm)

    verbal = tmp_path / "spec.json"
    verbal.write_text(json.dumps({
        "num_motors": 4, "prop_diameter_inch": 5.0, "blade_count": 2,
        "drone_class": "racing",
    }))
    truth = compute_truth(spec, cond)
    return wav, verbal, truth


def test_round_trip_rpm(golden_clip):
    wav, verbal, truth = golden_clip
    report = predict(audio=wav, verbal=verbal, n=6000, seed=0)
    rpm = report.variables["rpm"]
    err = abs(rpm.median - truth.rpm) / truth.rpm
    assert err < 0.02, f"RPM round-trip error {err:.3%}"


def test_round_trip_mass_within_band(golden_clip):
    wav, verbal, truth = golden_clip
    report = predict(audio=wav, verbal=verbal, n=8000, seed=0)
    mass = report.variables["mass_kg"]
    # Sound-only mass is hard (C_T unknown): §10 target is 25-40% error.
    # The truth must at least fall inside the 90% interval.
    assert mass.interval_low <= truth.mass_kg <= mass.interval_high, (
        f"true mass {truth.mass_kg} outside [{mass.interval_low}, {mass.interval_high}]"
    )


def test_disk_loading_recovered(golden_clip):
    wav, verbal, truth = golden_clip
    report = predict(audio=wav, verbal=verbal, n=8000, seed=0)
    dl = report.variables["disk_loading_n_m2"]
    assert dl.interval_low <= truth.disk_loading_n_m2 <= dl.interval_high


def test_report_renders(golden_clip):
    wav, verbal, _ = golden_clip
    report = predict(audio=wav, verbal=verbal, n=2000, seed=0)
    text = render(report)
    assert "AUGUR" in text and "RPM" in text
    assert "audio" in text and "verbal" in text


def test_verbal_only_runs(golden_clip):
    _, verbal, _ = golden_clip
    report = predict(verbal=verbal, n=2000, seed=0)
    assert "rpm" in report.variables
    # No audio -> RPM should be wide/unconstrained.
    assert report.variables["rpm"].confidence in {"low", "unconstrained", "medium"}

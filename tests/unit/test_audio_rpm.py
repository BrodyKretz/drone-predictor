"""Audio RPM recovery on synthetic signals with known BPF (the §10 <2% target)."""

import numpy as np
import pytest

from augur.audio import hover, rpm, spectral
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform
from augur.types import FlightState


def make_spec(**kw) -> DroneSpec:
    base = dict(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0, blade_count=2,
                c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8, mass_kg=0.55,
                battery_wh=55.5, drone_class="racing")
    base.update(kw)
    return DroneSpec(**base)


@pytest.mark.parametrize("true_rpm", [9000.0, 15000.0, 21000.0])
def test_rpm_within_2pct_clean(true_rpm):
    spec = make_spec(blade_count=2)
    cond = FlightCondition(rpm=true_rpm, state="hover")
    sig, sr = synth_waveform(spec, cond, duration_s=2.0, snr_db=30.0,
                             rng=np.random.default_rng(0))
    freqs, spec_db = spectral.average_spectrum(sig, sr)
    est = rpm.estimate_rpm(freqs, spec_db, num_blades=2, rng=np.random.default_rng(0))
    err = abs(est.rpm.median - true_rpm) / true_rpm
    assert err < 0.02, f"RPM error {err:.3%} (got {est.rpm.median:.0f}, true {true_rpm})"
    assert est.confidence > 0.5


def test_rpm_within_2pct_noisy():
    spec = make_spec(blade_count=2)
    cond = FlightCondition(rpm=12000.0, state="hover")
    sig, sr = synth_waveform(spec, cond, duration_s=2.0, snr_db=5.0,
                             rng=np.random.default_rng(3))
    freqs, spec_db = spectral.average_spectrum(sig, sr)
    est = rpm.estimate_rpm(freqs, spec_db, num_blades=2, rng=np.random.default_rng(0))
    err = abs(est.rpm.median - 12000.0) / 12000.0
    assert err < 0.02


def test_three_blade_prop():
    spec = make_spec(blade_count=3)
    cond = FlightCondition(rpm=10000.0)
    sig, sr = synth_waveform(spec, cond, duration_s=2.0, snr_db=30.0,
                             rng=np.random.default_rng(1))
    freqs, spec_db = spectral.average_spectrum(sig, sr)
    est = rpm.estimate_rpm(freqs, spec_db, num_blades=3, rng=np.random.default_rng(0))
    assert abs(est.rpm.median - 10000.0) / 10000.0 < 0.02


def test_per_motor_spread_detected():
    spec = make_spec(blade_count=2)
    cond = FlightCondition(rpm=12000.0, rpm_spread_fraction=0.025)
    sig, sr = synth_waveform(spec, cond, duration_s=3.0, snr_db=40.0,
                             rng=np.random.default_rng(2))
    freqs, spec_db = spectral.average_spectrum(sig, sr, nperseg=16384)
    est = rpm.estimate_rpm(freqs, spec_db, num_blades=2, rng=np.random.default_rng(0))
    assert est.rpm_spread_fraction > 0.0


def test_rpm_distribution_is_distribution():
    spec = make_spec()
    cond = FlightCondition(rpm=14000.0)
    sig, sr = synth_waveform(spec, cond, duration_s=1.5, snr_db=25.0,
                             rng=np.random.default_rng(4))
    freqs, spec_db = spectral.average_spectrum(sig, sr)
    est = rpm.estimate_rpm(freqs, spec_db, num_blades=2)
    lo, hi = est.rpm.interval(0.9)
    assert lo < est.rpm.median < hi
    assert est.rpm.confidence_label in {"high", "medium"}


def test_hover_detected_on_steady_signal():
    spec = make_spec()
    cond = FlightCondition(rpm=12000.0, state="hover")
    sig, sr = synth_waveform(spec, cond, duration_s=2.0, snr_db=30.0,
                             rng=np.random.default_rng(5))
    assert hover.classify_state(sig, sr) == FlightState.hover


def test_climb_detected_on_ramp():
    spec = make_spec()
    cond = FlightCondition(rpm=12000.0, state="climb")
    sig, sr = synth_waveform(spec, cond, duration_s=2.0, snr_db=30.0,
                             rpm_drift_fraction=0.4, rng=np.random.default_rng(6))
    assert hover.classify_state(sig, sr) != FlightState.hover

"""Forward simulator: spectrum has correct tones; truth is self-consistent."""

import numpy as np
import pytest

from augur.physics import core
from augur.physics.simulator import DroneSpec, FlightCondition, compute_truth, simulate


def make_spec(**kw) -> DroneSpec:
    base = dict(
        num_motors=4,
        prop_diameter_inch=5.0,
        prop_pitch_inch=3.0,
        blade_count=2,
        c_t=0.11,
        c_p=0.05,
        pole_count=14,
        efficiency=0.8,
        mass_kg=0.55,
        battery_wh=55.5,
        drone_class="racing",
    )
    base.update(kw)
    return DroneSpec(**base)


def test_truth_thrust_to_weight_consistent():
    spec = make_spec()
    cond = FlightCondition(rpm=20000.0, state="hover")
    t = compute_truth(spec, cond)
    assert t.total_thrust_n == pytest.approx(t.thrust_per_motor_n * spec.num_motors)
    assert t.thrust_to_weight == pytest.approx(t.total_thrust_n / t.weight_n)
    assert t.weight_n == pytest.approx(spec.mass_kg * 9.80665)


def test_truth_bpf_matches_core():
    spec = make_spec(blade_count=3)
    cond = FlightCondition(rpm=12000.0)
    t = compute_truth(spec, cond)
    assert t.bpf_hz == pytest.approx(core.bpf_from_rpm(12000.0, 3))


def test_spectrum_peak_at_bpf():
    spec = make_spec(blade_count=2)
    cond = FlightCondition(rpm=12000.0, state="hover")
    # No spread, no Doppler -> clean fundamental at BPF = 12000/60*2 = 400 Hz.
    sample = simulate(spec, cond, sample_rate=44100, duration_s=1.0, noise_db=-120.0,
                      rng=np.random.default_rng(0))
    freqs, spec_db = sample.freqs_hz, sample.spectrum_db
    bpf = 400.0
    # Find the strongest tone below 1 kHz; it should be the fundamental BPF.
    band = freqs < 1000.0
    peak_freq = freqs[band][np.argmax(spec_db[band])]
    assert peak_freq == pytest.approx(bpf, abs=2.0)


def test_spectrum_has_harmonics():
    spec = make_spec(blade_count=2)
    cond = FlightCondition(rpm=12000.0)
    sample = simulate(spec, cond, duration_s=1.0, noise_db=-120.0, n_harmonics=4,
                      rng=np.random.default_rng(1))
    spec_db = sample.spectrum_db
    bpf = 400.0
    for h in (1, 2, 3, 4):
        idx = int(round(bpf * h))
        local = spec_db[max(0, idx - 2): idx + 3]
        floor = np.median(spec_db)
        assert local.max() > floor + 6.0  # harmonic stands clearly above floor


def test_doppler_shifts_tone_up_on_approach():
    spec = make_spec(blade_count=2)
    rng = np.random.default_rng(2)
    still = simulate(spec, FlightCondition(rpm=12000.0, forward_velocity_m_s=0.0),
                     noise_db=-120.0, rng=rng)
    approaching = simulate(spec, FlightCondition(rpm=12000.0, forward_velocity_m_s=30.0),
                           noise_db=-120.0, rng=rng)

    def peak(sample):
        f, s = sample.freqs_hz, sample.spectrum_db
        band = f < 1000.0
        return f[band][np.argmax(s[band])]

    assert peak(approaching) > peak(still)


def test_endurance_positive_and_finite():
    spec = make_spec()
    t = compute_truth(spec, FlightCondition(rpm=18000.0))
    assert 0 < t.endurance_s < 1e5

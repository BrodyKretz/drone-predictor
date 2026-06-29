"""Property-based round-trip tests (spec §9).

The example-based tests pin specific operating points; these sweep randomized
inputs through the forward model and its exact identities, then back out through
the inverse path, asserting the invariants hold everywhere — not just at the
points we happened to pick.
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from augur.audio import rpm as rpm_mod
from augur.audio import spectral
from augur.physics import core
from augur.physics.simulator import DroneSpec, FlightCondition, compute_truth, synth_waveform

RHO = 1.225
G = 9.80665


# --- exact algebraic identities ---------------------------------------------


@given(rpm=st.floats(500, 30000), blades=st.integers(1, 5))
def test_bpf_rpm_round_trip(rpm, blades):
    bpf = core.bpf_from_rpm(rpm, blades)
    np.testing.assert_allclose(float(core.rpm_from_bpf(bpf, blades)), rpm, rtol=1e-9)


@given(
    n1=st.floats(1000, 30000),
    n2=st.floats(1000, 30000),
    diameter_m=st.floats(0.05, 0.8),
    c_t=st.floats(0.05, 0.2),
)
def test_thrust_ratio_cancels_coefficients(n1, n2, diameter_m, c_t):
    """T2/T1 = (n2/n1)^2 regardless of C_T, rho, D — the basis for T/W from RPM."""
    t1 = float(core.thrust_per_motor(n1, diameter_m, c_t, RHO))
    t2 = float(core.thrust_per_motor(n2, diameter_m, c_t, RHO))
    np.testing.assert_allclose(t2 / t1, (n2 / n1) ** 2, rtol=1e-9)


@given(
    mass=st.floats(0.3, 25.0),
    diameter_inch=st.floats(4.0, 24.0),
    num_motors=st.sampled_from([4, 6, 8]),
    c_t=st.floats(0.08, 0.14),
)
def test_hover_force_balance_round_trip(mass, diameter_inch, num_motors, c_t):
    """Put the drone at its true hover RPM, then the inverse hover relation must
    recover the mass and give T/W = 1."""
    d = diameter_inch * core.INCH_TO_M
    n_rev_s = np.sqrt(mass * G / (num_motors * c_t * RHO * d**4))
    rpm = n_rev_s * 60.0

    t_per = core.thrust_per_motor(rpm, d, c_t, RHO)
    t_total = float(core.total_thrust(t_per, num_motors))

    np.testing.assert_allclose(float(core.hover_mass(t_total, G)), mass, rtol=1e-9)
    np.testing.assert_allclose(t_total / (mass * G), 1.0, rtol=1e-9)


@given(
    t_per=st.floats(0.5, 200.0),
    num_motors=st.sampled_from([3, 4, 6, 8]),
    diameter_m=st.floats(0.05, 0.8),
)
def test_disk_loading_is_thrust_over_swept_area(t_per, num_motors, diameter_m):
    total = core.total_thrust(t_per, num_motors)
    dl = float(core.disk_loading(total, num_motors, diameter_m))
    expected = (t_per * num_motors) / (num_motors * np.pi * (diameter_m / 2) ** 2)
    assert dl > 0
    np.testing.assert_allclose(dl, expected, rtol=1e-9)


@given(diameter_m=st.floats(0.05, 0.8), num_motors=st.integers(2, 8))
def test_frame_diagonal_fits_props(diameter_m, num_motors):
    """Adjacent props mustn't overlap, so the min diagonal is always >= one prop
    diameter (equality only at N=2)."""
    diag = float(core.min_frame_diagonal_m(diameter_m, num_motors))
    assert diag >= diameter_m - 1e-9


@given(
    battery_wh=st.floats(20, 400),
    extra_wh=st.floats(1, 200),
    rpm=st.floats(4000, 20000),
)
def test_endurance_monotonic_in_battery(battery_wh, extra_wh, rpm):
    """More stored energy at a fixed operating point can only extend endurance."""
    base = _spec(battery_wh=battery_wh)
    more = _spec(battery_wh=battery_wh + extra_wh)
    cond = FlightCondition(rpm=rpm, state="hover")
    assert compute_truth(more, cond).endurance_s > compute_truth(base, cond).endurance_s


# --- forward simulator -> inverse audio recovery ----------------------------


@settings(max_examples=30, deadline=None)
@given(rpm=st.floats(3000, 24000), blades=st.integers(2, 3))
def test_audio_rpm_recovered_within_tolerance(rpm, blades):
    """Synthesize a hover clip, run the acoustic RPM estimator, recover RPM.

    Constrain the blade-pass fundamental to a band where it and its first
    several harmonics sit inside the estimator's search window; outside that the
    forward model itself can't represent the tone faithfully (Nyquist/aliasing),
    so it isn't a meaningful round-trip case.
    """
    bpf = rpm / 60.0 * blades
    assume(120.0 <= bpf <= 800.0)

    spec = _spec(blade_count=blades)
    cond = FlightCondition(rpm=rpm, state="hover")
    seed = int(rpm) % (2**32)  # deterministic per example -> reproducible failures
    sig, sr = synth_waveform(spec, cond, duration_s=2.0, snr_db=25.0,
                             rng=np.random.default_rng(seed))
    freqs, spec_db = spectral.average_spectrum(sig, sr)
    est = rpm_mod.estimate_rpm(freqs, spec_db, num_blades=blades,
                               rng=np.random.default_rng(seed))

    err = abs(est.rpm.median - rpm) / rpm
    assert err < 0.03, f"RPM error {err:.3%} (got {est.rpm.median:.0f}, true {rpm:.0f})"


def _spec(**kw) -> DroneSpec:
    base = dict(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0, blade_count=2,
                c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8, mass_kg=1.0,
                battery_wh=60.0, drone_class="racing")
    base.update(kw)
    return DroneSpec(**base)

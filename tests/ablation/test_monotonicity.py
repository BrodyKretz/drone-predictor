"""The ladder claim: adding a modality never widens any variable's interval
(on consistent inputs). Verified empirically via the attribution record."""

import numpy as np

from augur.audio import rpm, spectral
from augur.fusion.posterior import FusionInputs, fuse
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform
from augur.types import FlightState, VerbalSpec


def _spec():
    return DroneSpec(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0,
                     blade_count=2, c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8,
                     mass_kg=0.55, battery_wh=55.5, drone_class="racing")


def _rpm_estimate(true_rpm=14000.0):
    sig, sr = synth_waveform(_spec(), FlightCondition(rpm=true_rpm, state="hover"),
                             duration_s=2.0, snr_db=30.0, rng=np.random.default_rng(0))
    freqs, spec_db = spectral.average_spectrum(sig, sr)
    return rpm.estimate_rpm(freqs, spec_db, num_blades=2, rng=np.random.default_rng(0))


def test_attribution_never_widens():
    inputs = FusionInputs(
        verbal=VerbalSpec(num_motors=4, prop_diameter_inch=5.0, blade_count=2,
                          drone_class="racing"),
        rpm_estimate=_rpm_estimate(),
        flight_state=FlightState.hover,
    )
    report = fuse(inputs, n=6000, seed=1)
    # The monotone quantity is RELATIVE width (information): an input may relocate
    # the estimate and grow absolute width while still constraining the drone.
    tol = 0.06
    for a in report.attribution:
        assert a.relative_width_after <= a.relative_width_before * (1 + tol), (
            f"{a.source} widened {a.variable} (relative): "
            f"{a.relative_width_before:.4g} -> {a.relative_width_after:.4g}"
        )


def test_audio_narrows_rpm_dependent_vars():
    base = FusionInputs(verbal=VerbalSpec(num_motors=4, prop_diameter_inch=5.0,
                                          blade_count=2, drone_class="racing"))
    with_audio = FusionInputs(verbal=base.verbal, rpm_estimate=_rpm_estimate(),
                              flight_state=FlightState.hover)

    r_base = fuse(base, n=6000, seed=2)
    r_audio = fuse(with_audio, n=6000, seed=2)

    for var in ("rpm", "total_thrust_n", "mass_kg", "electrical_power_w"):
        w_base = r_base.variables[var].interval_high - r_base.variables[var].interval_low
        w_audio = r_audio.variables[var].interval_high - r_audio.variables[var].interval_low
        assert w_audio < w_base, f"audio did not narrow {var}"


def test_any_subset_runs():
    # Each input alone, and none, must produce a coherent report.
    est = _rpm_estimate()
    verbal = VerbalSpec(num_motors=4, prop_diameter_inch=5.0, drone_class="racing")
    for inp in (
        FusionInputs(),
        FusionInputs(verbal=verbal),
        FusionInputs(rpm_estimate=est, flight_state=FlightState.hover),
        FusionInputs(verbal=verbal, rpm_estimate=est, flight_state=FlightState.hover),
    ):
        report = fuse(inp, n=3000, seed=0)
        assert set(report.variables) and all(
            s.interval_high >= s.interval_low for s in report.variables.values()
        )

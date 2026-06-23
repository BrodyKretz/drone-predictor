"""Generate a self-consistent synthetic drone clip + verbal spec for demos/tests.

Writes data/demo/drone.wav and data/demo/spec.json. The hover RPM is solved so
sum(T) = m*g, so the recovered mass should match the spec's true mass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from augur.physics.simulator import DroneSpec, FlightCondition, compute_truth, synth_waveform


def hover_rpm(spec: DroneSpec, rho: float = 1.225, g: float = 9.80665) -> float:
    n_rev_s = np.sqrt(spec.mass_kg * g / (spec.num_motors * spec.c_t * rho * spec.diameter_m**4))
    return n_rev_s * 60.0


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "demo"
    out.mkdir(parents=True, exist_ok=True)

    spec = DroneSpec(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0,
                     blade_count=2, c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8,
                     mass_kg=0.55, battery_wh=55.5, drone_class="racing")
    cond = FlightCondition(rpm=hover_rpm(spec), rpm_spread_fraction=0.015, state="hover")

    sig, sr = synth_waveform(spec, cond, duration_s=3.0, snr_db=25.0,
                             rng=np.random.default_rng(0))
    pcm = np.int16(sig / np.max(np.abs(sig)) * 32767)
    wavfile.write(out / "drone.wav", sr, pcm)

    (out / "spec.json").write_text(json.dumps({
        "num_motors": 4, "prop_diameter_inch": 5.0, "blade_count": 2,
        "drone_class": "racing",
    }, indent=2))

    truth = compute_truth(spec, cond)
    print(f"wrote {out}/drone.wav and spec.json")
    print(f"true hover RPM = {cond.rpm:.0f}, true mass = {truth.mass_kg} kg, "
          f"T/W = {truth.thrust_to_weight:.2f}")


if __name__ == "__main__":
    main()

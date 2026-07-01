"""Generate labeled demo/template clips for the web UI from the simulator.

Each template is a real hover point for a plausible drone of its class (RPM solved
so sum(T)=m*g), synthesized to a WAV, plus a manifest entry carrying the matching
verbal spec and the true RPM. The UI loads these so a visitor can try the pipeline
without recording anything. Synthetic (our own data) — no third-party license.

Usage: python scripts/make_templates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from augur.physics.simulator import DroneSpec, FlightCondition, compute_truth, synth_waveform

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web" / "public" / "templates"

RHO, G = 1.225, 9.80665

# (id, label, spec kwargs). Pitch/coeffs are class-typical; mass sets the hover RPM.
TEMPLATES = [
    ("racing", "Racing quad (5\", 4 motors)",
     dict(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0, blade_count=2,
          c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.80, mass_kg=0.55,
          battery_wh=55.5, drone_class="racing")),
    ("cinematic", "Cinematic quad (9\", 4 motors)",
     dict(num_motors=4, prop_diameter_inch=9.0, prop_pitch_inch=4.5, blade_count=2,
          c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.80, mass_kg=1.2,
          battery_wh=100.0, drone_class="cinematic")),
    ("survey", "Survey hex (15\", 6 motors)",
     dict(num_motors=6, prop_diameter_inch=15.0, prop_pitch_inch=5.0, blade_count=2,
          c_t=0.12, c_p=0.06, pole_count=14, efficiency=0.80, mass_kg=3.5,
          battery_wh=220.0, drone_class="survey")),
]


def hover_rpm(spec: DroneSpec) -> float:
    d_m = spec.prop_diameter_inch * 0.0254
    n_rev_s = np.sqrt(spec.mass_kg * G / (spec.num_motors * spec.c_t * RHO * d_m**4))
    return float(n_rev_s * 60.0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for tid, label, kw in TEMPLATES:
        spec = DroneSpec(**kw)
        cond = FlightCondition(rpm=hover_rpm(spec), state="hover")
        sig, sr = synth_waveform(spec, cond, duration_s=2.0, snr_db=25.0,
                                 rng=np.random.default_rng(0))
        pcm = np.int16(sig / np.max(np.abs(sig)) * 32767)
        wavfile.write(OUT_DIR / f"{tid}.wav", sr, pcm)

        truth = compute_truth(spec, cond)
        manifest.append({
            "id": tid,
            "label": label,
            "file": f"{tid}.wav",
            "spec": {
                "drone_class": spec.drone_class,
                "num_motors": spec.num_motors,
                "blade_count": spec.blade_count,
                "prop_diameter_inch": spec.prop_diameter_inch,
            },
            "true_rpm": round(truth.rpm),
        })

    (OUT_DIR / "templates.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} templates to {OUT_DIR}")
    for m in manifest:
        print(f"  {m['id']:10} true RPM {m['true_rpm']}")


if __name__ == "__main__":
    main()

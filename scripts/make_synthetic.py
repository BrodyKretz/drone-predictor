"""Generate labeled synthetic samples from the forward simulator.

Sweeps drone specs + flight states, writes WAVs to data/audio/ and appends rows
to data/manifest.parquet with full ground truth. This is the infinite labeled-
data source for training/augmenting the RPM detector and any ML layer.

Usage: python scripts/make_synthetic.py --n 50
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from augur.data_manifest import empty_manifest, save_manifest
from augur.physics.simulator import DroneSpec, FlightCondition, compute_truth, synth_waveform

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "audio"
MANIFEST = ROOT / "data" / "manifest.parquet"

CLASS_RANGES = {
    "racing": dict(diam=(4, 6), motors=[4], mass=(0.25, 0.9), rpm=(15000, 28000)),
    "cinematic": dict(diam=(6, 15), motors=[4, 6], mass=(0.5, 4.0), rpm=(6000, 14000)),
    "survey": dict(diam=(13, 30), motors=[4, 6, 8], mass=(1.5, 25.0), rpm=(3000, 8000)),
}


def random_spec(rng: np.random.Generator) -> tuple[DroneSpec, str]:
    cls = rng.choice(list(CLASS_RANGES))
    r = CLASS_RANGES[cls]
    return DroneSpec(
        num_motors=int(rng.choice(r["motors"])),
        prop_diameter_inch=float(rng.uniform(*r["diam"])),
        prop_pitch_inch=float(rng.uniform(*r["diam"]) * 0.5),
        blade_count=int(rng.choice([2, 3])),
        c_t=float(rng.uniform(0.09, 0.13)),
        c_p=float(rng.uniform(0.045, 0.07)),
        pole_count=14,
        efficiency=float(rng.uniform(0.72, 0.83)),
        mass_kg=float(rng.uniform(*r["mass"])),
        battery_wh=float(rng.uniform(20, 300)),
        drone_class=cls,
    ), cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = empty_manifest()

    for _ in range(args.n):
        spec, cls = random_spec(rng)
        r = CLASS_RANGES[cls]
        cond = FlightCondition(rpm=float(rng.uniform(*r["rpm"])),
                               rpm_spread_fraction=float(rng.uniform(0, 0.03)),
                               state="hover")
        sig, sr = synth_waveform(spec, cond, duration_s=2.0,
                                 snr_db=float(rng.uniform(5, 30)), rng=rng)
        sid = f"{int(rng.integers(0, 2**48)):012x}"
        wav_path = AUDIO_DIR / f"{sid}.wav"
        wavfile.write(wav_path, sr, np.int16(sig / np.max(np.abs(sig)) * 32767))

        truth = compute_truth(spec, cond)
        df.loc[len(df)] = {
            "sample_id": sid, "drone_id": f"synth_{sid}",
            "audio_path": str(wav_path.relative_to(ROOT)), "image_paths": None,
            "video_path": None, "verbal": {"num_motors": spec.num_motors,
                                           "prop_diameter_inch": spec.prop_diameter_inch,
                                           "blade_count": spec.blade_count,
                                           "drone_class": cls},
            "flight_state": "hover", "truth": asdict(truth), "source": "synthetic",
            "split": "train", "license": "self",
        }

    existing = pd.read_parquet(MANIFEST) if MANIFEST.exists() else empty_manifest()
    save_manifest(pd.concat([existing, df], ignore_index=True), MANIFEST)
    print(f"wrote {args.n} synthetic samples to {AUDIO_DIR} and updated {MANIFEST}")


if __name__ == "__main__":
    main()

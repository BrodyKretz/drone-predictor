"""Validate the acoustic RPM recovery on a real (or any) WAV clip.

Point it at a recording and, if you know the ground-truth RPM (e.g. from a
dataset's motor-speed log), get the recovered-vs-true error. Works on DREGON
clips, DroneAudioset, your own recordings, or our synthetic templates.

Getting real data + finding its true RPM: see docs/REAL_AUDIO.md. Mind the
licenses — DREGON is academic/educational use only; do NOT commit its files to
this (public) repo. This script reads local files and commits nothing.

Usage:
  python scripts/validate_real_audio.py --wav clip.wav --blades 2 --true-rpm 4800
"""

from __future__ import annotations

import argparse

import numpy as np

from augur.pipeline import audio_to_inputs


def main():
    ap = argparse.ArgumentParser(description="Recover RPM from a WAV and (optionally) score it.")
    ap.add_argument("--wav", required=True, help="Path to a WAV recording")
    ap.add_argument("--blades", type=int, default=2, help="Blades per prop (sets BPF→RPM)")
    ap.add_argument("--true-rpm", type=float, default=None, help="Ground-truth RPM, if known")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    est, state = audio_to_inputs(args.wav, num_blades_hint=args.blades,
                                 rng=np.random.default_rng(args.seed))
    lo, hi = est.rpm.interval(0.9)

    print(f"clip:          {args.wav}")
    print(f"flight state:  {state.value}")
    print(f"fundamental:   {est.fundamental_hz:.1f} Hz  (comb score {est.confidence:.2f})")
    print(f"recovered RPM: {est.rpm.median:.0f}  [90%: {lo:.0f}–{hi:.0f}]  "
          f"<{est.rpm.confidence_label}>")
    if args.true_rpm is not None:
        err = abs(est.rpm.median - args.true_rpm) / args.true_rpm
        print(f"true RPM:      {args.true_rpm:.0f}")
        print(f"error:         {err * 100:.1f}%")


if __name__ == "__main__":
    main()

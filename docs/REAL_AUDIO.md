# Validating on real drone audio

The bundled demo clips (`web/public/templates/`) are synthetic — clean, with known
truth. To check the acoustic RPM recovery against **real** recordings, use
`scripts/validate_real_audio.py`. It reads local files and commits nothing, so you
can point it at licensed datasets without re-hosting them.

```bash
python scripts/validate_real_audio.py --wav <clip>.wav --blades <n> --true-rpm <rpm>
```

`--true-rpm` is optional; supply it (from a dataset's motor-speed log) to get a
recovered-vs-true error. `--blades` sets the BPF→RPM conversion and must match the
prop.

## Datasets

### DREGON (best for RPM validation) — ⚠️ do not commit
On-board quadrotor audio (8-channel, 44.1 kHz) **with per-rotor motor-speed logs**
and VICON ground truth — ideal for checking recovered vs. true RPM.

- **License: academic/educational use only, no redistribution grant.** Use it
  locally; do **not** add its files to this public repo.
- Site: https://dregon.inria.fr/datasets/dregon/ (TLS cert may be expired — proceed
  via the browser).
- To use: download a recording zip to a gitignored folder (e.g. `data/external/`),
  take one audio channel (or average to mono), read the mean rotor speed over the
  clip from the motor-speed `.mat` log as `--true-rpm`, and note the prop blade
  count for `--blades`.

### DroneAudioset (safe to bundle) — MIT
23.5 h of drone audio, MIT-licensed, so it *may* be committed if you want real
clips in-repo. Caveat: search-and-rescue audio — far-field and very low SNR, so
RPM recovery will be rough and may undersell the pipeline.

- https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/

### Others
- **RWDA** (IEEE DataPort) — DJI Air 3S / Mini 4K, 48 kHz; usually needs an account.
- **Sara Al-Emadi DroneAudioDataset** (GitHub) — small, indoor prop noise.

## Expectations

Recovery is cleanest on steady, near-field hover audio with a clear blade-pass
tone (what DREGON's on-board mics capture). Far-field or heavily-mixed clips will
have lower comb-score confidence and wider intervals — which the report reflects
honestly rather than hiding.

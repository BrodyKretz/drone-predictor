# Augur — Project Plan & Status

Living status doc. Full spec: [`docs/SPEC.md`](docs/SPEC.md). Build agreements:
spec §0. **Update the status column whenever a phase advances.**

_Last updated: 2026-10-18_

## Where this stands right now

A working, tested **sound + verbal → calibrated distributions** pipeline exists
end-to-end (Phases 0–4 + the testable physics from later phases), exposed via both
a CLI and a FastAPI service, with a property-based round-trip suite over the
physics and audio recovery. Image/video/calibration are now implemented and
tested (VLM call via injected fake client, video tracker on synthetic frames,
eval harness on synthetic reports) — not stubs. The UIUC prop data is downloaded
and ingested (250 props → `config/prop_db.parquet`) and the fusion posterior now
draws measured C_T/C_P when the verbal spec asserts a prop diameter. The
remaining blockers are purely external: golden-set data / API key / hardware
(see "Blocked on you"). 225 tests pass, plus 33 web (node:test) tests.

Run it:
```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python scripts/make_demo_sample.py
./.venv/bin/augur predict --audio data/demo/drone.wav --verbal data/demo/spec.json
./.venv/bin/pytest tests/ -q          # 225 tests
```

The HTTP API needs the serve extra: `./.venv/bin/pip install -e ".[serve]"`, then
`./.venv/bin/augur serve` (POST multipart audio/verbal to `/predict`, one audio
window to `/identify`; `/health` for a liveness check). Without the extra the
API tests skip; the rest still run.

## Phase status

| phase | what | status |
|---|---|---|
| 0 | Scaffold: pyproject, types (`Distribution`+pydantic), config loader, manifest schema, CI | ✅ done |
| 1 | Forward simulator (`physics/simulator.py`) — spectrum + waveform + truth | ✅ done |
| 2 | Audio → RPM (`audio/spectral,rpm,hover`) — <2% RPM incl. noisy | ✅ done |
| 3 | Physics core + priors + verbal + MC uncertainty + round-trip | ✅ done |
| 4 | Fusion skeleton (`fusion/observations,posterior`) + per-modality attribution + monotonicity | ✅ done |
| 5 | Image module: VLM extraction + geometry | 🟡 near-complete — geometry + JSON parser + **live VLM call all done+tested (fake-client)**; needs one real API run to confirm |
| 6 | Video module: tracking + maneuvers + coast-down mass | 🟡 near-complete — segmentation + drag mass + **pixel→velocity tracking done+tested**; needs real footage to validate end-to-end |
| 7 | Calibration + eval harness + hardening | 🟡 partial — conformal calibrator + **eval harness (splits/metrics/fit) done+tested**; needs the golden set to actually fit + produce the §10 table |

## What's implemented and verified

- **Physics core** (`physics/core.py`): thrust, power, disk loading, T/W from RPM
  ratio, hover mass, frame geometry. Hand-computed unit tests + the C_T-cancels
  identity.
- **Forward simulator** (`physics/simulator.py`): truth + synthetic spectrum +
  time-domain waveform (BPF harmonics, commutation tone, per-motor spread,
  Doppler, noise). The test oracle.
- **Audio** (`audio/`): STFT front-end, BPF comb-detection RPM (<2% at 5 dB SNR),
  per-motor spread, robust hover/climb classification locked to the fundamental.
- **Fusion** (`fusion/posterior.py`): common-random-number Monte Carlo, likelihood
  weighting of soft observations, cumulative per-modality attribution. Monotonicity
  enforced in **relative** width (the information measure — see note below).
- **Drag** (`physics/drag.py`): coast-down mass + cruise absolute-thrust,
  round-tripped against the forward drag relation.
- **Image geometry** (`image/geometry.py`): pixel→metric with scale; wide
  ratio-only prior when scale absent.
- **Image VLM** (`image/extract.py`): defensive JSON parse + observation builder,
  and the live Anthropic Messages call (`extract_from_image`) with an injectable
  client — tested via a fake client (no key/network).
- **Video tracking** (`video/track.py`): pixel→metric velocity/tilt kinematics +
  a numpy blob detector + opencv decode; tested on synthetic frames and a real
  encode/decode round-trip. Feeds maneuver segmentation.
- **Maneuver segmentation** (`video/maneuvers.py`): hover/climb/cruise/coast from
  velocity series → coast mass observations.
- **Audio-from-video** (`video/audio_extract.py`): when a video is given without
  explicit audio, its audio track is demuxed (ffmpeg subprocess, no new Python
  dep) and fed through the acoustic pipeline — footage that never shows the
  drone still yields the full RPM analysis (the hiding-observer scenario).
  Explicit `--audio` always wins to avoid double-counting the same sound.
  Silent videos degrade with an honest note. Tested end-to-end with
  ffmpeg-synthesized fixtures (AAC round-trip recovers RPM within 3%).
- **Eval harness** (`eval.py`): drone-level leakage-free split assignment,
  §10 metrics (MAPE / coverage / sharpness), and conformal-calibrator fitting
  from a calib split. Capture protocol in `docs/GOLDEN_SET.md`.
- **Conformal calibration** (`fusion/calibration.py`): split-conformal width
  factor; corrects overconfident intervals to nominal coverage on a holdout.
- **Reliability diagrams** (`eval.reliability_curve` + `viz.plot_reliability`,
  viz extra): empirical coverage vs nominal level from raw posterior samples
  (`fuse_distributions`), one curve per variable. Verified on synthetic drones —
  it correctly exposes the pre-calibration overconfidence the conformal step
  will fix.
- **Manifest** (`data_manifest.py`): schema + round-trip + drone-level leakage
  guard.
- **CLI** (`augur predict`, `augur version`, `augur serve`), report renderer,
  synthetic data gen.
- **HTTP API** (`api.py`): FastAPI `/predict` (multipart audio + verbal), `/identify`
  (one audio window + optional known facts), `/reference` (the no-input 99% bands + class
  descriptions) and `/health`, dev-permissive CORS for the web UI. Core logic is in
  framework-free helpers (`predict_from_uploads`, `identify_from_upload`,
  `reference_info`) so it unit-tests without FastAPI; HTTP routes tested via
  TestClient when the serve extra is present.
- **Live identification** (`fusion/identify.py`, `POST /identify`, `web/`): hover-physics
  evidence per (class, blade count) from the `classes.yaml` mass/prop bands, mixed into
  class-weighted estimates; asserted prop size / motor count add their likelihood. On 48
  simulated drones it is right 54% of the time (chance 25%) with mean confidence 50% —
  calibrated, but military is nearly unidentifiable from audio (its envelope overlaps
  every class). The web UI is a single live screen (file, sample or mic → class guess +
  interval estimates with sparklines). Frame diagonal and disk loading now carry
  motor-count uncertainty instead of using the modal count.
- **Real-audio validation** (`scripts/validate_real_audio.py`, `docs/REAL_AUDIO.md`):
  recover RPM from any WAV and score vs. a known true RPM. DREGON is the ideal
  real dataset (on-board audio + rotor-speed truth) but is **academic-use only —
  not redistributable**, so it's run locally (gitignored `data/external/`), never
  committed. DroneAudioset (MIT) is the bundle-able but noisier alternative.
- **Prop-DB ingest** (`prop_ingest.py` + `scripts/ingest_uiuc.py`): parses UIUC/APC
  static-test files (`<maker>_<D>x<P>_static_*.txt`) into one representative
  (C_T, C_P, blade_count) per test file → `config/prop_db.parquet` (250 props,
  UIUC vols 1–4, downloaded 2026-07-07 — provenance + naming-quirk normalization
  in `data/public/SOURCES.md`). The fusion posterior (`posterior.py`) draws
  coefficients from matched props (diameter ±0.75", pitch ±1" and exact blade
  count when asserted) via a CRN-aligned stream, falling back to the config
  bands when nothing matches; the report's assumptions block says which path ran.
  **Honesty note:** with pitch unasserted, measured 5" props span C_T 0.054–0.18,
  so intervals got *wider* than the old hand-set fallback band [0.08, 0.14] —
  the band was optimistic, and per design rule §0.4 the wide interval is correct.
  Asserting pitch/blade count is what narrows it.
- **Property-based round-trip** (`tests/property/`): Hypothesis sweeps over the
  physics identities (BPF↔RPM, C_T-cancelling thrust ratio, hover force balance,
  disk loading, frame geometry, endurance monotonicity) and audio RPM recovery
  across the blade-pass band.

## Design note worth remembering

Monotonicity ("adding a modality never widens an interval") is enforced and tested
on **relative** width (width/|median|), not absolute. An input like verbal can
*relocate* the estimate (e.g. fixing prop diameter), which legitimately grows
absolute width while reducing relative uncertainty. Absolute monotonicity holds
cleanly only for the soft-observation reweighting path (image/video). This is the
honest interpretation of the "confidence ladder."

## Blocked on you (external dependencies, not code)

1. **Golden set (spec §5.1) — the gate for everything trustworthy.** 15–30 drones,
   all four modalities + measured truth. Highest-value single item: a thrust-stand
   sweep (RCbenchmark + tachometer) — it calibrates C_T/C_P and validates RPM.
   Without it, calibration (Phase 7) and the §10 metric table cannot be produced.
   **Capture protocol + landing steps are written up in `docs/GOLDEN_SET.md`; the
   eval harness (`eval.py`) already consumes it.**
2. **One real VLM run** (`image/extract.py::extract_from_image`): the call is
   wired + tested via a fake client, and the default model id was verified
   current 2026-07-08 (`claude-sonnet-5`, high-res vision; thinking disabled for
   fast extraction). Remaining: set `ANTHROPIC_API_KEY`, install `.[vision]`,
   and run once on a real drone photo to validate.
3. **Real footage for video tracking** (`video/track.py::track_drone`): the
   tracker is implemented + tested on synthetic frames and an opencv round-trip.
   Validating end-to-end needs real footage + flight-log truth + the `vision`
   extra.
4. ~~UIUC prop data~~ **Resolved 2026-07-07:** downloaded + ingested (250 props;
   provenance in `data/public/SOURCES.md`). The site has no explicit license
   (© UIUC Applied Aerodynamics Group, citation requested), so the derived
   `config/prop_db.parquet` is **local-only like DREGON** — gitignored, with the
   rebuild recipe in the README. Could be revisited by emailing Prof. Selig for
   explicit permission. (Betaflight blackbox / DJI logs remain stubs documenting
   their target schema.)

## Next steps (in order)

1. Start collecting the golden set per `docs/GOLDEN_SET.md`; thrust-stand sweeps
   first. This is now the critical path — nearly everything else is code-complete.
2. One real VLM run to confirm the wired call (set the key, pick the model).
3. Once golden `calib`/`test` exist: fit conformal (`eval.fit_calibrator`),
   produce reliability diagrams + the §10 table (`eval.format_metrics_table`).
4. A drone-vs-not detector trained on real clips (DroneAudioDataset): physics alone can't
   tell a drone from a fan or mains hum, so the live mic can false-alarm.
5. `/predict` still assumes 2 blades when blade count isn't given; reuse identify's
   blade marginalization there.
6. The class prop bands overlap: a given 15″ prop reads cinematic over survey because
   cinematic's 6–15″ band gives it about twice the likelihood of survey's 13–30″. Revisit
   the `classes.yaml` bands against real specs.

## Not yet built

- Golden-set data itself + a thin `scripts/ingest_golden.py` (the schema, split
  assignment, and eval it feeds are all built; see `docs/GOLDEN_SET.md`). The
  §10 table and reliability diagrams are fully coded — they just need this data.

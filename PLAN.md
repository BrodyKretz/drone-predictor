# Augur — Project Plan & Status

Living status doc. Full spec: [`docs/SPEC.md`](docs/SPEC.md). Build agreements:
spec §0. **Update the status column whenever a phase advances.**

_Last updated: 2026-06-28_

## Where this stands right now

A working, tested **sound + verbal → calibrated distributions** pipeline exists
end-to-end (Phases 0–4 + the testable physics from later phases), exposed via both
a CLI and a FastAPI endpoint, with a property-based round-trip suite over the
physics and audio recovery. Image/video/calibration are now implemented and
tested (VLM call via injected fake client, video tracker on synthetic frames,
eval harness on synthetic reports) — not stubs. The remaining blockers are purely
external: data / API key / hardware (see "Blocked on you"). The code is close to
feature-complete; what's missing is real data. 112 tests pass.

Run it:
```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python scripts/make_demo_sample.py
./.venv/bin/augur predict --audio data/demo/drone.wav --verbal data/demo/spec.json
./.venv/bin/pytest tests/ -q          # 112 tests
```

The HTTP API needs the serve extra: `./.venv/bin/pip install -e ".[serve]"`, then
`./.venv/bin/augur serve` (POST multipart audio/verbal to `/predict`; `/health`
for a liveness check). Without the extra the API tests skip; the rest still run.

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
- **Eval harness** (`eval.py`): drone-level leakage-free split assignment,
  §10 metrics (MAPE / coverage / sharpness), and conformal-calibrator fitting
  from a calib split. Capture protocol in `docs/GOLDEN_SET.md`.
- **Conformal calibration** (`fusion/calibration.py`): split-conformal width
  factor; corrects overconfident intervals to nominal coverage on a holdout.
- **Manifest** (`data_manifest.py`): schema + round-trip + drone-level leakage
  guard.
- **CLI** (`augur predict`, `augur version`, `augur serve`), report renderer,
  synthetic data gen.
- **HTTP API** (`api.py`): FastAPI `/predict` (multipart audio + verbal) and
  `/health`, dev-permissive CORS for the web UI. Core logic is a framework-free
  helper (`predict_from_uploads`) so it unit-tests without FastAPI; HTTP routes
  tested via TestClient when the serve extra is present.
- **Web UI** (`web/`, Vite + React): audio upload + verbal-spec dropdowns + one-click
  synthetic demo templates (racing/cinematic/survey, each recovering true RPM within
  ~1%) → interval bars with confidence colors + per-input attribution. Build verified
  (`npm run build`) and components server-rendered against a live API report.
- **Real-audio validation** (`scripts/validate_real_audio.py`, `docs/REAL_AUDIO.md`):
  recover RPM from any WAV and score vs. a known true RPM. DREGON is the ideal
  real dataset (on-board audio + rotor-speed truth) but is **academic-use only —
  not redistributable**, so it's run locally (gitignored `data/external/`), never
  committed. DroneAudioset (MIT) is the bundle-able but noisier alternative.
- **Prop-DB ingest** (`prop_ingest.py` + `scripts/ingest_uiuc.py`): parses UIUC/APC
  static-test files (`<maker>_<D>x<P>_static_*.txt`) into one representative
  (C_T, C_P) per prop → `config/prop_db.parquet`, which `prop_db` then prefers
  over the fallback bands. Validated against format fixtures; **awaiting the real
  data download + license check (see "Blocked on you" #4)**.
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
   wired + tested via a fake client. Confirm the default model id against live
   docs, set `ANTHROPIC_API_KEY`, install `.[vision]`, and run once on a real
   photo to validate.
3. **Real footage for video tracking** (`video/track.py::track_drone`): the
   tracker is implemented + tested on synthetic frames and an opencv round-trip.
   Validating end-to-end needs real footage + flight-log truth + the `vision`
   extra.
4. **Public data download** (UIUC/APC prop data): the *ingest pipeline is built and
   tested* (`prop_ingest.py` + `scripts/ingest_uiuc.py`). What remains is a human
   call: download the UIUC data, verify its license permits use/redistribution in
   this MIT repo, log provenance in `data/public/SOURCES.md`, then run
   `python scripts/ingest_uiuc.py --src <dir>` to build `config/prop_db.parquet`.
   Until then C_T/C_P use the wide config fallback bands. (Betaflight blackbox /
   DJI logs remain stubs documenting their target schema.)

## Next steps (in order)

1. Download UIUC/APC prop data + license check, then run `scripts/ingest_uiuc.py`
   to build `config/prop_db.parquet` (the ingest code is done; this tightens every
   thrust/power/mass interval). No hardware — just the download + license call.
2. Start collecting the golden set per `docs/GOLDEN_SET.md`; thrust-stand sweeps
   first. This is now the critical path — nearly everything else is code-complete.
3. One real VLM run to confirm the wired call (set the key, pick the model).
4. Once golden `calib`/`test` exist: fit conformal (`eval.fit_calibrator`),
   produce reliability diagrams + the §10 table (`eval.format_metrics_table`).

## Not yet built

- `config/prop_db.parquet` (ingest pipeline ready; data not yet downloaded).
- Reliability-diagram plots (`viz` extra / matplotlib). The metric table + coverage
  numbers are coded (`eval.py`); only the plotted diagrams and a real populated
  §10 table (needs the golden set) remain.
- Golden-set data itself + a thin `scripts/ingest_golden.py` (the schema, split
  assignment, and eval it feeds are all built; see `docs/GOLDEN_SET.md`).

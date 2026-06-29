# Augur — Project Plan & Status

Living status doc. Full spec: [`docs/SPEC.md`](docs/SPEC.md). Build agreements:
spec §0. **Update the status column whenever a phase advances.**

_Last updated: 2026-06-28_

## Where this stands right now

A working, tested **sound + verbal → calibrated distributions** pipeline exists
end-to-end (Phases 0–4 + the testable physics from later phases), exposed via both
a CLI and a FastAPI endpoint, with a property-based round-trip suite over the
physics and audio recovery. Image/video/calibration are scaffolded with honest
stubs and tested where the logic is self-contained. The true blockers are
external-data / API dependent (see "Blocked on you" below). 83 tests pass.

Run it:
```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python scripts/make_demo_sample.py
./.venv/bin/augur predict --audio data/demo/drone.wav --verbal data/demo/spec.json
./.venv/bin/pytest tests/ -q          # 83 tests
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
| 5 | Image module: VLM extraction + geometry | 🟡 partial — geometry done+tested; VLM JSON parser done+tested; **live VLM call stubbed** |
| 6 | Video module: tracking + maneuvers + coast-down mass | 🟡 partial — maneuver segmentation + drag mass done+tested; **pixel→velocity tracking stubbed** |
| 7 | Calibration + eval harness + hardening | 🟡 partial — conformal calibrator done+tested; **needs golden set to fit + reliability diagrams + full metric table** |

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
- **VLM parsing** (`image/extract.py`): defensive JSON parse (bare/fenced/prose/
  malformed) + observation builder for the direct battery path.
- **Maneuver segmentation** (`video/maneuvers.py`): hover/climb/cruise/coast from
  velocity series → coast mass observations.
- **Conformal calibration** (`fusion/calibration.py`): split-conformal width
  factor; corrects overconfident intervals to nominal coverage on a holdout.
- **Manifest** (`data_manifest.py`): schema + round-trip + drone-level leakage
  guard.
- **CLI** (`augur predict`, `augur version`, `augur serve`), report renderer,
  synthetic data gen.
- **HTTP API** (`api.py`): FastAPI `/predict` (multipart audio + verbal) and
  `/health`. Core logic is a framework-free helper (`predict_from_uploads`) so it
  unit-tests without FastAPI; HTTP routes tested via TestClient when the serve
  extra is present.
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
2. **Live VLM call** (`image/extract.py::extract_from_image`): needs an Anthropic
   API key + the *current* model id / tool-use setup from live docs (claude-api
   skill). Parser + schema already done; wiring is a thin layer.
3. **Video tracking** (`video/track.py::track_drone`): needs real footage +
   flight-log truth + the `vision` extra (opencv+norfair). Downstream maneuver/
   mass logic is done and tested on synthetic series.
4. **Public data ingest** (`scripts/ingest_*.py`): UIUC/APC prop data (build
   `config/prop_db.parquet` — until then C_T/C_P use wide config fallback bands),
   Betaflight blackbox, DJI logs. Stubs document the target schema. Verify
   licenses; log provenance in `data/public/SOURCES.md`.

## Next steps (in order)

1. Ingest UIUC/APC prop data → `config/prop_db.parquet` (tightens every thrust/
   power/mass interval; no hardware needed, just download + license check).
2. Start collecting the golden set; capture thrust-stand sweeps first.
3. Wire the live VLM call once a model is chosen (Phase 5 completion).
4. Once golden `calib`/`test` exist: fit conformal, produce reliability diagrams
   and the §10 metric table (Phase 7).

## Not yet built

- `config/prop_db.parquet` (using fallback bands).
- Reliability diagrams / full §10 metric table (needs golden set).

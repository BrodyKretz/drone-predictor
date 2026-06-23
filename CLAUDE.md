# CLAUDE.md — Augur (drone-predictor)

Multimodal drone property inference: predict RPM, thrust, weight, power, disk
loading, T/W, battery, endurance, class, features from any subset of {sound,
verbal spec, image, video}, as **calibrated distributions, never point estimates**.

## Read these first
- **[`PLAN.md`](PLAN.md)** — current phase status, what's done, what's blocked, next steps. *Start here.*
- **[`docs/SPEC.md`](docs/SPEC.md)** — full project spec (the source of truth for what & why).

## Non-negotiable design rules (from spec §0, §7.7, §14)
1. Every *predicted* quantity is a `Distribution` (MC samples), never a bare
   float. Asserted facts can be scalars; inferences cannot.
2. The forward simulator (`physics/simulator.py`) is the test oracle. The inverse
   pipeline is validated by round-tripping against it. Build/validate against it.
3. All physical assumptions live in `config/priors.yaml` / `config/classes.yaml`,
   versioned — never hardcode efficiency/energy-density/Cd/mass-fraction in logic.
4. When inputs don't constrain a variable, **widen the interval — never invent a
   number.** A wide honest interval passes; a narrow wrong one fails.
5. Mass comes from **drag (coast/cruise) or hover force-balance only.** Climb/
   throttle acceleration is mass-independent — using it for mass is a bug (§14).
6. `sum(T) = m*g` is valid **only in hover** — gate on detected hover state.
7. Feature inventory is image/video-only; sound never contributes to it.

## How to run
```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python scripts/make_demo_sample.py
./.venv/bin/augur predict --audio data/demo/drone.wav --verbal data/demo/spec.json
./.venv/bin/pytest tests/ -q     # full suite
./.venv/bin/ruff check src/ tests/
```
A local `.venv` is used deliberately (don't install into the user's conda env).
Optional extras: `.[audio]` (librosa), `.[vision]` (opencv+norfair+anthropic),
`.[serve]` (fastapi), `.[viz]` (matplotlib).

## Working agreements for this repo
- Plan before editing anything non-trivial (>1 file / >20 lines); confirm approach
  first. Trivial fixes can skip the loop.
- TDD / verify-as-you-go: write the test, run it, read real output. Don't claim
  something works without running it. Property-based round-trip tests are
  mandatory for the physics core.
- No code comments unless genuinely necessary; self-documenting names. Ask before
  adding dependencies (the spec's dep list is pre-approved).
- Tests live under `tests/{unit,integration,calibration,ablation}`; match the
  existing style.

## Layout
`src/augur/{audio,physics,image,video,fusion}` + `types.py`, `config.py`,
`pipeline.py`, `report.py`, `cli.py`, `data_manifest.py`. Config in `config/`,
data + manifest in `data/`, ingest/synthetic scripts in `scripts/`.

# Live drone identification — design

_Status: proposed 2026-10-14. Built from a working throwaway prototype the user reviewed._

## Goal

One screen. You drop a recording, pick a sample, or record from the mic, and watch it update
live: a big **best guess of what kind of drone it is**, and the physical estimates (mass,
RPM, power, …) changing on the side. The input area is always visible. Every estimate keeps
its interval (design rule 1); the class guess is a probability, never a bare label.

## What the user sees

- **Input** (top): drop zone with *Choose file* (WAV/MP3/OGG, decoded in the browser) and
  *Record* (mic); four sample chips; an optional "Know anything about it?" row (blades per
  prop, prop size, motors). While audio plays or records: a live header (state, source,
  clock, Stop, and the input buttons) plus the averaged spectrum with harmonic markers.
- **What it is** (main): best-guess class name and %, a confidence word (confident ≥ 80%,
  likely ≥ 60%, leaning ≥ 40%, unsure), the class description from `classes.yaml`, ranked
  bars for every class, and "why" lines (blade-pass tone, hover evidence, blade count). States
  for *listening / no tone*, *not hovering*, and *doesn't fit a known class*.
- **Live estimates** (side): mass, rotor speed, hover power, hover endurance, battery, total
  thrust, disk loading, prop size, motors, thrust-to-weight. Each row: median (or the range
  when the interval is wide), an interval bar on a log scale spanning the no-input 99% band,
  a confidence chip, and a sparkline of the session history. When blade count is unknown,
  rotor speed shows both readings ("11.8k or 7.85k rpm").

Visual language stays the SIMNET-derived dark palette (pane `#222`, black tab strips,
`#663399` fills, `#a986ec` marks); the dense three-pane dashboard is dropped.

## Backend

1. **Bug fix first (rule 4).** `posterior.py` computes disk loading and the minimum frame
   diagonal from the *modal* motor count, so with an unknown motor count the frame diagonal
   comes back as an exact value that changes with the seed (0.147 vs 0.332 m for a 5″ prop).
   Compute both per motor-count subset.
2. **`augur/fusion/identify.py`** — `identify(samples, sr, verbal=None, n, seed) -> IdentifyReport`.
   - Acoustics: averaged spectrum max-pooled to 256 points over `[0, 7·BPF]`, the 6 comb
     harmonics as hit/miss, comb score, and flight-state evidence (pitch variation vs the
     4% hover threshold) instead of a bare label.
   - Hypotheses: every class in `classes.yaml` except `unknown`, × blade counts from a new
     `blade_count_options` in `priors.yaml` (today `[2, 3]` is hardcoded in `posterior.py`).
     A verbal spec narrows them (given class or blade count).
   - Evidence: for hypothesis (c, B) run the existing plug-in fusion (RPM = BPF·60/B, latents
     from class c). The class's all-up-mass band is a uniform prior on mass and hover gives
     m ∝ rpm², so `p(BPF | c, B) = (60/B) · E[1{m ∈ band} · 2m / (rpm · (hi − lo))]`. The same
     term is the importance weight that turns the plug-in samples into the posterior under
     (c, B). Estimates are the evidence-weighted mixture over hypotheses.
   - Gating (rule 6): no tone → `no_tone`, priors only; not hovering → classes stay at the
     prior and mass comes from the class prior; a tone at 50/60/100/120 Hz (±1.5%) adds a
     "may be mains hum" note.
3. **Types** (`types.py`, pydantic): `Harmonic`, `FlightStateEvidence`, `AcousticsDiagnostics`,
   `ClassProbability`, `BladeProbability`, `IdentifyReport`.
4. **API**: `POST /identify` (multipart WAV + optional `spec` JSON) and `GET /reference`
   (no-input 99% bands and class descriptions, cached). `/predict` and the CLI are unchanged.
5. **Config**: add `blade_count_options` to `priors.yaml`, read it in `posterior.py` too, and
   bump the config version to 3.
6. **Demo clips**: `make_templates.py` gets a duration option; regenerate the four templates
   at 8 s so live playback has time to update.

## Frontend (`web/`)

- `App.jsx` owns the session (mode, latest result, history, known facts, errors).
- Components: `InputPanel`, `Spectrum`, `Identification`, `EstimatesPanel`, `EstimateRow`,
  `Sparkline`.
- `lib/audio.js`: decode files, play a buffer while handing out the latest window, mic ring
  buffer (`getUserMedia` with echo cancellation / noise suppression / AGC off), WAV encoder.
- Live loop: every 600 ms send the latest 3 s to `/identify`; skip a tick while a request is
  in flight. `Report.jsx` and `VariableBar.jsx` are removed.
- No new npm dependencies. `npm test` runs `node --test` on `src/lib`.

## Testing

- pytest, test first: the frame-diagonal fix; `identify` — posterior sums to 1; the racing
  clip ranks racing first; a given 5″ prop pushes racing ≥ 0.9; a throttle ramp (not
  hovering) leaves classes at the prior with a note; white noise → `no_tone`; a 60 Hz hum
  carries the hum note; simulator round-trip over drones drawn from each class's priors
  checks top-1 accuracy beats chance and the posterior isn't overconfident. API tests for
  `/identify` and `/reference` (helper-level so they run without the serve extra).
- `node --test`: WAV encoder header/length, formatters.
- Live checks: `npm run build`, headless Chromium in real time (fake mic) screenshots.

## Known limits (surfaced in the UI)

- It names a **class**, not a model. On the demos it's 41–54% sure, and it calls the 3-blade
  military clip "cinematic": a 2-blade 9″ quad and a 3-blade 13″ quad make the same tone, and
  the classes in `classes.yaml` overlap.
- It can't tell a drone from other tonal sounds: a test beep scored "cinematic 37%", a 60 Hz
  hum "survey 68%". Next step: a drone-vs-not detector trained on real clips
  (DroneAudioDataset).
- `/predict` still assumes 2 blades when blade count isn't given (follow-up).

## Out of scope

3D airframe view, the SIMNET multi-pane dashboard, image/video input, CI changes, any trained
model.

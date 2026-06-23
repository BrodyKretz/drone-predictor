# Augur — Multimodal Drone Property Inference (full spec)

> Verbatim project specification. This is the source of truth for *what* we're
> building and *why*. For current build status and next steps, see
> [`../PLAN.md`](../PLAN.md).

The system predicts a drone's physical and performance properties from any
combination of four inputs (sound, verbal spec, image, video), getting strictly
more confident as more inputs are supplied. It outputs calibrated distributions,
never point estimates.

## Working agreements

- Build in the phase order in §11 (see PLAN.md). Do not skip ahead. Each phase
  has acceptance criteria that must pass before moving on.
- The physics forward-simulator (§5.7) is built first and is the test oracle for
  everything else. The inverse pipeline is validated against it.
- Every predicted quantity is a distribution (samples or a parametric posterior),
  not a scalar. Enforced in output types (§7.7). Reject any design that returns a
  bare float for a predicted property.
- All physical assumptions (efficiency bands, mass fractions, energy density,
  drag coefficients) live in a single versioned config (§13), never hardcoded.
- Write tests as you go. Property-based round-trip tests (forward → inverse) are
  mandatory for the physics core.
- When something is genuinely unknowable from the inputs, widen the interval, do
  not invent a number (§14).

## 1. What we're building

A progressive-enhancement estimator. Inputs (any subset): sound, verbal spec,
image, video. Estimates: RPM, per-motor RPM spread, thrust per motor, total
thrust, weight, shaft power, electrical power, disk loading, frame size,
thrust-to-weight ratio, drone class, battery capacity (Wh, mAh when cell count
known), battery mass fraction, feature inventory, flight endurance.

Each input is an independent constraint, fused as likelihoods so adding an input
can only narrow the posterior (given consistent evidence). The system reports,
per prediction, which inputs tightened which variables.

## 2. Physics reference

### Tonal structure
- Blade pass frequency: `BPF = (RPM/60) * num_blades` (dominant tone; harmonics
  at integer multiples).
- Commutation tone: `f_elec = (RPM/60) * pole_pairs`.
- `RPM = BPF * 60 / num_blades` — primary, high-confidence recovery.

### Thrust and power (per motor)
- `T = C_T * rho * n^2 * D^4`, `n = RPM/60`.
- `P = C_P * rho * n^3 * D^5`.
- `P_elec = P_shaft / eta`, `eta ∈ [0.70, 0.85]`.
- `C_T`, `C_P` from prop DB by diameter+pitch; pitch usually unknown → carry as a
  distribution over the size class.

### Three non-obvious rules
1. **Weight equals thrust only in hover.** Detect hover (steady RPM, ~zero
   vertical velocity) before applying `sum(T) = m*g`.
2. **Thrust ratios cancel C_T; absolute thrust does not.** `T2/T1 = (n2/n1)^2`
   exactly. So `T_max/W = (n_max/n_hover)^2 = 1 + a_max/g` — clean from RPM ratio.
   Do NOT extract mass from climb/throttle accel: `(n_accel/n_hover)^2 = 1 + a/g`
   is mass-independent AND C_T-independent — a consistency check, not a mass.
3. **Mass comes from drag, which only appears in motion.** Drag scales with
   frontal area (visible) and velocity² (measurable), not mass.
   - Coast-down (preferred): `m*dv/dt = -drag`, `drag = 0.5*rho*Cd*A*v^2`.
   - Steady cruise: `T*sin(theta) = drag` → absolute thrust independent of C_T →
     back-solves C_T.

### Battery (the soft end)
- `E = m_battery * energy_density`, `energy_density ∈ [150, 250]` Wh/kg.
- Get `m_battery` via: residual (`m_hover - m_components - m_payload`, wide),
  ceiling (`m_max_lift - m_components`, upper bound), or direct (read pack
  dims/label from image — best). Gap between residual and ceiling = thrust margin
  → class signal.
- Endurance `t = E / P_elec`. Convert Wh→mAh ONLY when cell count S known
  (`mAh = Wh / V_nominal * 1000`).
- Disk loading `DL = sum(T) / (N * pi * (D/2)^2)`. `t ∝ f * sqrt(1/DL)` — sqrt
  makes the observable part forgiving; the brutal uncertainty lives in `f`
  (battery mass fraction).

## 3. Architecture

Each input module emits *observations* with uncertainty (e.g. "RPM = 9000 ± 90",
"frontal area = 0.06 m² ± 0.01", "coast decel = 3.1 m/s²"). The fusion layer
combines whatever observations are present with the physics core + priors into
per-variable posteriors. Runs with any subset; missing modules contribute no
observations.

## 5. Data assets (order: golden → public bootstrap → synthetic → scaled)

### 5.1 Golden set (do first)
15–30 fully-characterized drones, all four modalities + measured truth: weight
(with/without battery), battery (Wh, mAh, S, dims, mass), prop (diameter, pitch,
blades), motor (KV, stator, poles), thrust-stand sweep (thrust + electrical power
vs RPM with tachometer — the calibration goldmine), frame (diagonal, frontal
area), class + feature inventory.

### 5.2 Audio
WAV mono ≥44.1 kHz, log sample rate + mic distance. Captures: steady hover (≥10s),
throttle ramp, horizontal pass (Doppler), thrust-stand at known RPM steps. Labels:
ground-truth RPM (blackbox/stand), drone id, flight-state segments. Bench
recordings (single motor, swept RPM logged) are gold for fitting C_T/C_P.

### 5.3 Images
JPEG/PNG with EXIF, multiple angles. Scale reference (known-size object or camera
distance/FoV) — without it, area is ratio-only. Labels: class, battery (visible,
dims, printed mAh/S), payload, features, frontal area, true weight.

### 5.4 Video
MP4 ≥30 fps, log fps+resolution. Prioritize clips with coast-downs and steady
cruise (they carry mass). Ground truth: synchronized flight log (Betaflight
blackbox / DJI). Labels: per-segment maneuver tags, true mass, scale metadata.

### 5.5 Verbal/metadata
Structured pydantic records: num_motors, props_per_motor, prop_diameter,
prop_pitch?, blade_count?, voltage?, cell_count?, + freeform notes. Link to drone
id.

### 5.6 Public datasets
UIUC/APC prop coefficients → prop_db; manufacturer thrust tables → component/
thrust priors; drone-audio sets (DREGON etc.) for detector robustness. Verify
licenses; record provenance in data/public/SOURCES.md.

### 5.7 Synthetic (forward simulator — build first)
`physics/simulator.py` takes a full spec + flight state → synthetic spectrum/
audio (BPF + harmonics + commutation + broadband, configurable noise/Doppler/
distance) AND the true values of every output. It is the test oracle, infinite
labeled data source, and ablation rig.

## 6. Manifest (data/manifest.parquet)

One row per sample: sample_id, drone_id, audio_path?, image_paths?, video_path?,
verbal(json)?, flight_state?, truth(json), source, split(train/calib/test),
license. Truth only on golden/synthetic. Splits disjoint at the DRONE level.

## 7. Modules (see repo structure)

audio (spectral/rpm/hover), physics (core/prop_db/drag/simulator), image
(extract/geometry), video (track/maneuvers), fusion (observations/posterior/
calibration), report, api, cli. Output types: `Distribution` (samples + mean/
median/interval/confidence_label). Never emit a bare scalar for a prediction.

## 8. Uncertainty & calibration

Monte Carlo over all uncertain priors + observation noise (no closed-form
shortcuts that drop covariance). Conformal calibration on `calib` split so stated
X% intervals cover X% on held-out drones; reliability diagrams. Honesty rule: if
inputs don't constrain a variable, the interval must be wide.

## 9. Testing

Unit (RPM detection on synthetic incl. noise/Doppler/harmonics; every physics eq
vs hand-computed; prop-db; VLM JSON parsing incl. malformed). Property-based
round-trip (simulator→inverse, Hypothesis). Integration (golden holdout, each
subset). Calibration (coverage within tolerance). Ablation (monotonicity: adding
a modality never widens). Robustness (wind, clipping, low SNR, occlusion, missing
scale, offset RPMs). CI runs unit+property+calibration every push.

## 10. Targets (golden test split)

| variable | metric | target (sound → +image → +video) |
|---|---|---|
| RPM | % error | < 2% all levels |
| disk loading | % error | < 10% |
| thrust (total) | % error | 25–40% → ~20% → ~12% |
| weight | % error | 25–40% → ~18% → ~12% |
| T/W | % error | ~20% (needs max-throttle or video) |
| power (elec) | % error | 30–50% → ~25% |
| battery (Wh) | % error | 60%+ → ~20% if pack visible |
| endurance | % error | ~2× → ~40% (never "solved") |
| class | accuracy | high with image/video |
| features | F1 | 0 from sound → high with image |
| interval coverage | \|cov − nominal\| | < 5% everywhere |

Coverage is the most important row. Narrow-but-wrong is a failure; wide-but-honest
passes.

## 11. Phases

See [`../PLAN.md`](../PLAN.md) for the phase list with live status.

## 13. Config registry

`config/priors.yaml` is the single source of every assumption (air density,
efficiency band, energy density by chemistry, drag coefficient, pitch-by-class,
battery mass fraction by class, component mass by prop inch). Any change = version
bump. Predictions log the config version.

## 14. Risks & known limits (encode, don't paper over)

- Battery, mass fraction, endurance have an irreducible floor — no sensor recovers
  a pack you can't see. When not visible: widen, classify, say so. Don't let
  class-prior inference masquerade as measurement.
- Pitch and efficiency are never directly observed — stay distributions.
- Scale metadata is the silent failure mode for image/video — without it, area
  and frame size are ratio-only; propagate a wide prior.
- Climb/throttle acceleration does NOT give mass. If a contributor adds such a
  "mass observation," it is a bug. Mass = drag (coast/cruise) only.
- Feature inventory is image/video-only. Sound contributes nothing to it.

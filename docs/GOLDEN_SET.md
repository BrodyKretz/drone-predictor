# Golden Set — capture protocol

The golden set is the gate for everything trustworthy (calibration + the §10
metric table). It is 15–30 real drones, each captured across all four modalities
with **measured** ground truth. This doc says exactly what to record so the data
drops straight into the pipeline.

Consumers already built: `augur.data_manifest` (schema + drone-level leakage
guard), `augur.eval` (splits + metrics + conformal fit), `augur.prop_ingest`
(prop coefficients). Nothing here needs new code — just data in the right shape.

## Per drone — measured truth (the hard part)

The single highest-value capture is a **thrust-stand sweep** (e.g. RCbenchmark
Series 1580/1585 + a tachometer). For one representative motor+prop:

| measure | how | why it matters |
|---|---|---|
| RPM vs throttle | tachometer / ESC telemetry | validates the acoustic RPM (§10 <2% target) and anchors T/W |
| thrust vs RPM | thrust stand | calibrates `C_T` for that prop (kills the biggest sound-only unknown) |
| electrical power vs RPM | stand wattmeter | calibrates `C_P` + motor/ESC efficiency band |
| all-up mass | scale, ±1 g | ground truth for the mass estimate |
| battery Wh | printed mAh × cell count × nominal V | ground truth for the battery/endurance path |

Also record, per drone: motor count, prop diameter × pitch, blade count, cell
count, frame class (racing / cinematic / survey), and any payload.

## Per drone — the four modalities

- **Sound**: ≥3 s WAV per flight state (hover, climb, cruise, a power-off coast).
  Steady hover is mandatory (that's where `sum(T)=m·g` is valid). Note mic
  distance. 44.1 kHz mono is plenty.
- **Verbal**: the asserted spec as JSON (the `VerbalSpec` fields). Only what a
  user would actually state — don't leak measured truth into it.
- **Image**: 1–3 stills, ideally with a **scale reference** in frame (a ruler or
  a known-size object) so pixel→metric geometry resolves. Get the battery label
  legible in at least one shot — the printed mAh is the direct battery path.
- **Video**: a clip containing a clear cruise→coast so drag mass is recoverable,
  plus a climb for the T/W consistency check. Note the scale (`m_per_pixel`) or
  include an in-frame reference.

## File layout

```
data/
  golden/
    <drone_id>/
      audio/hover.wav climb.wav coast.wav
      images/front.jpg battery.jpg
      video/flight.mp4
      verbal.json
      truth.json          # measured ground truth (see table above)
```

`drone_id` is one physical airframe. Reusing the same airframe with a different
battery is still the same `drone_id` — splits are disjoint per airframe to
prevent leakage.

## Landing it in the manifest

Build one manifest row per sample with the `MANIFEST_COLUMNS` schema, put `truth`
(the measured dict) only on golden rows, and assign splits at the drone level:

```python
from augur.eval import assign_splits
splits = assign_splits(all_drone_ids, fractions=(0.6, 0.2, 0.2))
# write rows, set row["split"] = splits[row["drone_id"]]
```

`save_manifest` re-validates leakage on write, so a mistake fails loudly.

## Then, and only then

1. Fit the conformal calibrator on the `calib` split (`augur.eval.fit_calibrator`).
2. Score the `test` split (`augur.eval.evaluate` → `format_metrics_table`) —
   that table is the §10 deliverable and the honest measure of every claim.

Target coverage ≈ nominal (a 90% interval should contain truth ~90% of the time);
`is_calibrated` flags variables that miss.

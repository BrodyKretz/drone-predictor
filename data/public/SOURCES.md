# Public data sources & provenance

Record every ingested public dataset here **before** ingest: name, URL, license,
date pulled, and what it's used for. Verify the license permits your use.

| dataset | url | license | pulled | used for | status |
|---|---|---|---|---|---|
| UIUC Propeller Data Site | https://m-selig.ae.illinois.edu/props/propDB.html | © UIUC Applied Aerodynamics Group; no explicit license, citation requested (see note) | 2026-07-07 | C_T/C_P → config/prop_db.parquet | ingested (250 props, vols 1–4) |
| APC performance files | https://www.apcprop.com/technical-information/performance-data/ | check site terms | — | C_T/C_P, component priors | not yet ingested |
| DREGON (drone audio) | https://dregon.inria.fr/ | check terms | — | acoustic detector robustness/augmentation (often no RPM truth) | not yet ingested |
| T-Motor / iFlight thrust tables | manufacturer sites | manufacturer terms | — | component-mass + thrust priors | not yet ingested |

Notes:
- **UIUC prop DB**: raw zip (MD5 a41e484f1fd0fb6ff80b76e27410808b, verified) lives
  in gitignored `data/external/uiuc/` — never committed. The site states no
  license; it requests citation: "J.B. Brandt, R.W. Deters, G.K. Ananda,
  O.D. Dantsker, and M.S. Selig (2026-07-07), UIUC Propeller Database, Vols 1-4,
  https://m-selig.ae.illinois.edu/props/propDB.html". `config/prop_db.parquet`
  is a derived summary — one representative static (C_T, C_P) per test file,
  not the measured curves. Naming quirks normalized on ingest: mm-designated
  micro props (pl/ef/vp/kpf makers) converted to inches; dropped decimal points
  restored (125x9 → 12.5x9, 12x65 → 12x6.5); `_3b_`/`_4b_` blade-count suffixes
  parsed (unmarked = 2-blade).
- Many public drone-audio sets lack RPM ground truth — useful for detector
  robustness and augmentation, not for RPM calibration.
- The calibration goldmine is your own thrust-stand sweep (RCbenchmark +
  tachometer) on the golden set; public data only bootstraps priors.

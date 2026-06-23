# Public data sources & provenance

Record every ingested public dataset here **before** ingest: name, URL, license,
date pulled, and what it's used for. Verify the license permits your use.

| dataset | url | license | pulled | used for | status |
|---|---|---|---|---|---|
| UIUC Propeller Data Site | https://m-selig.ae.illinois.edu/props/propDB.html | check per-file | — | C_T/C_P → config/prop_db.parquet | not yet ingested |
| APC performance files | https://www.apcprop.com/technical-information/performance-data/ | check site terms | — | C_T/C_P, component priors | not yet ingested |
| DREGON (drone audio) | https://dregon.inria.fr/ | check terms | — | acoustic detector robustness/augmentation (often no RPM truth) | not yet ingested |
| T-Motor / iFlight thrust tables | manufacturer sites | manufacturer terms | — | component-mass + thrust priors | not yet ingested |

Notes:
- Many public drone-audio sets lack RPM ground truth — useful for detector
  robustness and augmentation, not for RPM calibration.
- The calibration goldmine is your own thrust-stand sweep (RCbenchmark +
  tachometer) on the golden set; public data only bootstraps priors.

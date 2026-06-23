"""Betaflight blackbox -> ground-truth RPM + flight-state segments.

STATUS: stub. Needs real .bbl/.bfl logs + a parser (orangebox or blackbox-tools,
the project lists both). Blackbox logs record per-motor eRPM (when bidirectional
DShot telemetry is on), throttle, gyro and accel — i.e. true RPM and the
hover/climb/cruise/coast segmentation that turns an audio clip into a *labeled*
training sample.

Target: append rows to data/manifest.parquet linking each audio clip to its
true RPM time-series and segment tags.
"""

from __future__ import annotations


def main():
    raise NotImplementedError(
        "ingest_blackbox is a stub. Parse Betaflight blackbox logs (orangebox / "
        "blackbox-tools), extract per-motor eRPM -> true RPM and segment the flight "
        "into hover/climb/cruise/coast, and write labels into data/manifest.parquet."
    )


if __name__ == "__main__":
    main()

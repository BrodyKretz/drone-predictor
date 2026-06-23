"""DJI flight log -> velocity / acceleration / attitude ground truth.

STATUS: stub. Needs DJI .txt/.dat logs + a parser. DJI logs give GPS velocity,
acceleration, attitude (tilt) and sometimes battery state — the truth that makes
a *video* clip trainable (coast-down deceleration, cruise speed + tilt for the
drag-based mass extraction in augur.physics.drag).

Target: append rows to data/manifest.parquet linking each video clip to true
velocity/accel/tilt time-series and per-segment maneuver tags.
"""

from __future__ import annotations


def main():
    raise NotImplementedError(
        "ingest_dji_log is a stub. Parse DJI flight logs into velocity/accel/tilt "
        "time-series + maneuver tags, and write them into data/manifest.parquet so "
        "video clips gain ground truth for the drag-based mass path."
    )


if __name__ == "__main__":
    main()

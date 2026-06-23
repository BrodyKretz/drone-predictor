"""Pixel -> metric geometry for images.

Scale metadata is the silent failure mode (spec §14): without a known-size
reference or camera distance+FoV, area and frame size are ratio-only and must be
returned as a wide prior, never fabricated metric values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from augur.types import Distribution


@dataclass
class ScaleReference:
    """Either a metres-per-pixel scale, or camera distance + horizontal FoV +
    image width to derive it. None means no scale -> ratio-only output."""

    m_per_pixel: float | None = None
    distance_m: float | None = None
    hfov_deg: float | None = None
    image_width_px: int | None = None

    def resolve_m_per_pixel(self) -> float | None:
        if self.m_per_pixel is not None:
            return self.m_per_pixel
        if self.distance_m and self.hfov_deg and self.image_width_px:
            view_width_m = 2.0 * self.distance_m * np.tan(np.radians(self.hfov_deg) / 2.0)
            return view_width_m / self.image_width_px
        return None


def frontal_area(bbox_px: tuple[float, float], scale: ScaleReference,
                 fill_fraction: float = 0.55, rel_sigma: float = 0.20,
                 n: int = 6000, rng: np.random.Generator | None = None) -> tuple[Distribution, bool]:
    """Estimate frontal area (m^2) from a bounding box.

    `fill_fraction` accounts for the drone not filling its bounding rectangle
    (props + gaps). Returns (distribution, is_metric). When no scale is available,
    is_metric is False and the distribution is a wide ratio-only prior expressed
    in m^2 spanning plausible drone sizes — the honest wide-interval behaviour.
    """
    rng = rng or np.random.default_rng()
    w_px, h_px = bbox_px
    mpp = scale.resolve_m_per_pixel()

    if mpp is None:
        # No scale: cannot produce a metric value. Return a deliberately wide
        # prior over plausible multirotor frontal areas (~5cm to ~50cm box side).
        side = rng.uniform(0.05, 0.5, size=n)
        area = side**2 * fill_fraction
        return Distribution(area, unit="m^2"), False

    w_m = w_px * mpp * (1.0 + rng.normal(0, rel_sigma, size=n))
    h_m = h_px * mpp * (1.0 + rng.normal(0, rel_sigma, size=n))
    area = np.clip(np.abs(w_m * h_m) * fill_fraction, 1e-5, None)
    return Distribution(area, unit="m^2"), True


def frame_diagonal(bbox_px: tuple[float, float], scale: ScaleReference,
                   rel_sigma: float = 0.15, n: int = 6000,
                   rng: np.random.Generator | None = None) -> tuple[Distribution, bool]:
    """Frame diagonal (m) from the bounding-box diagonal. Ratio-only without scale."""
    rng = rng or np.random.default_rng()
    w_px, h_px = bbox_px
    diag_px = float(np.hypot(w_px, h_px))
    mpp = scale.resolve_m_per_pixel()
    if mpp is None:
        diag = rng.uniform(0.08, 1.2, size=n)  # plausible frame diagonals
        return Distribution(diag, unit="m"), False
    diag = diag_px * mpp * (1.0 + rng.normal(0, rel_sigma, size=n))
    return Distribution(np.clip(diag, 1e-3, None), unit="m"), True

"""Mass from drag — the only honest path to absolute mass outside hover.

Per spec §2 rule 3: mass comes from drag, which only appears in motion. Climb /
throttle acceleration is mass-INDEPENDENT and must never be used for mass.

Two maneuvers:
  - coast-down (preferred): power off, m*dv/dt = -drag, drag = 0.5*rho*Cd*A*v^2.
    Measure dv/dt and v from video, prior on Cd, area from image -> solve m.
  - steady cruise: at constant speed T*sin(theta) = drag, giving an absolute
    thrust independent of C_T, which then back-solves C_T.

Both return distributions (Monte Carlo over the Cd prior and measurement noise).
"""

from __future__ import annotations

import numpy as np

from augur.config import load_priors
from augur.types import Distribution


def coast_down_mass(deceleration_m_s2: float, velocity_m_s: float, frontal_area_m2: float,
                    area_rel_sigma: float = 0.15, decel_rel_sigma: float = 0.10,
                    n: int = 6000, rng: np.random.Generator | None = None) -> Distribution:
    """Mass from a power-off coast-down.

        m = drag / |dv/dt| = 0.5 * rho * Cd * A * v^2 / |dv/dt|

    deceleration is the magnitude of dv/dt (positive). Cd is drawn from the
    config drag-coefficient prior; A and the deceleration carry relative noise.
    """
    if deceleration_m_s2 <= 0:
        raise ValueError("deceleration must be positive (magnitude of dv/dt)")
    if velocity_m_s <= 0:
        raise ValueError("velocity must be positive")
    if frontal_area_m2 <= 0:
        raise ValueError("frontal_area must be positive")

    rng = rng or np.random.default_rng()
    priors = load_priors()
    cd_band = priors.drag_coefficient
    rho = priors.air_density

    cd = rng.uniform(cd_band.low, cd_band.high, size=n)
    area = frontal_area_m2 * (1.0 + rng.normal(0, area_rel_sigma, size=n))
    decel = deceleration_m_s2 * (1.0 + rng.normal(0, decel_rel_sigma, size=n))
    area = np.clip(area, 1e-4, None)
    decel = np.clip(decel, 1e-3, None)

    mass = 0.5 * rho * cd * area * velocity_m_s**2 / decel
    return Distribution(mass, unit="kg")


def cruise_absolute_thrust(tilt_angle_rad: float, velocity_m_s: float, frontal_area_m2: float,
                           area_rel_sigma: float = 0.15, n: int = 6000,
                           rng: np.random.Generator | None = None) -> Distribution:
    """Absolute thrust at steady cruise, independent of C_T.

        T * sin(theta) = drag = 0.5 * rho * Cd * A * v^2   ->   T = drag / sin(theta)

    This back-solves C_T elsewhere and sharpens every thrust/weight number."""
    if not 0 < tilt_angle_rad < np.pi / 2:
        raise ValueError("tilt_angle must be in (0, pi/2)")
    rng = rng or np.random.default_rng()
    priors = load_priors()
    cd = rng.uniform(priors.drag_coefficient.low, priors.drag_coefficient.high, size=n)
    area = np.clip(frontal_area_m2 * (1.0 + rng.normal(0, area_rel_sigma, size=n)), 1e-4, None)
    drag = 0.5 * priors.air_density * cd * area * velocity_m_s**2
    thrust = drag / np.sin(tilt_angle_rad)
    return Distribution(thrust, unit="N")


def true_coast_deceleration(mass_kg: float, velocity_m_s: float, frontal_area_m2: float,
                            cd: float, rho: float = 1.225) -> float:
    """Forward relation (oracle for tests): |dv/dt| = 0.5*rho*Cd*A*v^2 / m."""
    return 0.5 * rho * cd * frontal_area_m2 * velocity_m_s**2 / mass_kg

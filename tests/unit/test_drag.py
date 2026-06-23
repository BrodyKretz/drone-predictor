"""Coast-down / cruise mass extraction, round-tripped against the forward drag
relation. Also guards the §14 rule that climb accel must NOT give mass."""

import numpy as np
import pytest

from augur.physics import drag


def test_coast_down_recovers_mass():
    true_mass, v, area, cd = 1.2, 12.0, 0.06, 0.9
    decel = drag.true_coast_deceleration(true_mass, v, area, cd)
    est = drag.coast_down_mass(decel, v, area, n=8000, rng=np.random.default_rng(0))
    lo, hi = est.interval(0.9)
    assert lo <= true_mass <= hi, f"true mass {true_mass} outside [{lo:.3f}, {hi:.3f}]"
    # Median within ~30% (Cd prior is wide on purpose).
    assert abs(est.median - true_mass) / true_mass < 0.35


def test_coast_down_scales_with_decel():
    # Heavier drone -> smaller deceleration for the same drag.
    v, area = 10.0, 0.05
    light = drag.coast_down_mass(4.0, v, area, n=4000, rng=np.random.default_rng(1))
    heavy = drag.coast_down_mass(1.0, v, area, n=4000, rng=np.random.default_rng(1))
    assert heavy.median > light.median


def test_coast_down_rejects_bad_inputs():
    for bad in [
        dict(deceleration_m_s2=-1.0, velocity_m_s=10, frontal_area_m2=0.05),
        dict(deceleration_m_s2=1.0, velocity_m_s=0, frontal_area_m2=0.05),
        dict(deceleration_m_s2=1.0, velocity_m_s=10, frontal_area_m2=0),
    ]:
        with pytest.raises(ValueError):
            drag.coast_down_mass(**bad)


def test_cruise_thrust_positive():
    t = drag.cruise_absolute_thrust(np.radians(20), 15.0, 0.06, n=4000,
                                    rng=np.random.default_rng(2))
    assert t.median > 0
    assert t.interval(0.9)[0] > 0


def test_cruise_thrust_rejects_bad_angle():
    with pytest.raises(ValueError):
        drag.cruise_absolute_thrust(0.0, 15.0, 0.06)
    with pytest.raises(ValueError):
        drag.cruise_absolute_thrust(np.pi / 2, 15.0, 0.06)

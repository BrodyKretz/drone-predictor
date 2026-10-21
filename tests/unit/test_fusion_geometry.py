"""Frame diagonal and disk loading must carry motor-count uncertainty, not a
modal motor count (design rule 4: widen, never invent)."""

import numpy as np
import pytest

from augur.fusion.posterior import FusionInputs, fuse, fuse_distributions
from augur.types import VerbalSpec

D_5IN_M = 5.0 * 0.0254


@pytest.mark.parametrize("seed", [0, 1])
def test_frame_diagonal_uncertain_when_motor_count_unknown(seed):
    report = fuse(FusionInputs(verbal=VerbalSpec(prop_diameter_inch=5.0)), n=4000, seed=seed)
    frame = report.variables["min_frame_diagonal_m"]
    assert frame.interval_low == pytest.approx(D_5IN_M / np.sin(np.pi / 3), rel=1e-6)
    assert frame.interval_high == pytest.approx(D_5IN_M / np.sin(np.pi / 8), rel=1e-6)


def test_frame_diagonal_exact_when_motor_count_given():
    verbal = VerbalSpec(num_motors=4, prop_diameter_inch=5.0)
    frame = fuse(FusionInputs(verbal=verbal), n=2000, seed=0).variables["min_frame_diagonal_m"]
    assert frame.interval_low == frame.interval_high
    assert frame.median == pytest.approx(D_5IN_M / np.sin(np.pi / 4), rel=1e-6)


def test_disk_loading_uses_each_samples_motor_count():
    out = fuse_distributions(FusionInputs(verbal=VerbalSpec(prop_diameter_inch=5.0)), n=4000, seed=0)
    motors = out["num_motors"].samples
    expected = out["total_thrust_n"].samples / (motors * np.pi * (D_5IN_M / 2) ** 2)
    assert np.allclose(out["disk_loading_n_m2"].samples, expected)

"""Physics equations against hand-computed values and exact identities."""

import numpy as np
import pytest

from augur.physics import core


def test_bpf_rpm_round_trip():
    rpm = 9000.0
    bpf = core.bpf_from_rpm(rpm, num_blades=2)
    assert bpf == pytest.approx(300.0)  # 9000/60 * 2
    assert core.rpm_from_bpf(bpf, num_blades=2) == pytest.approx(rpm)


def test_commutation_freq():
    # 12N14P motor -> 7 pole pairs. 6000 rpm -> 100 rev/s -> 700 Hz.
    assert core.commutation_freq(6000.0, pole_pairs=7) == pytest.approx(700.0)


def test_thrust_hand_computed():
    # T = C_T * rho * n^2 * D^4 ; n=100 rev/s, D=0.254 m, C_T=0.1, rho=1.225
    d = 0.254
    t = core.thrust_per_motor(6000.0, d, c_t=0.1, rho=1.225)
    expected = 0.1 * 1.225 * 100**2 * d**4
    assert float(t) == pytest.approx(expected)


def test_shaft_power_hand_computed():
    d = 0.254
    p = core.shaft_power_per_motor(6000.0, d, c_p=0.05, rho=1.225)
    expected = 0.05 * 1.225 * 100**3 * d**5
    assert float(p) == pytest.approx(expected)


def test_electrical_power_efficiency():
    assert float(core.electrical_power(100.0, 0.8)) == pytest.approx(125.0)
    with pytest.raises(ValueError):
        core.electrical_power(100.0, 1.5)


def test_disk_loading():
    # 4 motors, D=0.254, total thrust 40 N
    dl = core.disk_loading(40.0, 4, 0.254)
    disk_area = 4 * np.pi * (0.254 / 2) ** 2
    assert float(dl) == pytest.approx(40.0 / disk_area)


def test_thrust_ratio_cancels_ct():
    # The key identity: T2/T1 = (n2/n1)^2, independent of C_T, rho, D.
    rho, d = 1.225, 0.254
    t1 = core.thrust_per_motor(5000.0, d, c_t=0.11, rho=rho)
    t2 = core.thrust_per_motor(8000.0, d, c_t=0.11, rho=rho)
    assert float(t2 / t1) == pytest.approx((8000.0 / 5000.0) ** 2)
    # And the dedicated helper agrees and needs no C_T at all.
    assert float(core.thrust_to_weight_from_rpm_ratio(8000.0, 5000.0)) == pytest.approx((8000 / 5000) ** 2)


def test_hover_mass():
    # sum(T) = m g -> m = T/g
    assert float(core.hover_mass(19.6133, 9.80665)) == pytest.approx(2.0)


def test_min_frame_diagonal_quad():
    # Quad (N=4): radius = D / (2 sin(45deg)) = D/sqrt(2); diagonal = 2R = D*sqrt(2)
    d = 0.254
    diag = core.min_frame_diagonal_m(d, 4)
    assert float(diag) == pytest.approx(d * np.sqrt(2))


def test_min_frame_requires_two_motors():
    with pytest.raises(ValueError):
        core.min_frame_diagonal_m(0.254, 1)


def test_array_broadcasting():
    rpms = np.array([5000.0, 6000.0, 7000.0])
    t = core.thrust_per_motor(rpms, 0.254, c_t=0.1, rho=1.225)
    assert t.shape == (3,)
    assert np.all(np.diff(t) > 0)  # monotonic in RPM

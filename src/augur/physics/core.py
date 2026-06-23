"""Physics core: thrust, power, disk loading, thrust-to-weight, frame geometry.

All equations use SI units internally (metres, kg, seconds, Newtons, Watts).
Functions accept either scalars (for hand-checkable unit tests) or numpy arrays
(for Monte Carlo propagation). Diameters are in metres unless the name says inch.

Reference equations (per motor, momentum/blade-element form):
    n  = RPM / 60                       rev/s
    T  = C_T * rho * n^2 * D^4          thrust, Newtons
    P  = C_P * rho * n^3 * D^5          shaft power, Watts
    P_elec = P / eta                    electrical power, Watts
    DL = sum(T) / (N * pi * (D/2)^2)    disk loading, N/m^2
"""

from __future__ import annotations

import numpy as np

INCH_TO_M = 0.0254
ArrayLike = float | np.ndarray


def rpm_to_rev_per_s(rpm: ArrayLike) -> ArrayLike:
    return np.asarray(rpm, dtype=float) / 60.0


def bpf_from_rpm(rpm: ArrayLike, num_blades: int) -> ArrayLike:
    """Blade pass frequency (the dominant acoustic tone), Hz."""
    if num_blades <= 0:
        raise ValueError("num_blades must be positive")
    return rpm_to_rev_per_s(rpm) * num_blades


def rpm_from_bpf(bpf: ArrayLike, num_blades: int) -> ArrayLike:
    """Primary, high-confidence RPM recovery from the blade pass frequency."""
    if num_blades <= 0:
        raise ValueError("num_blades must be positive")
    return np.asarray(bpf, dtype=float) * 60.0 / num_blades


def commutation_freq(rpm: ArrayLike, pole_pairs: int) -> ArrayLike:
    """Electrical commutation tone, Hz. pole_pairs = pole_count / 2."""
    if pole_pairs <= 0:
        raise ValueError("pole_pairs must be positive")
    return rpm_to_rev_per_s(rpm) * pole_pairs


def thrust_per_motor(rpm: ArrayLike, diameter_m: ArrayLike, c_t: ArrayLike, rho: float) -> ArrayLike:
    n = rpm_to_rev_per_s(rpm)
    return np.asarray(c_t) * rho * n**2 * np.asarray(diameter_m) ** 4


def shaft_power_per_motor(rpm: ArrayLike, diameter_m: ArrayLike, c_p: ArrayLike, rho: float) -> ArrayLike:
    n = rpm_to_rev_per_s(rpm)
    return np.asarray(c_p) * rho * n**3 * np.asarray(diameter_m) ** 5


def electrical_power(shaft_power: ArrayLike, efficiency: ArrayLike) -> ArrayLike:
    eta = np.asarray(efficiency, dtype=float)
    if np.any(eta <= 0) or np.any(eta > 1):
        raise ValueError("efficiency must be in (0, 1]")
    return np.asarray(shaft_power) / eta


def total_thrust(thrust_per_motor_val: ArrayLike, num_motors: int) -> ArrayLike:
    if num_motors <= 0:
        raise ValueError("num_motors must be positive")
    return np.asarray(thrust_per_motor_val) * num_motors


def disk_loading(total_thrust_val: ArrayLike, num_motors: int, diameter_m: ArrayLike) -> ArrayLike:
    """Total thrust divided by total swept disk area. N/m^2.

    A strong class/efficiency proxy: endurance scales as ~sqrt(1/DL)."""
    if num_motors <= 0:
        raise ValueError("num_motors must be positive")
    disk_area = num_motors * np.pi * (np.asarray(diameter_m) / 2.0) ** 2
    return np.asarray(total_thrust_val) / disk_area


def hover_mass(total_thrust_val: ArrayLike, gravity: float) -> ArrayLike:
    """Mass from the hover force balance sum(T) = m*g.

    VALID ONLY IN HOVER. The caller must confirm hover state (steady RPM,
    ~zero vertical velocity) before using this — outside hover it is wrong."""
    return np.asarray(total_thrust_val) / gravity


def thrust_to_weight_from_rpm_ratio(rpm_max: ArrayLike, rpm_hover: ArrayLike) -> ArrayLike:
    """T/W from the RPM ratio alone.

    Between two states of one drone, T2/T1 = (n2/n1)^2 exactly — C_T, rho and D
    all cancel. So T_max/W = (n_max/n_hover)^2. This is the clean, high-confidence
    path to thrust-to-weight; it needs no coefficient or air-density assumption."""
    ratio = np.asarray(rpm_max, dtype=float) / np.asarray(rpm_hover, dtype=float)
    return ratio**2


def min_frame_diagonal_m(diameter_m: ArrayLike, num_motors: int) -> ArrayLike:
    """Lower bound on frame diagonal from prop geometry: props must not overlap.

    For an N-motor ring with props of diameter D, adjacent prop centres must be
    at least D apart. The motors sit on a circle of radius R = D / (2 sin(pi/N));
    the diagonal is 2R. This is a *minimum* — real frames are larger."""
    if num_motors < 2:
        raise ValueError("num_motors must be >= 2 for a frame")
    d = np.asarray(diameter_m, dtype=float)
    radius = d / (2.0 * np.sin(np.pi / num_motors))
    return 2.0 * radius

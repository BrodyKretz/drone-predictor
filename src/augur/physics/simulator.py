"""Forward simulator — the test oracle for the whole system.

Given a fully specified drone + flight state, produce (a) a synthetic acoustic
spectrum/signal and (b) the ground-truth value of every output variable. The
inverse pipeline must recover these inputs; round-trip tests are built on this.

Build order rationale: this exists before any inverse module so they always have
something to validate against and an infinite labeled-data source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from augur.physics import core


@dataclass
class DroneSpec:
    """A complete physical specification — everything the forward model needs."""

    num_motors: int
    prop_diameter_inch: float
    prop_pitch_inch: float
    blade_count: int
    c_t: float
    c_p: float
    pole_count: int
    efficiency: float
    mass_kg: float
    battery_wh: float
    drone_class: str = "unknown"
    payload_mass_kg: float = 0.0

    @property
    def diameter_m(self) -> float:
        return self.prop_diameter_inch * core.INCH_TO_M

    @property
    def pole_pairs(self) -> int:
        return self.pole_count // 2


@dataclass
class FlightCondition:
    """The state the drone is in for this capture."""

    rpm: float  # mean RPM across motors
    rpm_spread_fraction: float = 0.0  # per-motor offset for control
    vertical_velocity_m_s: float = 0.0
    forward_velocity_m_s: float = 0.0
    state: str = "hover"


@dataclass
class GroundTruth:
    """The true value of every output variable, for a (spec, condition) pair."""

    rpm: float
    rpm_per_motor: list[float]
    bpf_hz: float
    commutation_hz: float
    thrust_per_motor_n: float
    total_thrust_n: float
    weight_n: float
    mass_kg: float
    shaft_power_w: float
    electrical_power_w: float
    disk_loading_n_m2: float
    min_frame_diagonal_m: float
    thrust_to_weight: float
    battery_wh: float
    endurance_s: float
    drone_class: str


@dataclass
class SimulatedSample:
    spec: DroneSpec
    condition: FlightCondition
    truth: GroundTruth
    freqs_hz: np.ndarray = field(repr=False)
    spectrum_db: np.ndarray = field(repr=False)


def compute_truth(spec: DroneSpec, cond: FlightCondition, rho: float = 1.225, g: float = 9.80665) -> GroundTruth:
    rpm = cond.rpm
    d = spec.diameter_m

    t_per = float(core.thrust_per_motor(rpm, d, spec.c_t, rho))
    t_total = t_per * spec.num_motors
    p_shaft = float(core.shaft_power_per_motor(rpm, d, spec.c_p, rho)) * spec.num_motors
    p_elec = p_shaft / spec.efficiency
    dl = float(core.disk_loading(t_total, spec.num_motors, d))
    weight = spec.mass_kg * g
    twr = t_total / weight

    # Endurance from the energy budget at this operating point.
    endurance = (spec.battery_wh / p_elec) * 3600.0 if p_elec > 0 else float("inf")

    # Per-motor RPM offsets, symmetric around the mean.
    spread = cond.rpm_spread_fraction
    if spec.num_motors > 1 and spread > 0:
        offsets = np.linspace(-spread, spread, spec.num_motors)
    else:
        offsets = np.zeros(spec.num_motors)
    rpm_per_motor = [rpm * (1.0 + o) for o in offsets]

    return GroundTruth(
        rpm=rpm,
        rpm_per_motor=rpm_per_motor,
        bpf_hz=float(core.bpf_from_rpm(rpm, spec.blade_count)),
        commutation_hz=float(core.commutation_freq(rpm, spec.pole_pairs)),
        thrust_per_motor_n=t_per,
        total_thrust_n=t_total,
        weight_n=weight,
        mass_kg=spec.mass_kg,
        shaft_power_w=p_shaft,
        electrical_power_w=p_elec,
        disk_loading_n_m2=dl,
        min_frame_diagonal_m=float(core.min_frame_diagonal_m(d, spec.num_motors)),
        thrust_to_weight=twr,
        battery_wh=spec.battery_wh,
        endurance_s=endurance,
        drone_class=spec.drone_class,
    )


def synth_spectrum(
    spec: DroneSpec,
    cond: FlightCondition,
    sample_rate: int = 44100,
    duration_s: float = 1.0,
    n_harmonics: int = 6,
    noise_db: float = -40.0,
    mic_distance_m: float = 3.0,
    sound_speed_m_s: float = 343.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a magnitude spectrum (in dB) with BPF + harmonics, the
    electrical commutation tone, per-motor spread, optional Doppler shift from
    forward velocity, and a broadband noise floor.

    Returns (freqs_hz, spectrum_db).
    """
    rng = rng or np.random.default_rng()
    n_fft = int(sample_rate * duration_s)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    spectrum = np.zeros_like(freqs)

    # Doppler factor for an approaching/receding source along the mic axis.
    v = cond.forward_velocity_m_s
    doppler = sound_speed_m_s / (sound_speed_m_s - v) if abs(v) < sound_speed_m_s else 1.0

    spread = cond.rpm_spread_fraction
    if spec.num_motors > 1 and spread > 0:
        offsets = np.linspace(-spread, spread, spec.num_motors)
    else:
        offsets = np.zeros(spec.num_motors)

    def add_tone(f_hz: float, amp: float):
        if f_hz <= 0 or f_hz >= freqs[-1]:
            return
        bin_idx = int(round(f_hz / (sample_rate / n_fft)))
        if 0 <= bin_idx < spectrum.size:
            spectrum[bin_idx] += amp

    for off in offsets:
        rpm_m = cond.rpm * (1.0 + off)
        bpf = float(core.bpf_from_rpm(rpm_m, spec.blade_count)) * doppler
        for h in range(1, n_harmonics + 1):
            add_tone(bpf * h, amp=1.0 / h)  # harmonics roll off as 1/h
        comm = float(core.commutation_freq(rpm_m, spec.pole_pairs)) * doppler
        add_tone(comm, amp=0.3)

    # Distance attenuation (inverse-square in power) folded into amplitude.
    spectrum /= max(mic_distance_m, 0.1)

    noise_lin = 10.0 ** (noise_db / 20.0)
    spectrum += rng.uniform(0, noise_lin, size=spectrum.size)

    spectrum_db = 20.0 * np.log10(spectrum + 1e-12)
    return freqs, spectrum_db


def synth_waveform(
    spec: DroneSpec,
    cond: FlightCondition,
    sample_rate: int = 44100,
    duration_s: float = 2.0,
    n_harmonics: int = 6,
    snr_db: float = 20.0,
    rpm_drift_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Time-domain signal: summed sinusoids at BPF harmonics + commutation tone
    per motor, plus broadband noise at the requested SNR. `rpm_drift_fraction`
    linearly ramps RPM over the clip (use to simulate climb/throttle changes).

    Returns (samples, sample_rate).
    """
    rng = rng or np.random.default_rng()
    t = np.arange(int(sample_rate * duration_s)) / sample_rate

    spread = cond.rpm_spread_fraction
    if spec.num_motors > 1 and spread > 0:
        offsets = np.linspace(-spread, spread, spec.num_motors)
    else:
        offsets = np.zeros(spec.num_motors)

    # Instantaneous RPM trajectory (optional linear drift).
    rpm_t = cond.rpm * (1.0 + rpm_drift_fraction * (t / max(t[-1], 1e-9)))

    signal = np.zeros_like(t)
    for off in offsets:
        rpm_m = rpm_t * (1.0 + off)
        bpf_t = core.bpf_from_rpm(rpm_m, spec.blade_count)
        phase = 2.0 * np.pi * np.cumsum(bpf_t) / sample_rate
        for h in range(1, n_harmonics + 1):
            signal += (1.0 / h) * np.sin(h * phase)
        comm_t = core.commutation_freq(rpm_m, spec.pole_pairs)
        comm_phase = 2.0 * np.pi * np.cumsum(comm_t) / sample_rate
        signal += 0.3 * np.sin(comm_phase)

    signal /= max(spec.num_motors, 1)
    sig_power = float(np.mean(signal**2)) or 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    signal += rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal.astype(np.float64), sample_rate


def simulate(
    spec: DroneSpec,
    cond: FlightCondition,
    rho: float = 1.225,
    g: float = 9.80665,
    **spectrum_kwargs,
) -> SimulatedSample:
    truth = compute_truth(spec, cond, rho=rho, g=g)
    freqs, spectrum_db = synth_spectrum(spec, cond, **spectrum_kwargs)
    return SimulatedSample(spec=spec, condition=cond, truth=truth, freqs_hz=freqs, spectrum_db=spectrum_db)

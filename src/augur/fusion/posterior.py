"""Monte Carlo fusion → per-variable posteriors.

Generative scheme:
  1. Sample latent drone parameters from priors, fixing any the user asserted
     (verbal) and using the audio RPM estimate as the proposal for RPM.
  2. Run the physics forward for every sample.
  3. Weight samples by the Gaussian likelihood of every soft Observation
     (image/video), then resample to get posterior samples per variable.

Adding observations can only concentrate the weights, so intervals can only
narrow given consistent evidence — the monotonicity the system promises.

Per-modality attribution is computed by re-running the sampler on cumulative
input subsets ([], +verbal, +audio, +image, +video) and recording each
variable's 90% interval width before and after each modality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from augur.audio.rpm import RPMEstimate
from augur.config import Classes, Priors, load_classes, load_priors
from augur.fusion.observations import Observation
from augur.physics import core
from augur.types import (
    Attribution,
    Distribution,
    FlightState,
    PredictionReport,
    VerbalSpec,
)

# Output variables the fusion reports, with display units.
OUTPUT_UNITS = {
    "rpm": "rpm",
    "thrust_per_motor_n": "N",
    "total_thrust_n": "N",
    "mass_kg": "kg",
    "weight_n": "N",
    "shaft_power_w": "W",
    "electrical_power_w": "W",
    "disk_loading_n_m2": "N/m^2",
    "min_frame_diagonal_m": "m",
    "thrust_to_weight": "",
    "battery_wh": "Wh",
    "endurance_s": "s",
    "num_motors": "",
    "prop_diameter_inch": "in",
}


@dataclass
class FusionInputs:
    """Bundle of everything the fusion can use. Any field may be None/empty."""

    verbal: VerbalSpec | None = None
    rpm_estimate: RPMEstimate | None = None
    flight_state: FlightState = FlightState.unknown
    image_observations: list[Observation] = field(default_factory=list)
    video_observations: list[Observation] = field(default_factory=list)

    def sources_present(self) -> list[str]:
        s = []
        if self.verbal is not None:
            s.append("verbal")
        if self.rpm_estimate is not None:
            s.append("audio")
        if self.image_observations:
            s.append("image")
        if self.video_observations:
            s.append("video")
        return s


def _scale(u: np.ndarray, band) -> np.ndarray:
    """Scale a unit-uniform array into a [low, high] band. Using a shared unit
    draw across fusion stages gives common random numbers: narrowing a band
    rescales the *same* points, so interval widths change only from real
    information, not RNG noise."""
    return band.low + (band.high - band.low) * u


def _scale_lh(u: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (high - low) * u


def _sample_and_forward(inputs: FusionInputs, n: int, priors: Priors, classes: Classes,
                        rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Sample latents (respecting verbal fixes + audio proposal), run forward.

    Every latent is drawn from a unit uniform in a FIXED order regardless of which
    inputs are present, then scaled to its band and overridden by asserted facts.
    This keeps the RNG stream aligned across input subsets (common random numbers),
    which is what makes the monotonicity guarantee hold to Monte Carlo precision.
    """
    verbal = inputs.verbal
    class_name = (verbal.drone_class.value if verbal and verbal.drone_class else "unknown")
    cls = classes.get(class_name)

    # Draw all unit uniforms up front, in a fixed order.
    u_motor = rng.random(n)
    u_diam = rng.random(n)
    u_blade = rng.random(n)
    u_pitch = rng.random(n)
    u_ct = rng.random(n)  # placeholder draws to keep stream aligned with db path
    u_cp = rng.random(n)
    u_eff = rng.random(n)
    u_ed = rng.random(n)
    u_bf = rng.random(n)
    u_rpm = rng.random(n)
    u_twr = rng.random(n)
    u_mass = rng.random(n)

    # --- num_motors ---
    if verbal and verbal.num_motors:
        num_motors = np.full(n, verbal.num_motors, dtype=int)
    else:
        choices = np.array(cls["num_motors"])
        num_motors = choices[np.floor(u_motor * len(choices)).astype(int)]
    num_motors_mode = int(np.bincount(num_motors).argmax())

    # --- prop diameter (inches) ---
    if verbal and verbal.prop_diameter_inch:
        diameter_in = np.full(n, verbal.prop_diameter_inch)
    else:
        b = cls["prop_diameter_inch"]
        diameter_in = _scale_lh(u_diam, b["low"], b["high"])
    diameter_m = diameter_in * core.INCH_TO_M

    # --- blade count (unused downstream here, but drawn for stream alignment) ---
    if not (verbal and verbal.blade_count):
        _ = np.array([2, 3])[np.floor(u_blade * 2).astype(int)]

    # --- pitch (via pitch/diameter ratio prior) ---
    _pitch = _scale(u_pitch, priors.pitch_to_diameter(class_name)) * diameter_in

    # --- coefficients (fallback bands; same #draws as the db path) ---
    c_t = _scale(u_ct, priors.c_t_fallback)
    c_p = _scale(u_cp, priors.c_p_fallback)
    efficiency = _scale(u_eff, priors.efficiency)
    energy_density = _scale(u_ed, priors.energy_density("lipo"))
    batt_frac = _scale(u_bf, priors.battery_mass_fraction(class_name))

    # --- RPM: audio proposal if present, else wide prior ---
    if inputs.rpm_estimate is not None:
        src = inputs.rpm_estimate.rpm.samples
        idx = np.floor(u_rpm * len(src)).astype(int)
        rpm = src[idx]
    else:
        rpm = _scale_lh(u_rpm, 2000.0, 28000.0)

    rho, g = priors.air_density, priors.gravity

    # --- forward physics ---
    t_per = core.thrust_per_motor(rpm, diameter_m, c_t, rho)
    t_total = t_per * num_motors
    p_shaft = core.shaft_power_per_motor(rpm, diameter_m, c_p, rho) * num_motors
    p_elec = p_shaft / efficiency
    disk = core.disk_loading(t_total, num_motors_mode, diameter_m)
    frame = core.min_frame_diagonal_m(diameter_m, num_motors_mode)

    # --- mass: hover force balance when in hover, else class prior ---
    in_hover = inputs.flight_state in (FlightState.hover, FlightState.unknown)
    b = cls["all_up_mass_kg"]
    mass_prior = _scale_lh(u_mass, b["low"], b["high"])
    if in_hover:
        mass = t_total / g  # sum(T) = m g  (valid in hover only)
    else:
        mass = mass_prior
    weight = mass * g

    # --- thrust-to-weight: class prior unless video confirms it ---
    twr_b = cls["thrust_to_weight"]
    twr = _scale_lh(u_twr, twr_b["low"], twr_b["high"])

    # --- battery: residual path (mass fraction * energy density) ---
    battery_wh = mass * batt_frac * energy_density
    endurance = np.where(p_elec > 0, battery_wh / p_elec * 3600.0, np.inf)

    outputs = {
        "rpm": rpm,
        "thrust_per_motor_n": np.asarray(t_per),
        "total_thrust_n": np.asarray(t_total),
        "mass_kg": np.asarray(mass),
        "weight_n": np.asarray(weight),
        "shaft_power_w": np.asarray(p_shaft),
        "electrical_power_w": np.asarray(p_elec),
        "disk_loading_n_m2": np.asarray(disk),
        "min_frame_diagonal_m": np.asarray(frame),
        "thrust_to_weight": np.asarray(twr),
        "battery_wh": np.asarray(battery_wh),
        "endurance_s": np.asarray(endurance),
        "num_motors": num_motors.astype(float),
        "prop_diameter_inch": diameter_in,
    }
    return outputs


def _log_likelihood(outputs: dict[str, np.ndarray], obs_list: list[Observation]) -> np.ndarray:
    n = len(next(iter(outputs.values())))
    logl = np.zeros(n)
    for obs in obs_list:
        if obs.variable not in outputs:
            continue
        resid = (outputs[obs.variable] - obs.value) / obs.sigma
        logl += -0.5 * resid**2
    return logl


def _resample(outputs: dict[str, np.ndarray], obs_list: list[Observation],
              rng: np.random.Generator) -> dict[str, np.ndarray]:
    if not obs_list:
        return outputs
    logl = _log_likelihood(outputs, obs_list)
    w = np.exp(logl - logl.max())
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        return outputs
    w /= total
    n = len(logl)
    idx = rng.choice(n, size=n, p=w, replace=True)
    return {k: v[idx] for k, v in outputs.items()}


def _run(inputs: FusionInputs, n: int, priors: Priors, classes: Classes,
         rng: np.random.Generator) -> dict[str, np.ndarray]:
    outputs = _sample_and_forward(inputs, n, priors, classes, rng)
    soft = list(inputs.image_observations) + list(inputs.video_observations)
    return _resample(outputs, soft, rng)


def _subset(inputs: FusionInputs, sources: set[str]) -> FusionInputs:
    return FusionInputs(
        verbal=inputs.verbal if "verbal" in sources else None,
        rpm_estimate=inputs.rpm_estimate if "audio" in sources else None,
        flight_state=inputs.flight_state if "audio" in sources else FlightState.unknown,
        image_observations=inputs.image_observations if "image" in sources else [],
        video_observations=inputs.video_observations if "video" in sources else [],
    )


def fuse(inputs: FusionInputs, n: int = 8000, seed: int = 0) -> PredictionReport:
    """Fuse all present inputs into a PredictionReport with per-modality
    attribution of interval narrowing."""
    priors, classes = load_priors(), load_classes()
    present = inputs.sources_present()

    # Cumulative attribution stages in a fixed modality order.
    order = [s for s in ("verbal", "audio", "image", "video") if s in present]
    cumulative: set[str] = set()
    attribution: list[Attribution] = []

    # Baseline (priors only). All stages use the SAME seed (common random numbers)
    # so width changes reflect information gain, not RNG noise.
    base_out = _run(_subset(inputs, set()), n, priors, classes, np.random.default_rng(seed))
    prev_w = {v: _width(base_out[v]) for v in OUTPUT_UNITS}
    prev_m = {v: float(np.median(base_out[v])) for v in OUTPUT_UNITS}
    final_outputs = base_out

    for src in order:
        cumulative.add(src)
        out = _run(_subset(inputs, set(cumulative)), n, priors, classes,
                   np.random.default_rng(seed))
        for v in OUTPUT_UNITS:
            w_after = _width(out[v])
            m_after = float(np.median(out[v]))
            attribution.append(Attribution(
                variable=v, source=src,
                width_before=prev_w[v], width_after=w_after,
                median_before=prev_m[v], median_after=m_after,
            ))
            prev_w[v], prev_m[v] = w_after, m_after
        final_outputs = out

    variables = {
        v: Distribution(final_outputs[v], unit=OUTPUT_UNITS[v] or None).to_summary()
        for v in OUTPUT_UNITS
    }

    return PredictionReport(
        variables=variables,
        attribution=attribution,
        assumptions=_assumptions(inputs, priors),
        inputs_used=present,
        config_version=priors.version,
        notes=_honesty_notes(inputs),
    )


def _width(arr: np.ndarray) -> float:
    lo, hi = np.quantile(arr, [0.05, 0.95])
    return float(hi - lo)


def _assumptions(inputs: FusionInputs, priors: Priors) -> list[str]:
    cls = (inputs.verbal.drone_class.value if inputs.verbal and inputs.verbal.drone_class
           else "unknown")
    eff = priors.efficiency
    ed = priors.energy_density("lipo")
    bmf = priors.battery_mass_fraction(cls)
    used_hover = inputs.flight_state in (FlightState.hover, FlightState.unknown)
    mass_line = (
        f"mass from hover force balance sum(T)=m*g (state={inputs.flight_state.value})"
        if used_hover else
        f"mass from class prior — not in hover (state={inputs.flight_state.value}), "
        f"so thrust does not equal weight"
    )
    return [
        f"motor/ESC efficiency in [{eff.low}, {eff.high}]",
        f"LiPo energy density in [{ed.low}, {ed.high}] Wh/kg",
        f"battery mass fraction ({cls}) in [{bmf.low}, {bmf.high}]",
        "C_T/C_P from config fallback bands (prop_db.parquet not yet ingested)",
        mass_line,
    ]


def _honesty_notes(inputs: FusionInputs) -> list[str]:
    notes = []
    has_battery_obs = any(o.variable == "battery_wh" for o in inputs.image_observations)
    if not has_battery_obs:
        notes.append("Battery/endurance are residual estimates (pack not read from "
                     "image) — wide by design; see §14 irreducible floor.")
    if inputs.rpm_estimate is None:
        notes.append("No audio: RPM unconstrained, all thrust/power/mass intervals "
                     "are prior-width only.")
    if inputs.video_observations == []:
        notes.append("No video: thrust-to-weight is a class prior, not measured.")
    return notes

"""Live identification: which kind of drone is this, from audio alone?

In hover, a class's priors (prop size, motor count, all-up mass) plus the prop
coefficients imply a range of blade-pass frequencies. Each hypothesis (class c,
blade count B) reuses the plug-in fusion: RPM = BPF * 60 / B, latents from c.
The class's all-up-mass band is a uniform prior on mass, and hover gives
mass = k * rpm^2, so the evidence for (c, B) is

    p(BPF | c, B) = (60 / B) * E[ 1{m in band} * 2 m / (rpm * (hi - lo)) ]

and the same term is the importance weight that turns the plug-in samples into
the posterior under (c, B). Facts the user asserts (prop size, motor count) add
their likelihood under each class's band. Estimates are the evidence-weighted
mixture. Valid only in hover (rule 6); otherwise only asserted facts move the
classes off their prior.
"""

from __future__ import annotations

import numpy as np

from augur.audio import diagnostics, hover, rpm, spectral
from augur.config import load_classes, load_priors
from augur.fusion.posterior import OUTPUT_UNITS, FusionInputs, fuse_distributions
from augur.types import (
    BladeProbability,
    ClassProbability,
    Distribution,
    FlightState,
    IdentifyReport,
    VariableSummary,
    VerbalSpec,
)

MAINS_HZ = (50.0, 60.0, 100.0, 120.0)
MAINS_TOLERANCE = 0.015
# Noise rejection depends on how many 8192-sample STFT frames get averaged, not on seconds.
MIN_WINDOW_SAMPLES = 88200


def candidate_classes(verbal: VerbalSpec | None) -> list[str]:
    if verbal and verbal.drone_class:
        return [verbal.drone_class.value]
    return [name for name in load_classes().names if name != "unknown"]


def candidate_blades(verbal: VerbalSpec | None) -> list[int]:
    if verbal and verbal.blade_count:
        return [verbal.blade_count]
    return load_priors().blade_count_options


def fact_likelihood(class_cfg: dict, verbal: VerbalSpec | None) -> float:
    """p(asserted prop size and motor count | class): uniform over the class's
    band or choices, zero outside them."""
    p = 1.0
    if verbal and verbal.prop_diameter_inch:
        band = class_cfg["prop_diameter_inch"]
        inside = band["low"] <= verbal.prop_diameter_inch <= band["high"]
        p *= 1.0 / (band["high"] - band["low"]) if inside else 0.0
    if verbal and verbal.num_motors:
        choices = class_cfg["num_motors"]
        p *= 1.0 / len(choices) if verbal.num_motors in choices else 0.0
    return p


def mains_hum_note(fundamental_hz: float, bin_hz: float) -> list[str]:
    for mains in MAINS_HZ:
        tolerance = max(MAINS_TOLERANCE * mains, bin_hz)
        if fundamental_hz > 0 and abs(fundamental_hz - mains) <= tolerance:
            return [f"The tone sits at {fundamental_hz:.0f} Hz, where mains hum lives "
                    "(50/60 Hz and multiples). It may not be a drone."]
    return []


def _summaries(draws: dict[str, np.ndarray]) -> dict[str, VariableSummary]:
    return {v: Distribution(a, unit=OUTPUT_UNITS[v] or None).to_summary() for v, a in draws.items()}


def _prior_width_estimates(verbal: VerbalSpec | None, n: int,
                           seed: int) -> dict[str, VariableSummary]:
    out = fuse_distributions(FusionInputs(verbal=verbal, assume_hover_when_unknown=False),
                             n=n, seed=seed)
    return _summaries({v: d.samples for v, d in out.items()})


def _ranked_classes(weights: dict[str, float]) -> list[ClassProbability]:
    total = sum(weights.values())
    probs = {c: (w / total if total > 0 else 1.0 / len(weights)) for c, w in weights.items()}
    return sorted((ClassProbability(name=c, probability=p) for c, p in probs.items()),
                  key=lambda x: -x.probability)


def _ranked_blades(weights: dict[int, float]) -> list[BladeProbability]:
    total = sum(weights.values())
    probs = {b: (w / total if total > 0 else 1.0 / len(weights)) for b, w in weights.items()}
    return sorted((BladeProbability(blades=b, probability=p) for b, p in probs.items()),
                  key=lambda x: -x.probability)


def _sample_counts(probabilities: list[float], n: int) -> list[int]:
    """Largest-remainder split of n samples: floor each share, then one more to
    each of the largest fractional remainders (ties go to the earlier entry)."""
    shares = np.asarray(probabilities) * n
    counts = np.floor(shares).astype(int)
    leftover = n - int(counts.sum())
    counts[np.argsort(counts - shares, kind="stable")[:leftover]] += 1
    return counts.tolist()


def identify(samples: np.ndarray, sample_rate: int, verbal: VerbalSpec | None = None,
             n: int = 4000, seed: int = 0) -> IdentifyReport:
    if not isinstance(samples, np.ndarray):
        raise ValueError(f"samples must be a numpy array, got {type(samples).__name__}")
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1-D (mono), got shape {samples.shape}")
    if samples.size == 0:
        raise ValueError("samples is empty: need at least one audio sample")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if samples.size < MIN_WINDOW_SAMPLES:
        raise ValueError(f"need at least {MIN_WINDOW_SAMPLES} samples of audio "
                         f"({MIN_WINDOW_SAMPLES / sample_rate:.1f} s at {sample_rate} Hz)")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be an integer, got {n!r}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    evidence = hover.assess_flight_state(samples, sample_rate)
    freqs, spec_db = spectral.average_spectrum(samples, sample_rate)
    f0, conf, peaks = rpm.detect_fundamental(freqs, spec_db)
    has_tone = f0 > 0 and conf >= rpm.MIN_COMB_CONFIDENCE
    class_cfg = load_classes()
    classes, blades = candidate_classes(verbal), candidate_blades(verbal)
    facts = {c: fact_likelihood(class_cfg.get(c), verbal) for c in classes}
    acoustics = diagnostics.build_acoustics(freqs, spec_db, f0 if has_tone else 0.0, peaks, conf,
                                            evidence)
    common = dict(duration_s=samples.size / sample_rate, sample_rate=sample_rate,
                  acoustics=acoustics, config_version=load_priors().version)
    notes = mains_hum_note(f0, float(freqs[1] - freqs[0])) if has_tone else []
    ruled_out = [ClassProbability(name=c, probability=0.0) for c in classes]
    prior_blades = _ranked_blades(dict.fromkeys(blades, 1.0))

    if not has_tone:
        notes.append("No steady blade-pass tone yet, so there is nothing to classify. "
                     "Estimates are prior-width.")
    if sum(facts.values()) == 0:
        notes.append("The facts you gave don't match any known class in config/classes.yaml.")
        return IdentifyReport(**common, status="no_class_fits", classes=ruled_out,
                              blades=prior_blades,
                              variables=_prior_width_estimates(verbal, n, seed), notes=notes)
    if not has_tone:
        return IdentifyReport(**common, status="no_tone", classes=_ranked_classes(facts),
                              blades=prior_blades,
                              variables=_prior_width_estimates(verbal, n, seed), notes=notes)

    hovering = evidence.state == FlightState.hover
    components = []
    for c in classes:
        band = class_cfg.get(c)["all_up_mass_kg"]
        for b in blades:
            est = rpm.estimate_rpm(freqs, spec_db, num_blades=b, rng=np.random.default_rng(seed))
            spec_cb = VerbalSpec(**{**(verbal.model_dump() if verbal else {}),
                                    "drone_class": c, "blade_count": b})
            out = fuse_distributions(FusionInputs(verbal=spec_cb, rpm_estimate=est,
                                                  flight_state=evidence.state), n=n, seed=seed)
            draws = {v: d.samples for v, d in out.items()}
            if hovering:
                m, r = draws["mass_kg"], draws["rpm"]
                inside = (m >= band["low"]) & (m <= band["high"])
                weights = inside * (2.0 * m / r) / (band["high"] - band["low"])
                strength = facts[c] * (60.0 / b) * float(weights.mean())
            else:
                weights, strength = np.ones(n), facts[c]
            components.append((c, b, strength, weights, draws))

    total = sum(strength for _, _, strength, _, _ in components)
    if total <= 0:
        notes.append("The tone doesn't fit any known class in hover. It may be a drone "
                     "type outside config/classes.yaml.")
        return IdentifyReport(**common, status="no_class_fits", classes=ruled_out,
                              blades=prior_blades,
                              variables=_prior_width_estimates(verbal, n, seed), notes=notes)

    probabilities = [strength / total for _, _, strength, _, _ in components]
    counts = _sample_counts(probabilities, n)
    rng = np.random.default_rng(seed + 1)
    parts: dict[str, list[np.ndarray]] = {v: [] for v in OUTPUT_UNITS}
    class_w = dict.fromkeys(classes, 0.0)
    blade_w = dict.fromkeys(blades, 0.0)
    for (c, b, _, weights, draws), p, k in zip(components, probabilities, counts, strict=True):
        class_w[c] += p
        blade_w[b] += p
        if k == 0:
            continue
        idx = rng.choice(n, size=k, replace=True, p=weights / weights.sum())
        for v in OUTPUT_UNITS:
            parts[v].append(draws[v][idx])

    if not hovering:
        notes.append(f"Flight state is {evidence.state.value}, not hover: the audio can't "
                     "inform the class guess (only facts you gave can), and mass comes from "
                     "the class prior.")
    return IdentifyReport(**common, status="ok" if hovering else "not_hovering",
                          classes=_ranked_classes(class_w), blades=_ranked_blades(blade_w),
                          variables=_summaries({v: np.concatenate(p) for v, p in parts.items()}),
                          notes=notes)

"""Simulator round-trip for the class guess (rule 2): drones drawn from each
class's own priors -> hover audio -> identify. A Bayes-consistent posterior beats
chance and its stated confidence matches its hit rate. Measured during design
(48 drones): accuracy 0.54 vs mean top probability 0.50; chance is 0.25."""

import numpy as np

from augur.config import load_classes, load_priors
from augur.fusion.identify import candidate_classes, identify
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform

PER_CLASS = 12
DETECTABLE_BPF_HZ = (60.0, 4500.0)


def _drones(rng):
    classes, priors = load_classes(), load_priors()
    ct = priors.c_t_fallback
    for name in candidate_classes(None):
        cfg = classes.get(name)
        made = 0
        while made < PER_CLASS:
            n_motors = int(rng.choice(cfg["num_motors"]))
            d_in = rng.uniform(cfg["prop_diameter_inch"]["low"], cfg["prop_diameter_inch"]["high"])
            mass = rng.uniform(cfg["all_up_mass_kg"]["low"], cfg["all_up_mass_kg"]["high"])
            c_t = rng.uniform(ct.low, ct.high)
            blades = int(rng.choice(priors.blade_count_options))
            d_m = d_in * 0.0254
            rpm = float(np.sqrt(mass * priors.gravity
                                / (n_motors * c_t * priors.air_density * d_m**4)) * 60.0)
            if not DETECTABLE_BPF_HZ[0] <= rpm / 60.0 * blades <= DETECTABLE_BPF_HZ[1]:
                continue
            spec = DroneSpec(num_motors=n_motors, prop_diameter_inch=d_in, prop_pitch_inch=0.5 * d_in,
                             blade_count=blades, c_t=c_t, c_p=0.05, pole_count=14, efficiency=0.8,
                             mass_kg=mass, battery_wh=50.0, drone_class=name)
            yield name, spec, rpm
            made += 1


def test_class_guess_beats_chance_and_is_calibrated():
    hits, top_p = [], []
    for i, (name, spec, rpm) in enumerate(_drones(np.random.default_rng(42))):
        sig, sr = synth_waveform(spec, FlightCondition(rpm=rpm, state="hover"), duration_s=2.0,
                                 snr_db=25.0, rng=np.random.default_rng(i))
        report = identify(sig, sr, n=2000)
        assert report.status == "ok"
        hits.append(report.classes[0].name == name)
        top_p.append(report.classes[0].probability)
    accuracy, confidence = float(np.mean(hits)), float(np.mean(top_p))
    assert accuracy >= 0.40, f"top-1 accuracy {accuracy:.2f} (chance 0.25)"
    assert abs(accuracy - confidence) <= 0.15, f"accuracy {accuracy:.2f} vs confidence {confidence:.2f}"

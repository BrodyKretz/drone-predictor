// Display helpers shared across components.

export const PRETTY = {
  rpm: "RPM",
  thrust_per_motor_n: "Thrust / motor",
  total_thrust_n: "Total thrust",
  mass_kg: "Mass",
  weight_n: "Weight",
  shaft_power_w: "Shaft power",
  electrical_power_w: "Electrical power",
  disk_loading_n_m2: "Disk loading",
  min_frame_diagonal_m: "Frame diagonal (min)",
  thrust_to_weight: "Thrust-to-weight",
  battery_wh: "Battery capacity",
  endurance_s: "Endurance",
  num_motors: "Motor count",
  prop_diameter_inch: "Prop diameter",
};

// Preferred display order (falls back to alphabetical for anything unlisted).
export const ORDER = Object.keys(PRETTY);

export const CONFIDENCE_COLORS = {
  high: "#22c55e",
  medium: "#eab308",
  low: "#f97316",
  unconstrained: "#ef4444",
};

export function prettyName(key) {
  return PRETTY[key] || key;
}

export function fmt(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const a = Math.abs(x);
  if (a !== 0 && (a >= 10000 || a < 0.01)) return x.toExponential(2);
  if (a >= 100) return x.toFixed(0);
  if (a >= 1) return x.toFixed(2);
  return x.toFixed(3);
}

export function relativeWidth(v) {
  const denom = Math.abs(v.median);
  if (denom < 1e-12) return v.interval_high - v.interval_low > 1e-12 ? Infinity : 0;
  return (v.interval_high - v.interval_low) / denom;
}

// Which input tightened each variable most, and by how much (absolute width).
export function bestAttribution(attribution) {
  const best = {};
  for (const a of attribution) {
    const gain = a.width_before > 0 ? Math.max(0, (a.width_before - a.width_after) / a.width_before) : 0;
    if (!best[a.variable] || gain > best[a.variable].gain) {
      best[a.variable] = { source: a.source, gain };
    }
  }
  return best;
}

export function sortedVariableKeys(variables) {
  const keys = Object.keys(variables);
  return keys.sort((a, b) => {
    const ia = ORDER.indexOf(a);
    const ib = ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
}

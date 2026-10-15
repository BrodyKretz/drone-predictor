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

export function trimZeros(text) {
  return text.includes(".") && !text.includes("e") ? text.replace(/\.?0+$/, "") : text;
}

export function fmt(x) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  const a = Math.abs(x);
  if (a >= 1e6) return trimZeros((x / 1e6).toPrecision(3)) + "M";
  if (a >= 1e4) return trimZeros((x / 1e3).toPrecision(3)) + "k";
  if (a >= 1000) return Math.round(x).toLocaleString("en-US");
  if (a >= 100) return x.toFixed(0);
  if (a >= 10) return x.toFixed(1);
  if (a >= 1) return x.toFixed(2);
  if (a === 0) return "0";
  return trimZeros(x.toPrecision(2));
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

export function fmtK(x) {
  return Math.abs(x) >= 1000 ? trimZeros((x / 1000).toPrecision(3)) + "k" : fmt(x);
}

export function fmtRange(lo, hi) {
  const mid = Math.abs((lo + hi) / 2) || 1;
  const full = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en-US") : fmt(v));
  return (hi - lo) / mid < 0.1 ? `${full(lo)}–${full(hi)}` : `${fmt(lo)}–${fmt(hi)}`;
}

const DISPLAY = { endurance_s: { factor: 1 / 60, unit: "min" } };

export function displayFactor(key) {
  return DISPLAY[key] ? DISPLAY[key].factor : 1;
}

export function displayUnit(key, unit) {
  return DISPLAY[key] ? DISPLAY[key].unit : (unit || "").replace("^2", "²");
}

export function confidenceWord(p) {
  if (p >= 0.8) return "Confident";
  if (p >= 0.6) return "Likely";
  if (p >= 0.4) return "Leaning";
  return "Unsure";
}

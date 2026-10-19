import { displayFactor, displayUnit, fmt, fmtK, fmtRange } from "./format.js";

const MIN_LIKELY_BLADE_PROBABILITY = 0.01;

export function likelyBlades(blades) {
  if (!Array.isArray(blades)) {
    throw new TypeError(`likelyBlades: blades must be an array, got ${blades}`);
  }
  for (const entry of blades) {
    if (entry === null || typeof entry !== "object") {
      throw new TypeError(`likelyBlades: each entry must be an object, got ${entry}`);
    }
    if (!Number.isInteger(entry.blades) || entry.blades < 1) {
      throw new RangeError(`likelyBlades: blade counts must be positive integers, got ${entry.blades}`);
    }
    if (!Number.isFinite(entry.probability) || entry.probability < 0 || entry.probability > 1) {
      throw new RangeError(`likelyBlades: probabilities must be in [0, 1], got ${entry.probability}`);
    }
  }
  return blades
    .filter((entry) => entry.probability >= MIN_LIKELY_BLADE_PROBABILITY)
    .sort((a, b) => a.blades - b.blades);
}

function requireFiniteNumber(value, name) {
  if (!Number.isFinite(value)) {
    throw new TypeError(`estimateDisplay: ${name} must be a finite number, got ${value}`);
  }
}

function checkInputs(varKey, summary, result, isGiven) {
  if (typeof varKey !== "string" || varKey === "") {
    throw new TypeError(`estimateDisplay: varKey must be a non-empty string, got ${varKey}`);
  }
  if (summary === null || typeof summary !== "object") {
    throw new TypeError(`estimateDisplay: the summary for ${varKey} must be an object, got ${summary}`);
  }
  for (const field of ["median", "interval_low", "interval_high"]) {
    requireFiniteNumber(summary[field], `${varKey}.${field}`);
  }
  if (summary.interval_low > summary.interval_high) {
    throw new RangeError(
      `estimateDisplay: ${varKey} interval is inverted (${summary.interval_low} > ${summary.interval_high})`,
    );
  }
  if (typeof summary.confidence !== "string") {
    throw new TypeError(`estimateDisplay: ${varKey}.confidence must be a string, got ${summary.confidence}`);
  }
  if (result === null || typeof result !== "object" || typeof result.status !== "string") {
    throw new TypeError("estimateDisplay: result must be an identify report with a status");
  }
  if (!Array.isArray(result.blades) || result.acoustics === null || typeof result.acoustics !== "object") {
    throw new TypeError("estimateDisplay: result must carry blades and acoustics");
  }
  const f0 = result.acoustics.fundamental_hz;
  if (f0 !== null && !(Number.isFinite(f0) && f0 > 0)) {
    throw new RangeError(`estimateDisplay: fundamental_hz must be null or positive, got ${f0}`);
  }
  for (const { blades } of result.blades) {
    if (!Number.isInteger(blades) || blades < 1) {
      throw new RangeError(`estimateDisplay: blade counts must be positive integers, got ${blades}`);
    }
  }
  if (typeof isGiven !== "boolean") {
    throw new TypeError(`estimateDisplay: isGiven must be a boolean, got ${isGiven}`);
  }
}

function candidatesDisplay(summary, f0, likely) {
  const candidates = likely.map(({ blades }) => ({ blades, rpm: (f0 * 60) / blades }));
  return {
    kind: "candidates",
    valueText: candidates.map((c) => fmtK(c.rpm)).join(" or "),
    unit: "rpm",
    footText: `90% ${fmtRange(summary.interval_low, summary.interval_high)}`,
    note: `${candidates.map((c) => c.blades).join("- or ")}-blade props, count not given`,
    chipText: summary.confidence,
    chipTone: summary.confidence,
    isWide: false,
    isFixed: summary.interval_low === summary.interval_high,
    tickValues: candidates.map((c) => c.rpm),
    factor: 1,
  };
}

function chipFor({ varKey, summary, isFixed, isGiven, isWide }) {
  if (isFixed && isGiven) return { chipText: "given", chipTone: "fixed" };
  if (isFixed) return { chipText: summary.confidence, chipTone: summary.confidence };
  if (varKey === "thrust_to_weight") return { chipText: "class prior", chipTone: "wide" };
  if (isWide) return { chipText: "rough", chipTone: "wide" };
  return { chipText: summary.confidence, chipTone: summary.confidence };
}

function regularDisplay(varKey, summary, hasTone, isGiven) {
  const factor = displayFactor(varKey);
  const isFixed = summary.interval_low === summary.interval_high;
  const isPrior = varKey === "thrust_to_weight" || (varKey === "rpm" && !hasTone);
  const isWide = summary.confidence === "unconstrained" || isPrior;
  const isInteger = varKey === "num_motors";
  const range = isInteger
    ? `${Math.round(summary.interval_low)}–${Math.round(summary.interval_high)}`
    : fmtRange(summary.interval_low * factor, summary.interval_high * factor);
  const median = isInteger ? String(Math.round(summary.median)) : fmt(summary.median * factor);
  let footText = `90% ${range}`;
  if (isFixed) footText = isGiven ? "from your input" : "set by the class guess";
  return {
    kind: "regular",
    valueText: !isFixed && (isWide || isInteger) ? range : median,
    unit: displayUnit(varKey, summary.unit),
    footText,
    note: null,
    ...chipFor({ varKey, summary, isFixed, isGiven, isWide }),
    isWide,
    isFixed,
    tickValues: isWide || isFixed ? [] : [summary.median],
    factor,
  };
}

export function estimateDisplay(varKey, summary, result, isGiven) {
  checkInputs(varKey, summary, result, isGiven);
  const f0 = result.acoustics.fundamental_hz;
  const hasTone = f0 !== null;
  const likely = likelyBlades(result.blades);
  // an unconstrained rpm is only the prior, so two crisp candidates would claim more than the backend found
  const showsCandidates =
    varKey === "rpm" &&
    hasTone &&
    likely.length > 1 &&
    result.status !== "no_tone" &&
    summary.confidence !== "unconstrained";
  return showsCandidates
    ? candidatesDisplay(summary, f0, likely)
    : regularDisplay(varKey, summary, hasTone, isGiven);
}

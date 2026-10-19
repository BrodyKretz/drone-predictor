import { test } from "node:test";
import assert from "node:assert/strict";
import { estimateDisplay } from "./estimates.js";

const summary = (overrides = {}) => ({
  median: 5,
  mean: 5,
  interval_low: 4,
  interval_high: 6,
  interval_level: 0.9,
  confidence: "medium",
  unit: null,
  ...overrides,
});

const report = ({ status = "ok", f0 = 393, blades = [2, 3] } = {}) => ({
  status,
  blades: blades.map((count) => ({ blades: count, probability: 1 / blades.length })),
  acoustics: { fundamental_hz: f0 },
});

const rpmSummary = (overrides = {}) =>
  summary({ median: 11686, interval_low: 7795, interval_high: 11901, unit: "rpm", ...overrides });

test("a constrained rpm with a tone and two blade counts shows candidates sorted by blade count", () => {
  const display = estimateDisplay("rpm", rpmSummary(), report({ blades: [3, 2] }), false);
  assert.equal(display.kind, "candidates");
  assert.equal(display.valueText, "11.8k or 7.86k");
  assert.equal(display.unit, "rpm");
  assert.deepEqual(display.tickValues, [(393 * 60) / 2, (393 * 60) / 3]);
  assert.equal(display.footText, "90% 7,795–11.9k");
  assert.equal(display.note, "2- or 3-blade props, count not given");
  assert.equal(display.chipText, "medium");
  assert.equal(display.chipTone, "medium");
  assert.equal(display.isWide, false);
});

test("an unconstrained rpm stays a wide range even when a tone was heard", () => {
  const rpm = rpmSummary({ median: 15291, interval_low: 3472, interval_high: 26826, confidence: "unconstrained" });
  const display = estimateDisplay("rpm", rpm, report({ status: "no_class_fits", f0: 177.65 }), false);
  assert.equal(display.kind, "regular");
  assert.equal(display.valueText, "3,472–26.8k");
  assert.equal(display.unit, "rpm");
  assert.equal(display.footText, "90% 3,472–26.8k");
  assert.equal(display.note, null);
  assert.equal(display.chipText, "rough");
  assert.equal(display.chipTone, "wide");
  assert.equal(display.isWide, true);
  assert.deepEqual(display.tickValues, []);
});

test("rpm without a tone is a wide prior row whatever the backend labels it", () => {
  const result = report({ status: "no_tone", f0: null });
  for (const confidence of ["unconstrained", "low"]) {
    const rpm = rpmSummary({ median: 15291, interval_low: 3472, interval_high: 26826, confidence });
    const display = estimateDisplay("rpm", rpm, result, false);
    assert.equal(display.kind, "regular");
    assert.equal(display.valueText, "3,472–26.8k");
    assert.equal(display.chipText, "rough");
    assert.equal(display.isWide, true);
    assert.deepEqual(display.tickValues, []);
  }
});

test("a single blade-count candidate falls through to the regular row with a median tick", () => {
  const display = estimateDisplay("rpm", rpmSummary(), report({ blades: [2] }), false);
  assert.equal(display.kind, "regular");
  assert.equal(display.valueText, "11.7k");
  assert.equal(display.footText, "90% 7,795–11.9k");
  assert.equal(display.chipText, "medium");
  assert.deepEqual(display.tickValues, [11686]);
});

test("a given fact is labelled as the user's input", () => {
  const motors = summary({ median: 4, mean: 4, interval_low: 4, interval_high: 4, confidence: "high" });
  const display = estimateDisplay("num_motors", motors, report(), true);
  assert.equal(display.valueText, "4");
  assert.equal(display.isFixed, true);
  assert.equal(display.footText, "from your input");
  assert.equal(display.chipText, "given");
  assert.equal(display.chipTone, "fixed");
  assert.deepEqual(display.tickValues, []);
});

test("an inferred single value keeps the backend's confidence chip", () => {
  const motors = summary({ median: 4, mean: 4, interval_low: 4, interval_high: 4, confidence: "high" });
  const display = estimateDisplay("num_motors", motors, report(), false);
  assert.equal(display.valueText, "4");
  assert.equal(display.isFixed, true);
  assert.equal(display.footText, "set by the class guess");
  assert.equal(display.chipText, "high");
  assert.equal(display.chipTone, "high");
  assert.deepEqual(display.tickValues, []);
});

test("thrust-to-weight is always a class prior", () => {
  const twr = summary({ median: 5.67, interval_low: 2.25, interval_high: 11.26, confidence: "low" });
  const display = estimateDisplay("thrust_to_weight", twr, report(), false);
  assert.equal(display.valueText, "2.25–11.3");
  assert.equal(display.unit, "");
  assert.equal(display.footText, "90% 2.25–11.3");
  assert.equal(display.chipText, "class prior");
  assert.equal(display.chipTone, "wide");
  assert.equal(display.isWide, true);
  assert.deepEqual(display.tickValues, []);
});

test("a motor-count range prints as whole numbers in the value slot", () => {
  const motors = summary({ median: 4, mean: 4.6, interval_low: 3, interval_high: 8, confidence: "low" });
  const display = estimateDisplay("num_motors", motors, report(), false);
  assert.equal(display.valueText, "3–8");
  assert.equal(display.footText, "90% 3–8");
  assert.equal(display.chipText, "low");
  assert.equal(display.chipTone, "low");
  assert.equal(display.isWide, false);
  assert.deepEqual(display.tickValues, [4]);
});

test("endurance is shown in minutes while its ticks stay in seconds", () => {
  const endurance = summary({ median: 806.8, interval_low: 389.6, interval_high: 1573.2, confidence: "low", unit: "s" });
  const display = estimateDisplay("endurance_s", endurance, report(), false);
  assert.equal(display.valueText, "13.4");
  assert.equal(display.unit, "min");
  assert.equal(display.footText, "90% 6.49–26.2");
  assert.deepEqual(display.tickValues, [806.8]);
  assert.ok(Math.abs(display.factor * 60 - 1) < 1e-9);
});

test("estimateDisplay rejects malformed inputs", () => {
  const rpm = rpmSummary();
  assert.throws(() => estimateDisplay("", rpm, report(), false), TypeError);
  assert.throws(() => estimateDisplay("rpm", undefined, report(), false), TypeError);
  assert.throws(() => estimateDisplay("rpm", rpmSummary({ median: NaN }), report(), false), TypeError);
  assert.throws(() => estimateDisplay("rpm", rpmSummary({ interval_low: 12000 }), report(), false), RangeError);
  assert.throws(() => estimateDisplay("rpm", rpmSummary({ confidence: undefined }), report(), false), TypeError);
  assert.throws(() => estimateDisplay("rpm", rpm, null, false), TypeError);
  assert.throws(() => estimateDisplay("rpm", rpm, { ...report(), blades: null }, false), TypeError);
  assert.throws(() => estimateDisplay("rpm", rpm, report({ f0: -3 }), false), RangeError);
  assert.throws(() => estimateDisplay("rpm", rpm, report({ blades: [0, 2] }), false), RangeError);
  assert.throws(() => estimateDisplay("rpm", rpm, report(), "yes"), TypeError);
});

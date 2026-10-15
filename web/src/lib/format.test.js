import { test } from "node:test";
import assert from "node:assert/strict";
import { confidenceWord, displayFactor, displayUnit, fmt, fmtK, fmtRange } from "./format.js";

test("fmt picks precision by magnitude", () => {
  assert.equal(fmt(0.6123), "0.61");
  assert.equal(fmt(7.994), "7.99");
  assert.equal(fmt(117.9), "118");
  assert.equal(fmt(2042.4), "2,042");
  assert.equal(fmt(46480), "46.5k");
  assert.equal(fmt(1.46e6), "1.46M");
  assert.equal(fmt(NaN), "—");
});

test("fmtK keeps rpm-style numbers compact", () => {
  assert.equal(fmtK(11787), "11.8k");
  assert.equal(fmtK(7852), "7.85k");
  assert.equal(fmtK(950), "950");
});

test("fmtRange shows full digits for narrow intervals", () => {
  assert.equal(fmtRange(11654, 11920), "11,654–11,920");
  assert.equal(fmtRange(0.29, 0.83), "0.29–0.83");
});

const printedBounds = (text) => text.split("–").map((part) => Number(part.replaceAll(",", "")));

test("fmtRange keeps narrow sub-1000 intervals visible and containing", () => {
  const intervals = [
    [0.1051, 0.1149],
    [0.246, 0.254],
    [117.9, 118.1],
    [49.96, 50.04],
    [0.6101, 0.6149],
  ];
  for (const [lo, hi] of intervals) {
    const text = fmtRange(lo, hi);
    const [printedLo, printedHi] = printedBounds(text);
    assert.ok(printedLo <= lo, `${text} starts above ${lo}`);
    assert.ok(printedHi >= hi, `${text} ends below ${hi}`);
    assert.ok(printedLo < printedHi, `${text} has identical endpoints`);
  }
});

test("fmtRange prints a zero-width interval as one value", () => {
  assert.equal(fmtRange(250, 250), "250");
});

test("fmtRange uses the fewest digits that separate the endpoints and rounds outward", () => {
  assert.equal(fmtRange(0.1051, 0.1149), "0.105–0.115");
  assert.equal(fmtRange(0.246, 0.254), "0.246–0.254");
  assert.equal(fmtRange(117.9, 118.1), "117.9–118.1");
  assert.equal(fmtRange(49.96, 50.04), "49.96–50.04");
  assert.equal(fmtRange(0.6101, 0.6149), "0.610–0.615");
  assert.equal(fmtRange(0.5251, 0.5549), "0.52–0.56");
  assert.equal(fmtRange(0.29, 0.31), "0.29–0.31");
  assert.equal(fmtRange(11654.6, 11919.4), "11,654–11,920");
  assert.equal(fmtRange(117.9999999, 118.0000001), "117.999–118.001");
});

test("fmtRange never uses scientific notation for narrow intervals", () => {
  assert.equal(fmtRange(1.01e-7, 1.04e-7), "0.000000101–0.000000104");
  assert.match(fmtRange(1e-300, 1.01e-300), /^[0-9.]+–[0-9.]+$/);
});

test("fmtRange leaves non-finite endpoints as dashes", () => {
  assert.equal(fmtRange(NaN, 5), "—–5.00");
  assert.equal(fmtRange(1, Infinity), "1.00–—");
  assert.equal(fmtRange(null, null), "—–—");
});

test("endurance displays in minutes and units are prettified", () => {
  assert.ok(Math.abs(displayFactor("endurance_s") * 600 - 10) < 1e-9);
  assert.equal(displayUnit("endurance_s", "s"), "min");
  assert.equal(displayUnit("disk_loading_n_m2", "N/m^2"), "N/m²");
  assert.equal(displayUnit("thrust_to_weight", null), "");
});

test("confidence words follow the probability bands", () => {
  assert.equal(confidenceWord(0.85), "Confident");
  assert.equal(confidenceWord(0.65), "Likely");
  assert.equal(confidenceWord(0.54), "Leaning");
  assert.equal(confidenceWord(0.3), "Unsure");
});

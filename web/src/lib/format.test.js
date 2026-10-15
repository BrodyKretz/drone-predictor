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

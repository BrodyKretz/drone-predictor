import { test } from "node:test";
import assert from "node:assert/strict";
import { downmix, encodeWav, latestWindow } from "./wav.js";

test("encodeWav writes a 16-bit mono PCM header", () => {
  const buf = encodeWav(new Float32Array([0, 0.5, -1]), 48000);
  const view = new DataView(buf);
  const text = (offset, n) => String.fromCharCode(...new Uint8Array(buf, offset, n));
  assert.equal(buf.byteLength, 44 + 3 * 2);
  assert.equal(text(0, 4), "RIFF");
  assert.equal(text(8, 4), "WAVE");
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 48000);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 6);
});

test("encodeWav clips out-of-range samples", () => {
  const view = new DataView(encodeWav(new Float32Array([2, -2, 0.5]), 8000));
  assert.equal(view.getInt16(44, true), 32767);
  assert.equal(view.getInt16(46, true), -32767);
  assert.equal(view.getInt16(48, true), 16384);
});

test("downmix averages channels", () => {
  const mixed = downmix([new Float32Array([1, 0]), new Float32Array([0, 0])]);
  assert.deepEqual(Array.from(mixed), [0.5, 0]);
});

test("latestWindow returns the last N seconds before the end index", () => {
  const samples = Float32Array.from({ length: 10 }, (_, i) => i);
  assert.deepEqual(Array.from(latestWindow(samples, 2, 8, 1.5)), [5, 6, 7]);
  assert.deepEqual(Array.from(latestWindow(samples, 2, 2, 3)), [0, 1]);
});

test("encodeWav rejects invalid samples and sample rates", () => {
  const samples = new Float32Array([0, 0.5]);
  const invalidSamples = [
    null,
    undefined,
    "abc",
    5,
    { length: 2 },
    new Float64Array(2),
    new Array(2),
    [0, NaN],
    [0, Infinity],
    [0, "1"],
  ];
  const invalidSampleRates = [0, -8000, 44100.5, NaN, Infinity, "48000", null, undefined];
  for (const bad of invalidSamples) {
    assert.throws(() => encodeWav(bad, 8000), {
      name: "TypeError",
      message: /samples must be/,
    });
  }
  for (const bad of invalidSampleRates) {
    assert.throws(() => encodeWav(samples, bad), {
      name: "RangeError",
      message: /sampleRate must be/,
    });
  }
  assert.equal(encodeWav([0, 0.5], 1).byteLength, 44 + 2 * 2);
});

test("downmix rejects missing or mismatched channels", () => {
  const invalidChannels = [null, undefined, [], new Float32Array(2), "ab", { length: 1 }];
  for (const bad of invalidChannels) {
    assert.throws(() => downmix(bad), { name: "TypeError", message: /channels must be/ });
  }
  assert.throws(() => downmix([new Float32Array(3), new Float32Array(2)]), {
    name: "RangeError",
    message: /channel 1 has 2 samples, expected 3/,
  });
});

test("latestWindow rejects invalid rates, durations and end indices", () => {
  const samples = new Float32Array(10);
  const invalidPositiveNumbers = [0, -1, NaN, Infinity, "2", null, undefined];
  const invalidEndIndices = [-1, 1.5, NaN, Infinity, "8", null, undefined];
  for (const bad of invalidPositiveNumbers) {
    assert.throws(() => latestWindow(samples, bad, 8, 1.5), {
      name: "RangeError",
      message: /sampleRate must be/,
    });
    assert.throws(() => latestWindow(samples, 2, 8, bad), {
      name: "RangeError",
      message: /seconds must be/,
    });
  }
  for (const bad of invalidEndIndices) {
    assert.throws(() => latestWindow(samples, 2, bad, 1.5), {
      name: "RangeError",
      message: /endIndex must be/,
    });
  }
  assert.equal(latestWindow(samples, 2, 0, 1.5).length, 0);
});

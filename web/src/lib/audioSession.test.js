import { test } from "node:test";
import assert from "node:assert/strict";
import { MIN_WINDOW_SAMPLES, WINDOW_SECONDS, windowWav } from "./audioSession.js";

const header = (wav) => {
  const view = new DataView(wav);
  return { sampleRate: view.getUint32(24, true), samples: view.getUint32(40, true) / 2 };
};

test("windowWav sends nothing until the backend's minimum window has played", () => {
  const samples = new Float32Array(10 * 44100);
  assert.equal(windowWav(samples, 44100, 0), null);
  assert.equal(windowWav(samples, 44100, MIN_WINDOW_SAMPLES - 1), null);
  const wav = windowWav(samples, 44100, MIN_WINDOW_SAMPLES);
  assert.deepEqual(header(wav), { sampleRate: 44100, samples: MIN_WINDOW_SAMPLES });
});

test("windowWav sends the latest window once enough audio has played", () => {
  const samples = new Float32Array(10 * 48000);
  const wav = windowWav(samples, 48000, samples.length);
  assert.deepEqual(header(wav), { sampleRate: 48000, samples: WINDOW_SECONDS * 48000 });
});

test("windowWav stretches the window to the minimum at low sample rates", () => {
  const samples = new Float32Array(10 * 16000);
  assert.equal(windowWav(samples, 16000, MIN_WINDOW_SAMPLES - 1), null);
  const wav = windowWav(samples, 16000, samples.length);
  assert.deepEqual(header(wav), { sampleRate: 16000, samples: MIN_WINDOW_SAMPLES });
});

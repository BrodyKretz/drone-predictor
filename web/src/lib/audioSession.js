import { downmix, encodeWav, latestWindow } from "./wav.js";

export const WINDOW_SECONDS = 3;
// Mirrors MIN_WINDOW_SAMPLES in src/augur/fusion/identify.py, which rejects shorter windows.
export const MIN_WINDOW_SAMPLES = 88200;
export const TICK_MS = 600;
const PLAYBACK_GAIN = 0.25;
const MIC_BUFFER_SECONDS = 8;
const RESUME_TIMEOUT_MS = 2000;

let context = null;

function audioContext() {
  if (!context) context = new AudioContext();
  return context;
}

export function resumeAudio() {
  const ctx = audioContext();
  // resume() never settles while the autoplay policy blocks the context, so the wait is bounded.
  const timeout = new Promise((resolve) => setTimeout(resolve, RESUME_TIMEOUT_MS));
  return Promise.race([ctx.resume(), timeout]).then(() => ctx.state === "running");
}

export async function decodeFile(file) {
  return audioContext().decodeAudioData(await file.arrayBuffer());
}

export async function decodeUrl(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return audioContext().decodeAudioData(await res.arrayBuffer());
}

export function windowWav(samples, sampleRate, endIndex) {
  const length = Math.max(Math.round(WINDOW_SECONDS * sampleRate), MIN_WINDOW_SAMPLES);
  const win = latestWindow(samples, sampleRate, endIndex, length / sampleRate);
  return win.length >= MIN_WINDOW_SAMPLES ? encodeWav(win, sampleRate) : null;
}

export function playBuffer(buffer, { onWindow, onEnd }) {
  const ctx = audioContext();
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
  const mono = downmix(channels);
  const rate = buffer.sampleRate;
  const source = ctx.createBufferSource();
  const gain = ctx.createGain();
  gain.gain.value = PLAYBACK_GAIN;
  source.buffer = buffer;
  source.connect(gain).connect(ctx.destination);
  const startedAt = ctx.currentTime;
  let finished = false;
  const elapsed = () => Math.min(ctx.currentTime - startedAt, buffer.duration);
  const timer = setInterval(() => {
    const wav = windowWav(mono, rate, Math.floor(elapsed() * rate));
    if (wav) onWindow(wav);
  }, TICK_MS);
  const finish = (endIndex) => {
    if (finished) return;
    finished = true;
    clearInterval(timer);
    onEnd(windowWav(mono, rate, endIndex), rate);
  };
  source.onended = () => finish(mono.length);
  source.start();
  return {
    elapsed,
    stop() {
      source.onended = null;
      try {
        source.stop();
      } catch {
        /* already stopped */
      }
      finish(Math.floor(elapsed() * rate));
    },
  };
}

export async function startMic({ onWindow, onEnd }) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  try {
    return micSession(stream, { onWindow, onEnd });
  } catch (err) {
    stream.getTracks().forEach((track) => track.stop());
    throw err;
  }
}

function micSession(stream, { onWindow, onEnd }) {
  const ctx = audioContext();
  const rate = Math.round(ctx.sampleRate);
  const keep = Math.max(MIC_BUFFER_SECONDS * rate, MIN_WINDOW_SAMPLES);
  const input = ctx.createMediaStreamSource(stream);
  // ScriptProcessorNode is deprecated but works everywhere and needs no separate worklet module.
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  let ring = new Float32Array(0);
  processor.onaudioprocess = (event) => {
    const chunk = event.inputBuffer.getChannelData(0);
    const next = new Float32Array(Math.min(ring.length + chunk.length, keep));
    const kept = ring.subarray(Math.max(0, ring.length + chunk.length - keep));
    next.set(kept, 0);
    next.set(chunk.subarray(chunk.length - (next.length - kept.length)), kept.length);
    ring = next;
  };
  input.connect(processor);
  processor.connect(ctx.destination);
  const startedAt = performance.now();
  let stoppedAt = null;
  const timer = setInterval(() => {
    const wav = windowWav(ring, rate, ring.length);
    if (wav) onWindow(wav);
  }, TICK_MS);
  return {
    elapsed: () => ((stoppedAt ?? performance.now()) - startedAt) / 1000,
    stop() {
      if (stoppedAt !== null) return;
      stoppedAt = performance.now();
      clearInterval(timer);
      processor.disconnect();
      input.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      onEnd(windowWav(ring, rate, ring.length), rate);
    },
  };
}

function isFiniteNumberArray(values) {
  if (!Array.isArray(values)) return false;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) return false;
  }
  return true;
}

export function encodeWav(samples, sampleRate) {
  if (!(samples instanceof Float32Array) && !isFiniteNumberArray(samples)) {
    throw new TypeError("encodeWav: samples must be a Float32Array or an array of finite numbers");
  }
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`encodeWav: sampleRate must be a positive integer, got ${sampleRate}`);
  }
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), true);
  }
  return buffer;
}

export function downmix(channels) {
  if (!Array.isArray(channels) || channels.length === 0) {
    throw new TypeError("downmix: channels must be a non-empty array");
  }
  const length = channels[0].length;
  const mismatched = channels.findIndex((channel) => channel.length !== length);
  if (mismatched !== -1) {
    const got = channels[mismatched].length;
    throw new RangeError(`downmix: channel ${mismatched} has ${got} samples, expected ${length}`);
  }
  const out = new Float32Array(length);
  for (const channel of channels) {
    for (let i = 0; i < out.length; i++) out[i] += channel[i] / channels.length;
  }
  return out;
}

export function latestWindow(samples, sampleRate, endIndex, seconds) {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`latestWindow: sampleRate must be positive and finite, got ${sampleRate}`);
  }
  if (!Number.isFinite(seconds) || seconds <= 0) {
    throw new RangeError(`latestWindow: seconds must be positive and finite, got ${seconds}`);
  }
  if (!Number.isInteger(endIndex) || endIndex < 0) {
    throw new RangeError(`latestWindow: endIndex must be a non-negative integer, got ${endIndex}`);
  }
  const end = Math.min(endIndex, samples.length);
  return samples.subarray(Math.max(0, end - Math.round(seconds * sampleRate)), end);
}

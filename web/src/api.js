const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 10000;

async function errorDetail(res) {
  try {
    const body = await res.json();
    if (body.detail) return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* non-JSON error body */
  }
  return `${res.status} ${res.statusText}`;
}

export async function identify(wavBuffer, spec) {
  const form = new FormData();
  form.append("audio", new Blob([wavBuffer], { type: "audio/wav" }), "window.wav");
  if (spec) form.append("spec", JSON.stringify(spec));
  const res = await fetch(`${API_URL}/identify`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export async function reference() {
  const res = await fetch(`${API_URL}/reference`, { signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export async function health() {
  const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
  if (!res.ok) throw new Error("backend unreachable");
  return res.json();
}

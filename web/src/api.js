// Talks to the Augur FastAPI backend. Override the base URL with VITE_API_URL.
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function predict({ audio, verbalJson, samples, seed }) {
  const form = new FormData();
  if (audio) form.append("audio", audio);
  if (verbalJson) form.append("verbal_json", verbalJson);
  if (samples) form.append("samples", String(samples));
  if (seed !== undefined) form.append("seed", String(seed));

  const res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function health() {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("backend unreachable");
  return res.json();
}

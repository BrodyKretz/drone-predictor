import { useEffect, useState } from "react";
import { health, predict } from "./api.js";
import Report from "./components/Report.jsx";

export default function App() {
  const [audio, setAudio] = useState(null);
  const [verbal, setVerbal] = useState(null);
  const [samples, setSamples] = useState(8000);
  const [seed, setSeed] = useState(0);

  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendUp, setBackendUp] = useState(null);

  useEffect(() => {
    health()
      .then(() => setBackendUp(true))
      .catch(() => setBackendUp(false));
  }, []);

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    if (!audio && !verbal) {
      setError("Provide at least an audio clip or a verbal spec.");
      return;
    }
    setLoading(true);
    try {
      const result = await predict({ audio, verbal, samples, seed });
      setReport(result);
    } catch (err) {
      setError(err.message);
      setReport(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Augur</h1>
        <p className="tagline">
          Multimodal drone property inference — calibrated distributions, never point estimates.
        </p>
        <span className={`status ${backendUp === false ? "down" : backendUp ? "up" : ""}`}>
          {backendUp === null && "checking backend…"}
          {backendUp === true && "backend connected"}
          {backendUp === false && "backend offline — run: augur serve"}
        </span>
      </header>

      <div className="layout">
        <form className="panel controls" onSubmit={onSubmit}>
          <h2>Inputs</h2>

          <label>
            Audio (.wav)
            <input type="file" accept=".wav,audio/*" onChange={(e) => setAudio(e.target.files[0] || null)} />
            {audio && <span className="file-name">{audio.name}</span>}
          </label>

          <label>
            Verbal spec (.json)
            <input type="file" accept=".json,application/json" onChange={(e) => setVerbal(e.target.files[0] || null)} />
            {verbal && <span className="file-name">{verbal.name}</span>}
          </label>

          <div className="row">
            <label>
              MC samples
              <input type="number" min="500" step="500" value={samples} onChange={(e) => setSamples(+e.target.value)} />
            </label>
            <label>
              Seed
              <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
            </label>
          </div>

          <button type="submit" disabled={loading || backendUp === false}>
            {loading ? "Predicting…" : "Predict"}
          </button>

          {error && <p className="error">{error}</p>}

          <p className="hint">
            Demo files live in <code>data/demo/</code> — generate them with
            <br />
            <code>python scripts/make_demo_sample.py</code>
          </p>
        </form>

        <div className="panel results">
          <h2>Prediction</h2>
          {!report && !loading && <p className="empty">Upload inputs and hit Predict.</p>}
          {report && <Report report={report} />}
        </div>
      </div>
    </div>
  );
}

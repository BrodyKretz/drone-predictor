import { useEffect, useMemo, useState } from "react";
import { health, predict } from "./api.js";
import Report from "./components/Report.jsx";

// The verbal spec as a set of discrete dropdowns. Every field is optional — "Any"
// contributes no constraint, matching the backend VerbalSpec (all fields nullable).
const SPEC_FIELDS = [
  {
    key: "drone_class",
    label: "Drone class",
    type: "string",
    options: [["Any", ""], ["Racing", "racing"], ["Cinematic", "cinematic"], ["Survey", "survey"]],
  },
  {
    key: "num_motors",
    label: "Motors",
    type: "int",
    options: [["Any", ""], ["4", "4"], ["6", "6"], ["8", "8"]],
  },
  {
    key: "blade_count",
    label: "Blades / prop",
    type: "int",
    options: [["Any", ""], ["2", "2"], ["3", "3"]],
  },
  {
    key: "prop_diameter_inch",
    label: "Prop diameter",
    type: "float",
    options: [["Any", ""], ['3"', "3"], ['5"', "5"], ['7"', "7"], ['10"', "10"], ['13"', "13"], ['15"', "15"], ['18"', "18"], ['22"', "22"]],
  },
  {
    key: "cell_count",
    label: "Battery cells",
    type: "int",
    options: [["Any", ""], ["3S", "3"], ["4S", "4"], ["6S", "6"]],
  },
];

function buildSpec(selections) {
  const spec = {};
  for (const field of SPEC_FIELDS) {
    const raw = selections[field.key];
    if (raw === undefined || raw === "") continue;
    spec[field.key] = field.type === "string" ? raw : Number(raw);
  }
  return spec;
}

export default function App() {
  const [audio, setAudio] = useState(null);
  const [selections, setSelections] = useState({});
  const [samples, setSamples] = useState(8000);
  const [seed, setSeed] = useState(0);

  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendUp, setBackendUp] = useState(null);

  useEffect(() => {
    health().then(() => setBackendUp(true)).catch(() => setBackendUp(false));
  }, []);

  const spec = useMemo(() => buildSpec(selections), [selections]);
  const hasSpec = Object.keys(spec).length > 0;

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    if (!audio && !hasSpec) {
      setError("Pick at least an audio clip or one spec field.");
      return;
    }
    setLoading(true);
    try {
      const result = await predict({
        audio,
        verbalJson: hasSpec ? JSON.stringify(spec) : null,
        samples,
        seed,
      });
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

          <fieldset className="spec">
            <legend>Verbal spec</legend>
            {SPEC_FIELDS.map((field) => (
              <label key={field.key} className="spec-field">
                {field.label}
                <select
                  value={selections[field.key] ?? ""}
                  onChange={(e) => setSelections((s) => ({ ...s, [field.key]: e.target.value }))}
                >
                  {field.options.map(([text, value]) => (
                    <option key={value} value={value}>
                      {text}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </fieldset>

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
            Demo audio: generate with <code>python scripts/make_demo_sample.py</code> →
            {" "}<code>data/demo/drone.wav</code>. Leave the spec on "Any" to see the
            audio-only estimate widen.
          </p>
        </form>

        <div className="panel results">
          <h2>Prediction</h2>
          {!report && !loading && <p className="empty">Choose inputs and hit Predict.</p>}
          {report && <Report report={report} />}
        </div>
      </div>
    </div>
  );
}

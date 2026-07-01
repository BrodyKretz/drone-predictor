import VariableBar from "./VariableBar.jsx";
import { bestAttribution, sortedVariableKeys } from "../lib/format.js";

export default function Report({ report }) {
  const attr = bestAttribution(report.attribution || []);
  const keys = sortedVariableKeys(report.variables);

  return (
    <div className="report">
      <div className="report-meta">
        <span>
          inputs:{" "}
          {(report.inputs_used || []).map((i) => (
            <span key={i} className="chip">
              {i}
            </span>
          ))}
          {(!report.inputs_used || report.inputs_used.length === 0) && "none"}
        </span>
        <span className="config">config v{report.config_version}</span>
      </div>

      <div className="var-list">
        {keys.map((k) => (
          <VariableBar key={k} varKey={k} summary={report.variables[k]} attribution={attr[k]} />
        ))}
      </div>

      {report.assumptions?.length > 0 && (
        <section className="notes-block">
          <h3>Assumptions</h3>
          <ul>
            {report.assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </section>
      )}

      {report.notes?.length > 0 && (
        <section className="notes-block warn">
          <h3>Honesty notes</h3>
          <ul>
            {report.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

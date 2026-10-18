import { displayFactor, displayUnit, fmt, fmtK, fmtRange } from "../lib/format.js";
import { PanelTab } from "./chrome.jsx";
import Sparkline from "./Sparkline.jsx";

const ESTIMATES = [
  ["mass_kg", "Mass"],
  ["rpm", "Rotor speed"],
  ["electrical_power_w", "Hover power"],
  ["endurance_s", "Hover endurance"],
  ["battery_wh", "Battery"],
  ["total_thrust_n", "Total thrust"],
  ["disk_loading_n_m2", "Disk loading"],
  ["prop_diameter_inch", "Prop size"],
  ["num_motors", "Motors"],
  ["thrust_to_weight", "Thrust-to-weight"],
];

function logPosition(band) {
  const lo = Math.log10(Math.max(band[0], 1e-9));
  const hi = Math.log10(band[1]);
  return (x) => Math.min(1, Math.max(0, (Math.log10(Math.max(x, 1e-12)) - lo) / (hi - lo))) * 100;
}

function Bar({ left, width, wide, ticks }) {
  return (
    <div className="est-bar">
      <span className={`fill ${wide ? "wide" : ""}`} style={{ left: `${left}%`, width: `${width}%` }} />
      {ticks.map((t, i) => <span key={i} className="tick" style={{ left: `calc(${t}% - 1px)` }} />)}
    </div>
  );
}

function RpmRow({ label, summary, f0, blades, pos, history }) {
  const alts = [...blades].sort((a, b) => a.blades - b.blades).map((b) => ({ blades: b.blades, rpm: (f0 * 60) / b.blades }));
  const left = pos(summary.interval_low);
  return (
    <div className="est">
      <div className="est-top">
        <span className="est-name">{label}</span>
        <span className="est-val">{alts.map((a) => fmtK(a.rpm)).join(" or ")}<span className="u">rpm</span></span>
      </div>
      <Bar left={left} width={Math.max(pos(summary.interval_high) - left, 1.2)} wide={false} ticks={alts.map((a) => pos(a.rpm))} />
      <div className="est-foot">
        <span className="left">{alts.map((a) => `${a.blades}-blade`).join(" or ")} props · count not given</span>
        <Sparkline points={history} factor={1} />
      </div>
    </div>
  );
}

function EstimateRow({ varKey, label, summary, band, history, result, given }) {
  const pos = logPosition(band);
  const f0 = result.acoustics.fundamental_hz;
  if (varKey === "rpm" && f0 && result.blades.length > 1 && result.status !== "no_tone") {
    return <RpmRow label={label} summary={summary} f0={f0} blades={result.blades} pos={pos} history={history} />;
  }
  const f = displayFactor(varKey);
  const unit = displayUnit(varKey, summary.unit);
  const fixed = summary.interval_low === summary.interval_high;
  const prior = varKey === "thrust_to_weight" || (varKey === "rpm" && !f0);
  const wide = summary.confidence === "unconstrained" || prior;
  const isInt = varKey === "num_motors";
  const range = isInt
    ? `${Math.round(summary.interval_low)}–${Math.round(summary.interval_high)}`
    : fmtRange(summary.interval_low * f, summary.interval_high * f);
  const value = fixed ? (isInt ? Math.round(summary.median) : fmt(summary.median * f)) : wide || isInt ? range : fmt(summary.median * f);
  const left = pos(summary.interval_low);
  const chipText = fixed ? (given ? "given" : "fixed") : varKey === "thrust_to_weight" ? "class prior" : wide ? "rough" : summary.confidence;
  const chipTone = fixed ? "fixed" : wide ? "wide" : summary.confidence;
  return (
    <div className="est">
      <div className="est-top">
        <span className="est-name">{label}</span>
        <span className={`est-val ${wide && !fixed ? "wide" : ""}`}>{value}{unit && <span className="u">{unit}</span>}</span>
      </div>
      <Bar left={left} width={Math.max(pos(summary.interval_high) - left, 1.2)} wide={wide} ticks={wide || fixed ? [] : [pos(summary.median)]} />
      <div className="est-foot">
        <span className="left">
          {fixed ? (given ? "from your input" : "set by the class guess") : `90% ${range}`} <span className={`chip chip--${chipTone}`}><i />{chipText}</span>
        </span>
        <Sparkline points={history} factor={f} />
      </div>
    </div>
  );
}

export default function EstimatesPanel({ result, history, bands, known }) {
  return (
    <section className="panel">
      <PanelTab icon="list" title="Live estimates" />
      <div className="body--scroll">
        {!result || !bands ? (
          <div className="placeholder">Estimates appear here and update as audio comes in.</div>
        ) : (
          <>
            {ESTIMATES.map(([key, label]) => (
              <EstimateRow key={key} varKey={key} label={label} summary={result.variables[key]} band={bands[key]}
                history={history[key] || []} result={result} given={Boolean(known[key])} />
            ))}
            <div className="side-note">
              <div>Every value is a 90% interval: the bar spans it, the white tick is the median. Grey means the inputs barely constrain it.</div>
              <div>Bars use a log scale covering what Augur would say with no input at all.</div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

import { estimateDisplay, estimateRows } from "../lib/estimates.js";
import { PanelTab } from "./chrome.jsx";
import Sparkline from "./Sparkline.jsx";

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

function EstimateRow({ varKey, label, summary, band, history, result, isGiven }) {
  const pos = logPosition(band);
  const display = estimateDisplay(varKey, summary, result, isGiven);
  const left = pos(summary.interval_low);
  const chip = <span className={`chip chip--${display.chipTone}`}><i />{display.chipText}</span>;
  return (
    <div className="est">
      <div className="est-top">
        <span className="est-name">{label}</span>
        <span className={`est-val ${display.isWide && !display.isFixed ? "wide" : ""}`}>
          {display.valueText}{display.unit && <span className="u">{display.unit}</span>}
        </span>
      </div>
      <Bar left={left} width={Math.max(pos(summary.interval_high) - left, 1.2)} wide={display.isWide}
        ticks={display.tickValues.map(pos)} />
      <div className="est-foot">
        {display.note ? (
          <span className="left stack">
            <span className="foot-line">{display.footText} {chip}</span>
            <span className="foot-note">{display.note}</span>
          </span>
        ) : (
          <span className="left">{display.footText} {chip}</span>
        )}
        <Sparkline points={history} factor={display.factor} />
      </div>
    </div>
  );
}

function Placeholder({ failed }) {
  return failed ? (
    <div className="placeholder placeholder--error">
      Couldn't load the display scale from the analysis server. Restart augur serve and reload.
    </div>
  ) : (
    <div className="placeholder">Estimates appear here and update as audio comes in.</div>
  );
}

export default function EstimatesPanel({ result, history, bands, facts, referenceFailed }) {
  return (
    <section className="panel">
      <PanelTab icon="list" title="Live estimates" />
      <div className="body--scroll">
        {referenceFailed || !result || !bands ? (
          <Placeholder failed={referenceFailed} />
        ) : (
          <>
            {estimateRows(result.status).map(({ varKey, label }) => (
              <EstimateRow key={varKey} varKey={varKey} label={label} summary={result.variables[varKey]}
                band={bands[varKey]} history={history[varKey] || []} result={result}
                isGiven={facts[varKey] !== undefined} />
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

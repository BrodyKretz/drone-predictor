const PATHS = {
  drone: (
    <>
      <circle cx="3.5" cy="3.5" r="2.2" />
      <circle cx="12.5" cy="3.5" r="2.2" />
      <circle cx="3.5" cy="12.5" r="2.2" />
      <circle cx="12.5" cy="12.5" r="2.2" />
      <path d="M5.2 5.2l5.6 5.6M10.8 5.2l-5.6 5.6" />
    </>
  ),
  wave: <path d="M1 8h2l1.5-4 2 8 2-10 2 10 1.5-4H15" />,
  upload: <path d="M8 10.5V2M4.5 5.5L8 2l3.5 3.5M2 10.5v3h12v-3" />,
  stop: <rect x="3.5" y="3.5" width="9" height="9" rx="1" fill="currentColor" stroke="none" />,
  target: (
    <>
      <circle cx="8" cy="8" r="6" />
      <circle cx="8" cy="8" r="2.5" />
    </>
  ),
  list: <path d="M2 4h12M2 8h12M2 12h8" />,
};

export function Icon({ name, className = "" }) {
  return (
    <svg className={`icon ${className}`} viewBox="0 0 16 16" fill="none" stroke="currentColor"
      strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {PATHS[name]}
    </svg>
  );
}

export function PanelTab({ icon, title }) {
  return (
    <div className="strip">
      <span><Icon name={icon} />{title}</span>
    </div>
  );
}

export function TopBar({ backendUp }) {
  const status = backendUp === null ? "connecting…"
    : backendUp ? "analysis server connected" : "analysis server offline: run augur serve";
  return (
    <header className="topbar">
      <div className="brand"><Icon name="drone" /><span className="word">AUGUR</span></div>
      <div className="vdiv" />
      <span className="tagline">Identify a drone from its sound, live</span>
      <div className="spacer" />
      <div className="status">
        <span className={`dot ${backendUp ? "ok" : backendUp === false ? "bad" : ""}`} />
        {status}
      </div>
    </header>
  );
}

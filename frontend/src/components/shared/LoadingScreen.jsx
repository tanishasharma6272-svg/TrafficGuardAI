export function LoadingScreen() {
  return (
    <main className="app-shell app-shell--center" role="status" aria-live="polite">
      <div className="boot-panel">
        <div className="boot-radar" aria-hidden="true">
          <span className="boot-pulse" />
          <span className="boot-pulse boot-pulse--delay" />
          <span className="boot-core">TG</span>
        </div>
        <div className="boot-info">
          <h2>Booting TrafficGuard AI</h2>
          <p>Connecting to PostgreSQL demo data store &amp; initializing Nagpur network…</p>
        </div>
      </div>
    </main>
  )
}

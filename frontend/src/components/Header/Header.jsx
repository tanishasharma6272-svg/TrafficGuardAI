import { Badge } from '../shared/Badge'

export function Header({
  clock,
  lastUpdated,
  dataMode,
  modelType,
  trainingDataMode,
  onRefresh,
  loading,
}) {
  return (
    <header className="command-header">
      <div className="command-header-left">
        <div className="command-brand-mark">
          <span>TG</span>
        </div>
        <div className="command-title-block">
          <div className="command-title-row">
            <h1>TrafficGuard AI</h1>
            <Badge variant="demo">{dataMode || 'DEMO DATA'}</Badge>
            <span className="model-header-tag mono">
              ML Model: {modelType || 'BaselineRidge'}
            </span>
            <Badge variant="operational" dot pulse>
              SYSTEM OPERATIONAL
            </Badge>
          </div>
          <p className="command-subtitle">
            Nagpur Traffic Risk &amp; Police Deployment Command · ML model trained on synthetic development data
          </p>
        </div>
      </div>

      <div className="command-header-right">
        <div className="telemetry-pill telemetry-pill--subtle">
          <span className="telemetry-label">TRAINING DATA</span>
          <span className="telemetry-val mono">{trainingDataMode || 'SYNTHETIC_DEVELOPMENT'}</span>
        </div>

        <div className="telemetry-pill">
          <span className="telemetry-label">SYSTEM CLOCK</span>
          <span className="telemetry-val mono">{clock.toLocaleTimeString()}</span>
        </div>

        {lastUpdated && (
          <div className="telemetry-pill telemetry-pill--subtle">
            <span className="telemetry-label">LAST SYNC</span>
            <span className="telemetry-val mono">{lastUpdated.toLocaleTimeString()}</span>
          </div>
        )}

        <button
          type="button"
          className="refresh-button"
          onClick={onRefresh}
          disabled={loading}
          title="Reload ML locations and risk scores from backend"
          aria-label="Refresh dashboard data"
        >
          <svg
            className={`refresh-icon ${loading ? 'refresh-icon--spin' : ''}`}
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M23 4v6h-6" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
          <span>{loading ? 'Syncing…' : 'Refresh'}</span>
        </button>
      </div>
    </header>
  )
}

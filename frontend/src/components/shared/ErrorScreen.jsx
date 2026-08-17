import { API_BASE_URL } from '../../services/api'

export function ErrorScreen({ error, onRetry }) {
  return (
    <main className="app-shell app-shell--center" role="alert" aria-live="assertive">
      <div className="error-screen-panel">
        <div className="error-screen-icon" aria-hidden="true">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </div>

        <div className="error-screen-content">
          <h2>Connection Error</h2>
          <p className="error-message">{error || 'Unable to connect to TrafficGuard backend API.'}</p>
          <div className="error-details">
            <span>Target API Base URL:</span>
            <code>{API_BASE_URL}</code>
          </div>
          <p className="error-hint">
            Ensure the FastAPI backend is running with: <code>python -m uvicorn app.main:app --reload</code>
          </p>

          {onRetry && (
            <button type="button" className="retry-button" onClick={onRetry}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M23 4v6h-6" />
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
              </svg>
              Retry Connection
            </button>
          )}
        </div>
      </div>
    </main>
  )
}

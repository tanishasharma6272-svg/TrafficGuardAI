export function UnavailableNotice({
  title,
  description,
  badge = 'UNAVAILABLE IN DEMO API',
  targetEndpoint,
  children,
}) {
  return (
    <div className="unavailable-notice">
      <div className="unavailable-notice-header">
        <div className="unavailable-notice-icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </div>
        <div>
          <span className="unavailable-badge">{badge}</span>
          <h4 className="unavailable-title">{title}</h4>
        </div>
      </div>

      <p className="unavailable-description">{description}</p>

      {targetEndpoint && (
        <div className="unavailable-endpoint">
          <span className="endpoint-label">Upcoming API:</span>
          <code className="endpoint-code">{targetEndpoint}</code>
        </div>
      )}

      {children && <div className="unavailable-body">{children}</div>}
    </div>
  )
}

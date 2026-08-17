import { useState } from 'react'
import { RiskPill } from '../shared/RiskPill'

export function DeploymentPanel({
  deploymentResult,
  loading,
  error,
  onExecute,
  selectedId,
  onSelectLocation,
}) {
  const [units, setUnits] = useState(3)
  const [radius, setRadius] = useState(2.0)
  const [minRiskLevel, setMinRiskLevel] = useState('High')

  const handleRun = (e) => {
    e?.preventDefault()
    onExecute({
      available_units: Number(units),
      coverage_radius_km: Number(radius),
      min_risk_level: minRiskLevel,
    })
  }

  const selectedUnits = deploymentResult?.selected_units || []
  const optimizedMetrics = deploymentResult?.optimized_metrics

  return (
    <section className="deployment-panel" aria-label="Police Deployment Intelligence">
      <div className="panel-header">
        <div>
          <div className="panel-title-with-badge">
            <h2 className="panel-title">Police Deployment Intelligence</h2>
            <span className="deployment-algo-badge">Greedy Coverage Optimizer</span>
          </div>
          <p className="panel-subtitle">Algorithmic patrol placement based on ML risk density</p>
        </div>
      </div>

      <div className="deployment-body">
        {/* Optimizer Control Form */}
        <form onSubmit={handleRun} className="deployment-controls-grid">
          <div className="control-group">
            <label htmlFor="avail-units-input" className="control-label">
              Available Patrol Units: <strong className="mono">{units}</strong>
            </label>
            <input
              id="avail-units-input"
              type="number"
              min="1"
              max="20"
              className="control-input mono"
              value={units}
              onChange={(e) => setUnits(Math.max(1, parseInt(e.target.value, 10) || 1))}
              disabled={loading}
            />
            <span className="control-hint">Deployable patrol unit count</span>
          </div>

          <div className="control-group">
            <label htmlFor="radius-slider" className="control-label">
              Patrol Coverage Radius: <strong className="mono">{radius} km</strong>
            </label>
            <input
              id="radius-slider"
              type="range"
              min="0.5"
              max="5.0"
              step="0.5"
              className="control-slider"
              value={radius}
              onChange={(e) => setRadius(parseFloat(e.target.value))}
              disabled={loading}
            />
            <span className="control-hint">Geodesic Haversine buffer</span>
          </div>

          <div className="control-group">
            <label htmlFor="risk-level-select" className="control-label">
              Optimization Target Severity
            </label>
            <select
              id="risk-level-select"
              className="control-select"
              value={minRiskLevel}
              onChange={(e) => setMinRiskLevel(e.target.value)}
              disabled={loading}
            >
              <option value="High">Critical + High Risk Nodes</option>
              <option value="Critical">Critical Risk Nodes Only</option>
            </select>
            <span className="control-hint">Eligibility threshold</span>
          </div>

          <div className="deployment-actions-row">
            <button
              type="submit"
              className="deploy-action-btn"
              disabled={loading}
              title="Execute greedy maximum coverage optimization"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="12 2 2 7 12 12 22 7 12 2" />
                <polyline points="2 17 12 22 22 17" />
                <polyline points="2 12 12 17 22 12" />
              </svg>
              <span>{loading ? 'Optimizing Placement…' : 'Execute Deployment Optimizer'}</span>
            </button>
          </div>
        </form>

        {error && (
          <div className="deployment-error-alert" role="alert">
            <span>Optimization error: {error}</span>
          </div>
        )}

        {/* Deployment Results Output */}
        {deploymentResult && (
          <div className="deployment-results-section">
            <div className="deployment-summary-bar">
              <span className="deployment-summary-stat">
                Placed <strong>{selectedUnits.length}</strong> of <strong>{deploymentResult.available_units}</strong> units
              </span>
              <span className="deployment-summary-coverage mono">
                {optimizedMetrics?.risk_coverage_percent}% Coverage
              </span>
            </div>

            <div className="deployment-units-list">
              {selectedUnits.map((unit) => {
                const isSelected = unit.location_id === selectedId
                return (
                  <div
                    key={unit.rank}
                    className={`deployment-unit-card ${isSelected ? 'is-selected' : ''}`}
                    onClick={() => onSelectLocation(unit.location_id)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => e.key === 'Enter' && onSelectLocation(unit.location_id)}
                  >
                    <div className="deployment-unit-rank mono">
                      #{unit.rank}
                    </div>

                    <div className="deployment-unit-info">
                      <div className="deployment-unit-name-row">
                        <strong className="deployment-unit-name">{unit.location_name}</strong>
                        <RiskPill level={unit.risk_level} size="small" />
                      </div>
                      <div className="deployment-unit-meta">
                        <span>Score: <strong className="mono">{unit.risk_score.toFixed(1)}</strong></span>
                        <span>·</span>
                        <span>Covers: <strong className="mono">{unit.covered_location_count} nodes</strong></span>
                      </div>
                    </div>

                    <button
                      type="button"
                      className="deployment-focus-btn"
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelectLocation(unit.location_id)
                      }}
                      title="Focus on map"
                      aria-label={`Focus on ${unit.location_name}`}
                    >
                      Focus
                    </button>
                  </div>
                )
              })}

              {selectedUnits.length === 0 && (
                <p className="empty-message">No eligible high-risk locations required deployment.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

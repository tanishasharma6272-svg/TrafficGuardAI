import { RiskPill } from '../shared/RiskPill'
import { RiskGauge } from './RiskGauge'
import { ShapExplanation } from './ShapExplanation'
import { formatCoordinates } from '../../utils/riskStats'

export function LocationDetails({
  selectedRisk,
  selectedLocation,
  selectedDetail,
  detailLoading,
  detailError,
  selectedShap,
  shapLoading,
  shapError,
}) {
  if (!selectedRisk && !selectedLocation) {
    return (
      <aside className="details-panel" aria-label="Selected Location Intelligence">
        <div className="panel-header">
          <div>
            <h2 className="panel-title">Location Intelligence</h2>
            <p className="panel-subtitle">Select an intersection from map, queue, or table</p>
          </div>
        </div>
        <div className="details-empty-state">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 22s-8-4.5-8-11.8A8 8 0 0 1 12 2a8 8 0 0 1 8 8.2c0 7.3-8 11.8-8 11.8z" />
            <circle cx="12" cy="10" r="3" />
          </svg>
          <p>No location selected. Click a pin on the map or select a row in the table.</p>
        </div>
      </aside>
    )
  }

  // Combine baseline location fields with detailed ML risk response if loaded
  const data = selectedDetail || {
    ...selectedLocation,
    ...selectedRisk,
  }

  const congestionPct = typeof data.congestion_ratio === 'number'
    ? (data.congestion_ratio * 100).toFixed(1)
    : typeof data.congestion === 'number'
      ? (data.congestion * 100).toFixed(1)
      : null

  return (
    <aside className="details-panel" aria-label="Selected Location Intelligence">
      <div className="panel-header">
        <div>
          <h2 className="panel-title">Location Intelligence</h2>
          <p className="panel-subtitle">ML risk assessment &amp; kinematic telemetry</p>
        </div>
        {detailLoading && (
          <span className="detail-syncing-badge">Syncing /api/ml/risk/{data.id}…</span>
        )}
      </div>

      <div className="details-content-scroll">
        <div className="selected-location-header">
          <div className="selected-title-group">
            <RiskPill level={data.risk_level} size="large" />
            <h3 className="selected-name">{data.name}</h3>
          </div>
          <div className="selected-meta-tags">
            <span className="coordinate-source-pill">
              Source: {data.coordinate_source || 'Verified GPS'}
            </span>
            {data.model_type && (
              <span className="model-provenance-pill mono">
                ML: {data.model_type}
              </span>
            )}
          </div>
        </div>

        <div className="gauge-and-core-grid">
          <RiskGauge score={data.risk_score} level={data.risk_level} />

          <div className="core-telemetry-grid">
            <div className="telemetry-item">
              <span className="telemetry-item-label">Stationed Police</span>
              <strong className="telemetry-item-val mono">{data.police_officers ?? '—'} units</strong>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-item-label">Observed Speed</span>
              <strong className="telemetry-item-val mono">
                {data.traffic_speed != null ? `${data.traffic_speed} km/h` : '—'}
              </strong>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-item-label">Free-Flow Speed</span>
              <strong className="telemetry-item-val mono">
                {data.free_flow_speed != null ? `${data.free_flow_speed} km/h` : '—'}
              </strong>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-item-label">Traffic Volume</span>
              <strong className="telemetry-item-val mono">
                {data.traffic_volume != null ? `${data.traffic_volume.toLocaleString()} veh/day` : '—'}
              </strong>
            </div>
          </div>
        </div>

        {detailError && (
          <div className="detail-error-alert" role="alert">
            <span>Warning: {detailError}</span>
          </div>
        )}

        {/* Kinematic & Physical Attributes Section */}
        <div className="telemetry-attributes-section">
          <h4 className="section-minor-title">Kinematic &amp; Physical Indicators</h4>
          <div className="attribute-grid">
            <div className="attribute-box">
              <span className="attr-label">Congestion Ratio</span>
              <strong className="attr-val mono">
                {congestionPct != null ? `${congestionPct}%` : '—'}
              </strong>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Speed Deficit</span>
              <strong className="attr-val mono">
                {data.speed_deficit != null ? `${data.speed_deficit.toFixed(1)} km/h` : '—'}
              </strong>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Pressure Index</span>
              <strong className="attr-val mono">
                {data.traffic_pressure_composite != null ? data.traffic_pressure_composite.toFixed(2) : '—'}
              </strong>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Incident Frequency</span>
              <strong className="attr-val mono">
                {data.incident_frequency != null ? `${data.incident_frequency}/10` : '—'}
              </strong>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Accident History</span>
              <strong className="attr-val mono">
                {data.accident_history != null ? `${data.accident_history}/10` : '—'}
              </strong>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Road Hazard Factor</span>
              <strong className="attr-val mono">
                {data.road_factor != null ? data.road_factor.toFixed(2) : '—'}
              </strong>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Population Friction</span>
              <strong className="attr-val mono">
                {data.population_factor != null ? data.population_factor.toFixed(2) : '—'}
              </strong>
            </div>
            <div className="attribute-box">
              <span className="attr-label">GPS Coordinates</span>
              <strong className="attr-val mono">
                {formatCoordinates(data.latitude, data.longitude)}
              </strong>
            </div>
          </div>
        </div>

        {/* Derived Kinematic Contributing Factors */}
        {data.contributing_factors && data.contributing_factors.length > 0 && (
          <div className="contributing-factors-section">
            <div className="factors-section-header">
              <h4 className="section-minor-title">Derived Kinematic Hazard Factors</h4>
              <span className="factors-method-pill">Physical Heuristic (Non-SHAP)</span>
            </div>
            <ul className="factors-list">
              {data.contributing_factors.map((factor, idx) => (
                <li key={idx} className="factor-list-item">
                  <span className="factor-bullet" />
                  <span className="factor-text">{factor}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* SHAP Explainability Subcomponent */}
        <ShapExplanation
          shapData={selectedShap}
          loading={shapLoading}
          error={shapError}
        />
      </div>
    </aside>
  )
}

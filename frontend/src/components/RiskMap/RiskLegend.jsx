import { useState } from 'react'
import { RISK_LEVELS } from '../../constants/risk'

export function RiskLegend({ totalMonitored, filteredCount, deploymentActive }) {
  const [showHeatmapNotice, setShowHeatmapNotice] = useState(false)

  return (
    <div className="map-legend-container">
      <div className="legend-pills">
        {RISK_LEVELS.map((level) => (
          <span key={level} className="legend-item">
            <span className={`legend-dot legend-dot--${level.toLowerCase()}`} />
            <span className="legend-text">{level}</span>
          </span>
        ))}
        {deploymentActive && (
          <span className="legend-item legend-item--deployment">
            <span className="legend-deploy-icon">🛡️</span>
            <span className="legend-text">Police Deployment Hub</span>
          </span>
        )}
      </div>

      <div className="legend-layer-toggle">
        <button
          type="button"
          className="layer-button layer-button--active"
          title="Marker Pins Layer (Active)"
        >
          Markers ({filteredCount}/{totalMonitored})
        </button>

        <button
          type="button"
          className="layer-button layer-button--disabled"
          onClick={() => setShowHeatmapNotice((prev) => !prev)}
          title="Heatmap layer abstraction (Awaiting backend density endpoint)"
        >
          Heatmap (Future)
        </button>
      </div>

      {showHeatmapNotice && (
        <div className="heatmap-popover">
          <strong>Heatmap Layer (Planned)</strong>
          <p>
            Continuous spatial risk density contours will activate once the backend heatmap model
            is deployed.
          </p>
        </div>
      )}
    </div>
  )
}

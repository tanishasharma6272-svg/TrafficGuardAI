import { useMemo, useState } from 'react'
import { RiskPill } from '../shared/RiskPill'
import { getHighRiskLocations } from '../../utils/riskStats'

export function HighRiskLocations({ risks, selectedId, onSelectLocation }) {
  const [sortKey, setSortKey] = useState('score') // 'score' | 'police'

  const highRiskItems = useMemo(() => {
    const items = getHighRiskLocations(risks)
    if (sortKey === 'police') {
      return [...items].sort((a, b) => b.police_officers - a.police_officers)
    }
    return items
  }, [risks, sortKey])

  return (
    <section className="high-risk-panel" aria-label="High Risk Priority Queue">
      <div className="panel-header">
        <div>
          <div className="panel-title-with-badge">
            <h2 className="panel-title">High-Risk Priority Queue</h2>
            <span className="priority-count-badge">{highRiskItems.length} Locations</span>
          </div>
          <p className="panel-subtitle">Critical &amp; High hazard intersections requiring deployment priority</p>
        </div>

        <div className="high-risk-sort-controls">
          <span className="sort-label">Sort:</span>
          <button
            type="button"
            className={`sort-pill ${sortKey === 'score' ? 'is-active' : ''}`}
            onClick={() => setSortKey('score')}
          >
            Risk Score
          </button>
          <button
            type="button"
            className={`sort-pill ${sortKey === 'police' ? 'is-active' : ''}`}
            onClick={() => setSortKey('police')}
          >
            Police Units
          </button>
        </div>
      </div>

      <div className="high-risk-list">
        {highRiskItems.map((item, index) => {
          const isSelected = item.id === selectedId
          return (
            <div
              key={item.id}
              className={`high-risk-card ${isSelected ? 'is-selected' : ''}`}
              onClick={() => onSelectLocation(item.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && onSelectLocation(item.id)}
            >
              <div className="high-risk-rank mono">#{index + 1}</div>

              <div className="high-risk-info">
                <div className="high-risk-name-row">
                  <strong className="high-risk-name">{item.name}</strong>
                  <RiskPill level={item.risk_level} size="small" />
                </div>
                <div className="high-risk-meta">
                  <span>Score: <strong className="mono">{item.risk_score.toFixed(1)}</strong></span>
                  <span>·</span>
                  <span>Police: <strong className="mono">{item.police_officers} units</strong></span>
                </div>
              </div>

              <button
                type="button"
                className="high-risk-inspect-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  onSelectLocation(item.id)
                }}
                title="Focus on map"
                aria-label={`Inspect ${item.name}`}
              >
                Inspect
              </button>
            </div>
          )
        })}

        {highRiskItems.length === 0 && (
          <p className="empty-message">No high-risk locations currently identified.</p>
        )}
      </div>
    </section>
  )
}

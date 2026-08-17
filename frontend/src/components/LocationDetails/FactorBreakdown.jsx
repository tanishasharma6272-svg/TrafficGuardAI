export function FactorBreakdown({ detail }) {
  if (!detail || !detail.contributing_factors) {
    return null
  }

  const {
    congestion,
    contributing_factors: {
      congestion_component = 0,
      incident_frequency_component = 0,
      accident_history_component = 0,
      road_factor_component = 0,
      traffic_population_component = 0,
    } = {},
  } = detail

  const factors = [
    {
      label: 'Traffic Congestion',
      weight: '30%',
      componentScore: congestion_component,
      maxWeightScore: 30,
      info: `Congestion Ratio: ${(congestion * 100).toFixed(1)}%`,
    },
    {
      label: 'Incident Frequency',
      weight: '25%',
      componentScore: incident_frequency_component,
      maxWeightScore: 25,
      info: `Monthly frequency: ${detail.incident_frequency}/10`,
    },
    {
      label: 'Accident History',
      weight: '20%',
      componentScore: accident_history_component,
      maxWeightScore: 20,
      info: `Historical severity index: ${detail.accident_history}/10`,
    },
    {
      label: 'Road Infrastructure',
      weight: '15%',
      componentScore: road_factor_component,
      maxWeightScore: 15,
      info: `Road factor: ${detail.road_factor.toFixed(2)}`,
    },
    {
      label: 'Traffic & Population',
      weight: '10%',
      componentScore: traffic_population_component,
      maxWeightScore: 10,
      info: `Vol: ${detail.traffic_volume.toLocaleString()} | Pop: ${detail.population_factor.toFixed(2)}`,
    },
  ]

  return (
    <div className="factor-breakdown-section">
      <div className="factor-section-header">
        <h4>Formula Weight Decomposition</h4>
        <span className="factor-badge">Rule-Engine Model</span>
      </div>

      <div className="factor-bars-list">
        {factors.map((f) => {
          const fillPct = (f.componentScore / f.maxWeightScore) * 100
          return (
            <div key={f.label} className="factor-bar-item">
              <div className="factor-bar-labels">
                <span className="factor-name">
                  {f.label} <span className="factor-weight-tag">({f.weight})</span>
                </span>
                <span className="factor-value mono">
                  +{f.componentScore.toFixed(1)} pts
                </span>
              </div>

              <div className="factor-track">
                <div
                  className="factor-fill"
                  style={{ width: `${Math.min(100, Math.max(0, fillPct))}%` }}
                />
              </div>

              <span className="factor-meta">{f.info}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

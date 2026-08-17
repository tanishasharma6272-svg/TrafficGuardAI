export function CoverageComparison({ deploymentResult }) {
  if (!deploymentResult) {
    return (
      <section className="coverage-comparison-panel" aria-label="Deployment Coverage Impact">
        <div className="panel-header">
          <div>
            <h2 className="panel-title">Deployment Coverage Impact</h2>
            <p className="panel-subtitle">Algorithmic police allocation network coverage</p>
          </div>
        </div>

        <div className="coverage-body">
          <div className="coverage-awaiting-card">
            <div className="coverage-awaiting-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 6v6l4 2" />
              </svg>
            </div>
            <div className="coverage-awaiting-text">
              <strong className="coverage-awaiting-title">Awaiting Optimizer Run</strong>
              <p className="coverage-awaiting-desc">
                Execute the Deployment Optimizer to compute eligible risk exposure,
                protected nodes, and network coverage ratios.
              </p>
            </div>
          </div>
        </div>
      </section>
    )
  }

  const {
    baseline_metrics = {},
    optimized_metrics = {},
    coverage_radius_km,
  } = deploymentResult

  const {
    eligible_high_risk_locations = 0,
    total_eligible_risk_score = 0,
  } = baseline_metrics

  const {
    covered_locations = 0,
    covered_risk_score = 0,
    risk_coverage_percent = 0,
    uncovered_risk_score = 0,
    uncovered_risk_percent = 0,
  } = optimized_metrics

  return (
    <section className="coverage-comparison-panel" aria-label="Deployment Coverage Impact">
      <div className="panel-header">
        <div>
          <div className="panel-title-with-badge">
            <h2 className="panel-title">Deployment Coverage Impact</h2>
            <span className="coverage-radius-pill mono">{coverage_radius_km} km Radius</span>
          </div>
          <p className="panel-subtitle">Network coverage achieved across eligible high-risk corridors</p>
        </div>
      </div>

      <div className="coverage-body">
        {/* Visual Progress Bar */}
        <div className="coverage-progress-section">
          <div className="coverage-progress-labels">
            <span className="progress-label-covered">
              Covered: <strong>{risk_coverage_percent}%</strong>
            </span>
            <span className="progress-label-uncovered">
              Uncovered: <strong>{uncovered_risk_percent}%</strong>
            </span>
          </div>
          <div className="coverage-progress-track">
            <div
              className="coverage-progress-fill coverage-progress-fill--covered"
              style={{ width: `${Math.min(100, Math.max(0, risk_coverage_percent))}%` }}
              title={`Covered: ${risk_coverage_percent}%`}
            />
            <div
              className="coverage-progress-fill coverage-progress-fill--uncovered"
              style={{ width: `${Math.min(100, Math.max(0, uncovered_risk_percent))}%` }}
              title={`Uncovered: ${uncovered_risk_percent}%`}
            />
          </div>
        </div>

        {/* Detailed Metrics Grid */}
        <div className="coverage-metrics-real-grid">
          <div className="coverage-stat-card">
            <span className="stat-card-label">Eligible Risk Exposure</span>
            <strong className="stat-card-val mono">{total_eligible_risk_score.toFixed(1)} pts</strong>
            <span className="stat-card-sub">{eligible_high_risk_locations} target nodes</span>
          </div>

          <div className="coverage-stat-card coverage-stat-card--highlight">
            <span className="stat-card-label">Covered Risk Exposure</span>
            <strong className="stat-card-val mono">{covered_risk_score.toFixed(1)} pts</strong>
            <span className="stat-card-sub">{covered_locations} nodes protected</span>
          </div>

          <div className="coverage-stat-card">
            <span className="stat-card-label">Risk Coverage Ratio</span>
            <strong className="stat-card-val stat-card-val--accent mono">{risk_coverage_percent}%</strong>
            <span className="stat-card-sub">Of eligible hazard mass</span>
          </div>

          <div className="coverage-stat-card">
            <span className="stat-card-label">Uncovered Risk Exposure</span>
            <strong className="stat-card-val mono">{uncovered_risk_score.toFixed(1)} pts</strong>
            <span className="stat-card-sub">{uncovered_risk_percent}% remaining</span>
          </div>
        </div>

        <p className="coverage-disclaimer">
          Geographic risk coverage metric — does not assert causal risk reduction.
        </p>
      </div>
    </section>
  )
}

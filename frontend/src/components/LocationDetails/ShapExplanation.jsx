export function ShapExplanation({ shapData, loading, error }) {
  if (loading) {
    return (
      <div className="shap-explanation-card">
        <div className="shap-card-header">
          <div className="shap-title-group">
            <h4 className="shap-title">SHAP Feature Attribution</h4>
            <span className="shap-badge shap-badge--syncing">Computing SHAP…</span>
          </div>
          <span className="shap-model-tag">TreeSHAP Explainer</span>
        </div>
        <div className="shap-loading-state">
          <div className="shap-skeleton-bar" style={{ width: '85%' }} />
          <div className="shap-skeleton-bar" style={{ width: '65%' }} />
          <div className="shap-skeleton-bar" style={{ width: '75%' }} />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="shap-explanation-card shap-explanation-card--error">
        <div className="shap-card-header">
          <div className="shap-title-group">
            <h4 className="shap-title">SHAP Feature Attribution</h4>
            <span className="shap-badge shap-badge--error">Unavailable</span>
          </div>
        </div>
        <p className="shap-error-text">
          Feature attribution notice: {error}
        </p>
      </div>
    )
  }

  if (!shapData) {
    return null
  }

  const {
    base_value,
    raw_prediction,
    risk_score,
    training_data_mode,
    top_positive_contributors = [],
    top_negative_contributors = [],
    feature_attributions = [],
  } = shapData

  // Find max absolute SHAP value for scaling bars proportionately
  const maxAbsShap = Math.max(
    ...feature_attributions.map((f) => Math.abs(f.shap_value || 0)),
    1.0,
  )

  return (
    <div className="shap-explanation-card">
      <div className="shap-card-header">
        <div className="shap-title-group">
          <h4 className="shap-title">SHAP Model Feature Attribution</h4>
          <span className="shap-badge">TreeSHAP</span>
        </div>
        <span className="shap-mode-pill mono">
          {training_data_mode || 'SYNTHETIC_DEVELOPMENT'}
        </span>
      </div>

      <p className="shap-caveat-notice">
        <strong>Attribution Caveat:</strong> SHAP model attribution — not causal evidence.
      </p>

      {/* Model Expected Baseline vs Actual Prediction */}
      <div className="shap-baseline-row">
        <div className="shap-stat-box">
          <span className="shap-stat-label">Model Base Value E[f(X)]</span>
          <strong className="shap-stat-val mono">
            {typeof base_value === 'number' ? base_value.toFixed(2) : '—'}
          </strong>
        </div>
        <div className="shap-stat-arrow">→</div>
        <div className="shap-stat-box">
          <span className="shap-stat-label">Model Prediction</span>
          <strong className="shap-stat-val shap-stat-val--accent mono">
            {typeof risk_score === 'number' ? risk_score.toFixed(2) : raw_prediction?.toFixed(2)}
          </strong>
        </div>
      </div>

      {/* Positive Contributors (Risk Increasing) */}
      {top_positive_contributors.length > 0 && (
        <div className="shap-group">
          <div className="shap-group-header">
            <span className="shap-group-title shap-group-title--pos">
              Risk-Increasing Factors (+ SHAP)
            </span>
            <span className="shap-group-sub">Adds to baseline risk</span>
          </div>
          <div className="shap-bars-list">
            {top_positive_contributors.map((item) => {
              const barWidth = Math.min(
                100,
                Math.max(8, (Math.abs(item.shap_value) / maxAbsShap) * 100),
              )
              return (
                <div key={item.feature_name} className="shap-bar-row">
                  <div className="shap-bar-info">
                    <span className="shap-feat-name" title={item.feature_name}>
                      {item.human_label || item.feature_name}
                    </span>
                    <span className="shap-feat-val mono">
                      Observed: {typeof item.feature_value === 'number' ? item.feature_value.toFixed(2) : item.feature_value}
                    </span>
                  </div>
                  <div className="shap-bar-track">
                    <div
                      className="shap-bar-fill shap-bar-fill--pos"
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                  <span className="shap-contrib-val shap-contrib-val--pos mono">
                    +{item.shap_value.toFixed(2)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Negative Contributors (Risk Mitigating) */}
      {top_negative_contributors.length > 0 && (
        <div className="shap-group">
          <div className="shap-group-header">
            <span className="shap-group-title shap-group-title--neg">
              Risk-Mitigating Factors (− SHAP)
            </span>
            <span className="shap-group-sub">Reduces baseline risk</span>
          </div>
          <div className="shap-bars-list">
            {top_negative_contributors.map((item) => {
              const barWidth = Math.min(
                100,
                Math.max(8, (Math.abs(item.shap_value) / maxAbsShap) * 100),
              )
              return (
                <div key={item.feature_name} className="shap-bar-row">
                  <div className="shap-bar-info">
                    <span className="shap-feat-name" title={item.feature_name}>
                      {item.human_label || item.feature_name}
                    </span>
                    <span className="shap-feat-val mono">
                      Observed: {typeof item.feature_value === 'number' ? item.feature_value.toFixed(2) : item.feature_value}
                    </span>
                  </div>
                  <div className="shap-bar-track">
                    <div
                      className="shap-bar-fill shap-bar-fill--neg"
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                  <span className="shap-contrib-val shap-contrib-val--neg mono">
                    {item.shap_value.toFixed(2)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
